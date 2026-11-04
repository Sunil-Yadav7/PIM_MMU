// pimsim.cpp -- DRAM<->PIM data-transfer model for PIM-MMU (MICRO'24) reproduction.
//
// The DRAM and PIM memory systems are modelled cycle-accurately by Ramulator's
// DDR4 controllers (FR-FCFS, open-row). Everything that *generates* traffic is
// modelled here:
//   * SW engine  : UPMEM-style multi-threaded AVX-512 transfer (baseline "Base").
//                  8 cores, per-core line-fill-buffer (LFB) limit, 8x8 transpose,
//                  OS round-robin scheduling with time-slice preemption.
//   * DCE engine : PIM-MMU Data Copy Engine. Scheduling policy:
//                    seq  -> descriptor-order DMA ("Base+D", "Base+D+H")
//                    alg1 -> PIM-MS, Algorithm 1 of the paper ("Base+D+H+P")
//                    lrf  -> our extension: largest-remaining-first PIM-MS
//   * Mapping    : DRAM side uses locality-centric (Fig.7a) or MLP-centric with
//                  XOR hashing (Fig.7b, = HetMap). PIM side is always
//                  ChRaBgBkRoCo (locality) as required by bank-level PIM.
// One simulation step == one DDR4-2400 memory-controller cycle (1.2 GHz).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Config.h"
#include "Controller.h"
#include "DDR4.h"
#include "DRAM.h"
#include "Request.h"
#include "Statistics.h"

using namespace std;
using namespace ramulator;

namespace ramulator { bool warmup_complete = true; }

static long g_clk = 0;
static const double MEM_GHZ = 1.2;  // DDR4-2400 command clock
static const int LINE = 64;

// ----------------------------------------------------------------------------
// Address mapping
// ----------------------------------------------------------------------------
enum Lv { CH = 0, RA = 1, BG = 2, BK = 3, RO = 4, CO = 5, NLV = 6 };

struct Mapping {
  vector<pair<int, int>> fields;  // (level, bits), LSB -> MSB, on line address
  bool xorhash = false;
  int cb = 0, bgb = 0, bkb = 0;
  vector<int> map(long line) const {
    vector<int> av(NLV, 0), sh(NLV, 0);
    for (auto& f : fields) {
      if (f.second == 0) continue;
      int v = int(line & ((1L << f.second) - 1));
      line >>= f.second;
      av[f.first] |= v << sh[f.first];
      sh[f.first] += f.second;
    }
    if (xorhash) {  // permutation-based interleaving [Zhang MICRO'00]
      int row = av[RO];
      av[CH] ^= row & ((1 << cb) - 1);
      av[BK] ^= (row >> cb) & ((1 << bkb) - 1);
      av[BG] ^= (row >> (cb + bkb)) & ((1 << bgb) - 1);
    }
    return av;
  }
};
static int lg2(int x) { int b = 0; while ((1 << b) < x) b++; assert((1 << b) == x); return b; }

// DDR4 8Gb x8 rank: 4 BG x 4 BK, 64K rows, 128 bursts/row (8 KB row)
static Mapping dram_locality(int nch, int nra) {  // Fig.7(a): Ch Ra Ro Bk Bg Ro Co Bg Co
  Mapping m; m.fields = {{CO, 3}, {BG, 1}, {CO, 4}, {RO, 3}, {BG, 1}, {BK, 2}, {RO, 13}, {RA, lg2(nra)}, {CH, lg2(nch)}};
  return m;
}
static Mapping dram_mlp(int nch, int nra) {  // Fig.7(b): Ro Bk Bg Ra Co Bg Ch Co + XOR
  Mapping m; m.fields = {{CO, 2}, {CH, lg2(nch)}, {BG, 1}, {CO, 5}, {RA, lg2(nra)}, {BG, 1}, {BK, 2}, {RO, 16}};
  m.xorhash = true; m.cb = lg2(nch); m.bgb = 2; m.bkb = 2;
  return m;
}
// UPMEM rank seen by the host MC: 8 chips, 8 banks/chip (modelled as DDR4 x16
// geometry: 2 BG x 4 BK, 64K rows x 8 KB = 4 GB/rank = 8 chips x 4 Gb).
static Mapping pim_locality(int nch, int nra) {  // ChRaBgBkRoCo
  Mapping m; m.fields = {{CO, 7}, {RO, 16}, {BK, 2}, {BG, 1}, {RA, lg2(nra)}, {CH, lg2(nch)}};
  return m;
}
static const int PIM_UNIT_SHIFT = 23;  // 2^23 lines (512 MB) per bank unit = 8 DPUs x 64 MB MRAM

// ----------------------------------------------------------------------------
// Memory side (a set of channels sharing one mapping)
// ----------------------------------------------------------------------------
struct Agent;
struct OutReq { long ready; long line; vector<int> av; bool write; Agent* owner; int tag; };

struct MemSide {
  string name;
  DDR4* spec;
  vector<Controller<DDR4>*> ctrls;
  Mapping map;
  int nch, nra;
  vector<deque<OutReq>> outq;
  vector<long> rd_lines, wr_lines;
  vector<long> last_row;  // per (ch*nbank+bank) last open row, -1 = none
  long row_changes = 0, accesses = 0;
  long window = 12000;  // timeline window (10 us)
  vector<vector<long>> timeline;  // [window][channel] lines moved
  MemSide(const string& nm, const string& org, int ch, int ra, int idbase, int qsize, Mapping mp)
      : name(nm), map(mp), nch(ch), nra(ra) {
    spec = new DDR4(org, "DDR4_2400R");
    spec->set_channel_number(ch);
    spec->set_rank_number(ra);
    Config cfg;
    cfg.set_core_num(1);
    for (int c = 0; c < ch; c++) {
      auto* chan = new DRAM<DDR4>(spec, DDR4::Level::Channel);
      chan->id = c;  (void)idbase;
      chan->regStats("");
      auto* ctl = new Controller<DDR4>(cfg, chan);
      ctl->readq.max = qsize;
      ctl->writeq.max = qsize;
      ctrls.push_back(ctl);
    }
    outq.resize(ch);
    rd_lines.assign(ch, 0);
    wr_lines.assign(ch, 0);
    last_row.assign(ch * 2 * 2 * 4, -1);  // ch * ra(2) * bg(2) * bk(4)
  }
  void track(const vector<int>& av) {
    int bank = ((av[CH] * nra + av[RA]) * 2 + av[BG]) * 4 + av[BK];
    if (bank < (int)last_row.size()) {
      accesses++;
      if (last_row[bank] != av[RO]) { row_changes++; last_row[bank] = av[RO]; }
    }
  }
  void account(int ch) {
    size_t w = g_clk / window;
    if (timeline.size() <= w) timeline.resize(w + 1, vector<long>(nch, 0));
    timeline[w][ch]++;
  }
  bool idle() const { for (auto& q : outq) if (!q.empty()) return false; return true; }
  void push(long line, bool write, Agent* owner, int tag, long ready) {
    OutReq r{ready, line, map.map(line), write, owner, tag};
    track(r.av);
    outq[r.av[CH]].push_back(std::move(r));
  }
  void drain();  // move ready requests into controller queues
  void tick() { for (auto* c : ctrls) c->tick(); }
};

static long g_writes_issued = 0, g_writes_made = 0;

struct Agent {
  int lreq = 4, lret = 4;  // one-way on-chip latency (mem cycles)
  deque<pair<long, int>> ret;  // read data returning to agent (time, tag)
  virtual ~Agent() {}
  virtual void on_read(int tag) = 0;
  virtual void on_waccept(int tag) = 0;
  virtual void tick() = 0;
  virtual bool done() = 0;
  void drain_returns() {
    while (!ret.empty() && ret.front().first <= g_clk) { int t = ret.front().second; ret.pop_front(); on_read(t); }
  }
};

void MemSide::drain() {
  for (int c = 0; c < nch; c++) {
    auto& q = outq[c];
    while (!q.empty() && q.front().ready <= g_clk) {
      OutReq& o = q.front();
      Agent* ow = o.owner; int tag = o.tag; MemSide* self = this; int ch = c;
      auto cb = [ow, tag, self, ch](Request& r) {
        if (r.type == Request::Type::READ) {
          self->rd_lines[ch]++; self->account(ch);
          ow->ret.push_back({g_clk + ow->lret, tag});
        } else {
          self->wr_lines[ch]++; self->account(ch);
          g_writes_issued++;
        }
      };
      Request req(o.av, o.write ? Request::Type::WRITE : Request::Type::READ, cb, 0);
      req.addr = o.line;
      if (!ctrls[c]->enqueue(req)) break;  // controller queue full -> back-pressure
      if (o.write) ow->on_waccept(tag);
      q.pop_front();
    }
  }
}

static MemSide* g_dram = nullptr;
static MemSide* g_pim = nullptr;

struct Addr { bool pim; long line; };

// ----------------------------------------------------------------------------
// Jobs: describe the work as units x chunks; a chunk = reads -> (transpose) -> writes
// ----------------------------------------------------------------------------
struct ChunkSpec { vector<Addr> rd, wr; };

struct Job {
  virtual ~Job() {}
  virtual int nunits() = 0;
  virtual int nchunks(int u) = 0;
  virtual void build(int u, int k, ChunkSpec& c) = 0;
  virtual int unit_channel(int u) = 0;
  virtual long payload_bytes() = 0;
  virtual long useful_bytes() { return payload_bytes(); }
};

// DRAM<->PIM transfer. 8 DPUs (one per chip, same bank index) form one "bank
// unit": a 64 B burst to the unit carries 8 B for each of the 8 DPUs, so data
// must be 8x8 byte-transposed (Fig.3).
struct XferJob : Job {
  bool to_pim;
  int pch, pra, bpr;  // PIM channels, ranks/ch, bank units per rank (=8)
  vector<long> dpu_lines;  // per-DPU payload in 64 B lines
  vector<long> dpu_off;    // offset of each DPU buffer in the packed DRAM array
  vector<long> unit_lines;
  vector<long> useful_lines;  // un-padded sizes (for effective throughput)
  long dram_base;
  int G = 1;  // access granularity: lines per DPU per chunk (min_access_granularity)
  XferJob(bool tp, int pc, int pr, const vector<long>& lines, long base, int g)
      : to_pim(tp), pch(pc), pra(pr), bpr(8), dpu_lines(lines), dram_base(base), G(g) {
    int nd = pc * pr * bpr * 8;
    assert((int)dpu_lines.size() == nd);
    dpu_off.resize(nd);
    long o = 0;
    for (int d = 0; d < nd; d++) { dpu_off[d] = o; o += dpu_lines[d]; }
    unit_lines.resize(nd / 8);
    for (int u = 0; u < nd / 8; u++) {
      long m = 0; for (int j = 0; j < 8; j++) m = max(m, dpu_lines[8 * u + j]);
      unit_lines[u] = m;
    }
  }
  int nunits() override { return (int)unit_lines.size(); }
  int nchunks(int u) override { return (int)((unit_lines[u] + G - 1) / G); }
  int unit_channel(int u) override { return u / (pra * bpr); }
  long payload_bytes() override { long s = 0; for (auto l : dpu_lines) s += l; return s * LINE; }
  long useful_bytes() override { if (useful_lines.empty()) return payload_bytes(); long s = 0; for (auto l : useful_lines) s += l; return s * LINE; }
  void build(int u, int k, ChunkSpec& c) override {
    c.rd.clear(); c.wr.clear();
    vector<Addr> dr, pm;
    long k0 = (long)k * G, k1 = min(k0 + G, unit_lines[u]);
    for (int j = 0; j < 8; j++) {  // G consecutive source lines of each of the 8 DPUs
      int d = 8 * u + j;
      for (long kk = k0; kk < k1; kk++)
        if (kk < dpu_lines[d]) dr.push_back({false, dram_base + dpu_off[d] + kk});
    }
    for (long kk = k0; kk < k1; kk++)  // 8 transposed bursts per source line index
      for (int w = 0; w < 8; w++) pm.push_back({true, ((long)u << PIM_UNIT_SHIFT) | (8L * kk + w)});
    if (to_pim) { c.rd = dr; c.wr = pm; } else { c.rd = pm; c.wr = dr; }
  }
};

// DRAM->DRAM memcpy split into nseg contiguous segments (Fig.14 microbenchmark)
struct MemcpyJob : Job {
  int nseg; long lines_per_seg; long src, dst;
  MemcpyJob(int n, long total_lines, long s, long d) : nseg(n), lines_per_seg(total_lines / n), src(s), dst(d) {}
  int nunits() override { return nseg; }
  int nchunks(int) override { return (int)(lines_per_seg / 8); }
  int unit_channel(int u) override { return u; }
  long payload_bytes() override { return (long)nseg * lines_per_seg * LINE; }
  void build(int u, int k, ChunkSpec& c) override {
    c.rd.clear(); c.wr.clear();
    for (int i = 0; i < 8; i++) {
      long l = (long)u * lines_per_seg + 8L * k + i;
      c.rd.push_back({false, src + l}); c.wr.push_back({false, dst + l});
    }
  }
};

// DRAM read stream (Fig.8): sequential or strided over a region, nseg streams
struct StreamJob : Job {
  int nseg; long lines_per_seg; long stride;
  StreamJob(int n, long total_lines, long st) : nseg(n), lines_per_seg(total_lines / n), stride(st) {}
  int nunits() override { return nseg; }
  int nchunks(int) override { return (int)(lines_per_seg / 8); }
  int unit_channel(int u) override { return u; }
  long payload_bytes() override { return (long)nseg * lines_per_seg * LINE; }
  void build(int u, int k, ChunkSpec& c) override {
    c.rd.clear(); c.wr.clear();
    long base = (long)u * lines_per_seg;
    for (int i = 0; i < 8; i++) {
      long idx = 8L * k + i;  // idx-th access of this stream
      long l;
      if (stride <= 1) l = idx;
      else { long per = lines_per_seg / stride; l = (idx % per) * stride + idx / per; }
      c.rd.push_back({false, base + l});
    }
  }
};

// ----------------------------------------------------------------------------
// Chunk bookkeeping shared by both engines
// ----------------------------------------------------------------------------
struct ChunkState {
  bool used = false; int unit = -1;
  ChunkSpec s; int rd_issued = 0, rd_done = 0, wr_issued = 0, wr_acc = 0; long tready = -1;
  bool complete() const { return rd_done == (int)s.rd.size() && wr_acc == (int)s.wr.size(); }
};

static void issue(Agent* a, const Addr& ad, bool write, int tag) {
  (ad.pim ? g_pim : g_dram)->push(ad.line, write, a, tag, g_clk + a->lreq);
}

// ----------------------------------------------------------------------------
// Software baseline: cores + OS scheduler
// ----------------------------------------------------------------------------
struct Task { int unit; int next_k = 0, done_k = 0, total = 0; bool spin = false; };

struct OS;
struct Core : Agent {
  OS* os; int id;
  int lfb_cap, window, width, ttr;
  int lfb = 0;
  vector<ChunkState> slot; deque<int> order;
  Task* task = nullptr;
  enum St { IDLE, RUN, DRAIN, SWITCH } st = IDLE;
  long slice_end = 0, switch_end = 0;
  Core(OS* o, int i, int lfbc, int w, int wd, int tt) : os(o), id(i), lfb_cap(lfbc), window(w), width(wd), ttr(tt) {
    slot.resize(w); lreq = 18; lret = 18;  // ~15 ns core<->IMC mesh latency each way
  }
  void on_read(int tag) override { slot[tag].rd_done++; lfb--; }
  void on_waccept(int tag) override { slot[tag].wr_acc++; lfb--; }
  bool done() override { return task == nullptr && order.empty(); }
  void tick() override;
};

struct OS {
  Job* job; vector<Task> tasks; deque<Task*> rq; vector<Core*> cores;
  long slice, tcs; int finished = 0, ntransfer = 0;
  Task* pick() { if (rq.empty()) return nullptr; Task* t = rq.front(); rq.pop_front(); return t; }
};

void Core::tick() {
  drain_returns();
  if (st == SWITCH && g_clk >= switch_end) st = IDLE;
  if (st == IDLE) {
    task = os->pick();
    if (!task) return;
    st = RUN; slice_end = g_clk + os->slice;
  }
  if (st == SWITCH) return;
  if (task->spin) {  // compute-bound contender: occupies the core for a full slice
    if (g_clk >= slice_end) { os->rq.push_back(task); task = nullptr; st = SWITCH; switch_end = g_clk + os->tcs; }
    return;
  }
  // transpose latency
  for (int s : order) { auto& c = slot[s]; if (c.tready < 0 && c.rd_done == (int)c.s.rd.size()) c.tready = g_clk + ttr; }
  // issue (oldest chunk first, program order inside a chunk: loads then stores)
  int budget = width;
  for (int s : order) {
    auto& c = slot[s];
    while (budget > 0 && lfb < lfb_cap) {
      if (c.tready >= 0 && g_clk >= c.tready && c.wr_issued < (int)c.s.wr.size()) {
        issue(this, c.s.wr[c.wr_issued], true, s); c.wr_issued++; lfb++; budget--;
      } else if (c.rd_issued < (int)c.s.rd.size()) {
        issue(this, c.s.rd[c.rd_issued], false, s); c.rd_issued++; lfb++; budget--;
      } else break;
    }
  }
  if (st == RUN && g_clk >= slice_end) st = DRAIN;  // preempted: finish in-flight work
  if (st == RUN && (int)order.size() < window && task->next_k < task->total) {
    int s = -1; for (int i = 0; i < window; i++) if (!slot[i].used) { s = i; break; }
    auto& c = slot[s];
    c = ChunkState(); c.used = true; c.unit = task->unit;
    os->job->build(task->unit, task->next_k++, c.s);
    g_writes_made += c.s.wr.size();
    order.push_back(s);
  }
  // retire
  while (!order.empty() && slot[order.front()].complete()) {
    slot[order.front()].used = false; order.pop_front(); task->done_k++;
  }
  if (task->done_k == task->total) {
    os->finished++; task = nullptr; st = SWITCH; switch_end = g_clk + os->tcs;
  } else if (st == DRAIN && order.empty()) {
    os->rq.push_back(task); task = nullptr; st = SWITCH; switch_end = g_clk + os->tcs;
  }
}

// ----------------------------------------------------------------------------
// PIM-MMU Data Copy Engine (+ PIM-MS)
// ----------------------------------------------------------------------------
struct DCE : Agent {
  Job* job; string policy;
  int nslots, ttr; double rate, credit = 0;
  vector<ChunkState> slot; deque<int> order;
  // per-unit progress
  vector<int> next_k, done_k, inflight;
  // policy state
  int seq_u = 0;
  int nch; vector<vector<int>> ch_units; vector<int> ch_cur; int rr = 0;
  int lrf_cap;
  vector<int> gorder; int gptr = 0;
  long completed_units = 0;
  DCE(Job* j, const string& pol, int window_lines, double r, int tt, int nchan, const vector<vector<int>>& cu, int cap)
      : job(j), policy(pol), nslots(window_lines / 8), ttr(tt), rate(r), nch(nchan), ch_units(cu), lrf_cap(cap) {
    lreq = 4; lret = 4;  // DCE sits next to the memory controller
    slot.resize(nslots);
    int n = job->nunits();
    next_k.assign(n, 0); done_k.assign(n, 0); inflight.assign(n, 0);
    ch_cur.assign(nch, 0);
  }
  void on_read(int tag) override { slot[tag].rd_done++; }
  void on_waccept(int tag) override { slot[tag].wr_acc++; }
  bool done() override { return completed_units == job->nunits(); }

  int pick_unit() {
    int n = job->nunits();
    if (policy == "seq") {  // descriptor order, like a conventional DMA engine
      while (seq_u < n && next_k[seq_u] >= job->nchunks(seq_u)) seq_u++;
      return seq_u < n ? seq_u : -1;
    }
    if (policy == "lrf") {
      // Extension: global largest-remaining-first (water filling). Every unit may
      // hold at most lrf_cap chunks in the data buffer, so the heaviest bank
      // units start early and stay saturated while lighter units fill the rest.
      // Ties are broken in Algorithm-1 interleaved order (channel fastest) with a
      // rotating start, so equal-sized units still spread over channels/banks.
      if (gorder.empty())
        for (size_t i = 0; i < ch_units[0].size(); i++)
          for (int c = 0; c < nch; c++) if (i < ch_units[c].size()) gorder.push_back(ch_units[c][i]);
      int m = (int)gorder.size(), best = -1, bpos = 0; long bestrem = 0;
      for (int i = 0; i < m; i++) {
        int pos = (gptr + i) % m, u = gorder[pos];
        long rem = job->nchunks(u) - next_k[u];
        if (rem > bestrem && inflight[u] < lrf_cap) { bestrem = rem; best = u; bpos = pos; }
      }
      if (best >= 0) gptr = (bpos + 1) % m;
      return best;
    }
    for (int t = 0; t < nch; t++) {  // "#do-parallel channel": round-robin over channels
      int ch = (rr + t) % nch;
      auto& us = ch_units[ch];
      int m = (int)us.size(), best = -1;
      if (policy == "alg1") {  // Algorithm 1: one access granule per unit, bk->ra->bg order
        for (int i = 0; i < m; i++) {
          int u = us[(ch_cur[ch] + i) % m];
          if (next_k[u] < job->nchunks(u)) { best = u; ch_cur[ch] = (ch_cur[ch] + i + 1) % m; break; }
        }
      }
      if (best >= 0) { rr = (ch + 1) % nch; return best; }
    }
    return -1;
  }

  void tick() override {
    drain_returns();
    credit = min(credit + rate, rate + 1.0);
    for (int s : order) { auto& c = slot[s]; if (c.tready < 0 && c.rd_done == (int)c.s.rd.size()) c.tready = g_clk + ttr; }
    for (int s : order) {  // writes (oldest first) have priority, then reads
      auto& c = slot[s];
      while (credit >= 1 && c.tready >= 0 && g_clk >= c.tready && c.wr_issued < (int)c.s.wr.size()) {
        issue(this, c.s.wr[c.wr_issued++], true, s); credit -= 1;
      }
    }
    for (int s : order) {
      auto& c = slot[s];
      while (credit >= 1 && c.rd_issued < (int)c.s.rd.size()) { issue(this, c.s.rd[c.rd_issued++], false, s); credit -= 1; }
    }
    // admit new chunks into the data buffer
    while (credit >= 1 && (int)order.size() < nslots) {
      int u = pick_unit();
      if (u < 0) break;
      int s = -1; for (int i = 0; i < nslots; i++) if (!slot[i].used) { s = i; break; }
      auto& c = slot[s];
      c = ChunkState(); c.used = true; c.unit = u;
      job->build(u, next_k[u]++, c.s); inflight[u]++;
      g_writes_made += c.s.wr.size();
      order.push_back(s);
      while (credit >= 1 && c.rd_issued < (int)c.s.rd.size()) { issue(this, c.s.rd[c.rd_issued++], false, s); credit -= 1; }
    }
    // retire (out of order is fine: chunks are independent)
    for (auto it = order.begin(); it != order.end();) {
      auto& c = slot[*it];
      if (c.complete()) {
        c.used = false; inflight[c.unit]--; done_k[c.unit]++;
        if (done_k[c.unit] == job->nchunks(c.unit)) completed_units++;
        it = order.erase(it);
      } else ++it;
    }
  }
};

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------
struct Args {
  map<string, string> kv;
  string get(const string& k, const string& d) const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
  long geti(const string& k, long d) const { auto it = kv.find(k); return it == kv.end() ? d : atol(it->second.c_str()); }
  double getf(const string& k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : atof(it->second.c_str()); }
};

static void usage() {
  fprintf(stderr,
    "pimsim --op=d2p|p2d|memcpy|stream --engine=sw|dce --dmap=loc|mlp --sched=seq|alg1|lrf\n"
    "       [--pch=4 --pra=2 --dch=4 --dra=2] [--total_mb=16 | --per_dpu_kb=K | --sizes=file]\n"
    "       [--pad=0|1] [--os=rand|seq|inter --seed=1 --spin=0 --slice_us=1500 --cores=8 --lfb=12]\n"
    "       [--window_lines=256 --lrf_cap=2 --stride=1 --timeline=file]\n");
}

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    string s = argv[i];
    if (s == "-h" || s == "--help") { usage(); return 0; }
    if (s.rfind("--", 0) != 0) { usage(); return 1; }
    auto eq = s.find('=');
    if (eq == string::npos) a.kv[s.substr(2)] = "1"; else a.kv[s.substr(2, eq - 2)] = s.substr(eq + 1);
  }
  string op = a.get("op", "d2p"), engine = a.get("engine", "sw"), dmap = a.get("dmap", "loc"), sched = a.get("sched", "alg1");
  int pch = a.geti("pch", 4), pra = a.geti("pra", 2), dch = a.geti("dch", 4), dra = a.geti("dra", 2);
  int qsize = a.geti("qsize", 64);
  unsigned seed = a.geti("seed", 1);

  Mapping dm = dmap == "mlp" ? dram_mlp(dch, dra) : dram_locality(dch, dra);
  g_dram = new MemSide("dram", "DDR4_8Gb_x8", dch, dra, 0, qsize, dm);
  g_pim = new MemSide("pim", "DDR4_8Gb_x16", pch, pra, 100, qsize, pim_locality(pch, pra));

  // ---- job ----
  unique_ptr<Job> job;
  int ndpu = pch * pra * 8 * 8;
  if (op == "d2p" || op == "p2d") {
    vector<long> lines(ndpu);
    if (a.kv.count("sizes")) {  // per-DPU sizes in bytes, one per line
      ifstream f(a.get("sizes", ""));
      for (int d = 0; d < ndpu; d++) { long b; if (!(f >> b)) { fprintf(stderr, "sizes file too short\n"); return 1; } lines[d] = (b + LINE - 1) / LINE; }
    } else {
      long per;
      if (a.kv.count("per_dpu_kb")) per = a.geti("per_dpu_kb", 32) * 1024;
      else per = (long)(a.getf("total_mb", 16) * 1024 * 1024) / ndpu;
      for (auto& l : lines) l = max(1L, per / LINE);
    }
    if (a.geti("place", 0)) {
      // Extension: size-aware partition placement (runtime keeps a permutation).
      // 1) sort partitions by size, 2) pack 8 similar ones per bank unit (one
      // burst feeds all 8 DPUs, so a unit costs max-of-8), 3) LPT-assign units
      // to channels so every channel carries the same amount of traffic.
      vector<long> srt = lines; sort(srt.rbegin(), srt.rend());
      int nu = ndpu / 8, upc = nu / pch;
      vector<long> load(pch, 0); vector<int> used(pch, 0);
      vector<long> out(ndpu, 0);
      for (int g = 0; g < nu; g++) {  // groups arrive in descending size order
        int c = -1;
        for (int k = 0; k < pch; k++) if (used[k] < upc && (c < 0 || load[k] < load[c])) c = k;
        int u = c * upc + used[c]++;
        load[c] += srt[8 * g];
        for (int j = 0; j < 8; j++) out[8 * u + j] = srt[8 * g + j];
      }
      lines = out;
    }
    vector<long> useful = lines;
    if (a.geti("pad", 0)) { long m = *max_element(lines.begin(), lines.end()); for (auto& l : lines) l = m; }
    auto* xj = new XferJob(op == "d2p", pch, pra, lines, 0, a.geti("gran", 1));
    xj->useful_lines = useful;
    job.reset(xj);
  } else if (op == "memcpy") {
    long total = (long)(a.getf("total_mb", 64) * 1024 * 1024) / LINE;
    job.reset(new MemcpyJob(a.geti("cores", 8), total, 0, 1L << 24 /*1 GB away*/));
  } else if (op == "stream") {
    long total = (long)(a.getf("total_mb", 64) * 1024 * 1024) / LINE;
    job.reset(new StreamJob(a.geti("streams", 8), total, a.geti("stride", 1)));
  } else { usage(); return 1; }

  // ---- engine ----
  vector<Agent*> agents;
  OS os;
  unique_ptr<DCE> dce;
  int ncores = a.geti("cores", 8);
  if (engine == "sw") {
    os.job = job.get();
    os.slice = (long)(a.getf("slice_us", 1500) * 1e3 * MEM_GHZ);
    os.tcs = (long)(a.getf("ctxsw_us", 2.0) * 1e3 * MEM_GHZ);
    int n = job->nunits();
    os.tasks.resize(n + a.geti("spin", 0));
    for (int u = 0; u < n; u++) { os.tasks[u].unit = u; os.tasks[u].total = job->nchunks(u); }
    for (size_t i = n; i < os.tasks.size(); i++) { os.tasks[i].spin = true; os.tasks[i].unit = -1; }
    vector<Task*> order;
    for (auto& t : os.tasks) order.push_back(&t);
    string pol = a.get("os", "rand");
    if (pol == "rand") { mt19937 rng(seed); shuffle(order.begin(), order.end(), rng); }
    else if (pol == "inter") {  // thread i -> channel i % C (Fig.5c ideal)
      stable_sort(order.begin(), order.end(), [&](Task* x, Task* y) {
        auto key = [&](Task* t) { if (t->spin) return 1L << 40; int c = job->unit_channel(t->unit); int per = n / max(1, (op == "d2p" || op == "p2d") ? pch : n); return (long)(t->unit % max(1, per)) * 64 + c; };
        return key(x) < key(y); });
    }
    for (auto* t : order) os.rq.push_back(t);
    for (int c = 0; c < ncores; c++) {
      Core* core = new Core(&os, c, a.geti("lfb", 12), a.geti("cwin", 4), a.geti("width", 4), a.geti("ttr", 6));
      os.cores.push_back(core); agents.push_back(core);
    }
    os.ntransfer = n;
  } else {
    int n = job->nunits();
    vector<vector<int>> cu;
    if (op == "d2p" || op == "p2d") {  // Algorithm 1 order: bk -> ra -> bg within each channel
      cu.resize(pch);
      for (int c = 0; c < pch; c++)
        for (int bk = 0; bk < 4; bk++) for (int ra = 0; ra < pra; ra++) for (int bg = 0; bg < 2; bg++)
          cu[c].push_back(c * pra * 8 + ra * 8 + bg * 4 + bk);
    } else { cu.resize(n); for (int u = 0; u < n; u++) cu[u].push_back(u); }
    dce.reset(new DCE(job.get(), sched, a.geti("window_lines", 256) / max(1L, a.geti("gran", 1)), a.getf("issue_rate", 8.0 / 3.0), a.geti("ttr", 3),
                      (int)cu.size(), cu, a.geti("lrf_cap", 6)));
    agents.push_back(dce.get());
  }

  // ---- simulate ----
  long maxcyc = a.geti("max_cycles", 2000000000L);
  auto all_done = [&]() {
    if (engine == "sw") { if (os.finished < os.ntransfer) return false; }
    else if (!dce->done()) return false;
    for (auto* ag : agents) if (!ag->ret.empty()) return false;
    return g_dram->idle() && g_pim->idle() && g_writes_issued == g_writes_made;
  };
  while (true) {
    for (auto* ag : agents) ag->tick();
    g_dram->drain(); g_pim->drain();
    g_dram->tick(); g_pim->tick();
    g_clk++; Stats::curTick++;
    if ((g_clk & 1023) == 0 && all_done()) break;
    if (g_clk > maxcyc) { fprintf(stderr, "max cycles reached\n"); break; }
  }
  double secs = g_clk / (MEM_GHZ * 1e9);
  long payload = job->payload_bytes();
  long drl = 0, dwl = 0, prl = 0, pwl = 0;
  for (int c = 0; c < dch; c++) { drl += g_dram->rd_lines[c]; dwl += g_dram->wr_lines[c]; }
  for (int c = 0; c < pch; c++) { prl += g_pim->rd_lines[c]; pwl += g_pim->wr_lines[c]; }
  printf("op=%s engine=%s dmap=%s sched=%s pch=%d pra=%d dch=%d dra=%d ndpu=%d payload_B=%ld cycles=%ld time_us=%.3f "
         "GBps=%.3f useful_GBps=%.3f dram_rd_GBps=%.3f dram_wr_GBps=%.3f pim_rd_GBps=%.3f pim_wr_GBps=%.3f seed=%u\n",
         op.c_str(), engine.c_str(), dmap.c_str(), engine == "sw" ? a.get("os", "rand").c_str() : sched.c_str(),
         pch, pra, dch, dra, ndpu, payload, g_clk, secs * 1e6, payload / secs / 1e9, job->useful_bytes() / secs / 1e9,
         drl * 64 / secs / 1e9, dwl * 64 / secs / 1e9, prl * 64 / secs / 1e9, pwl * 64 / secs / 1e9, seed);
  if (a.kv.count("debug")) {
    auto dump = [](MemSide* m) {
      // approx: use serving to infer; print lines moved per channel
      fprintf(stderr, "[%s] per-ch rd:", m->name.c_str());
      for (int c = 0; c < m->nch; c++) fprintf(stderr, " %ld", m->rd_lines[c]);
      fprintf(stderr, "  wr:");
      for (int c = 0; c < m->nch; c++) fprintf(stderr, " %ld", m->wr_lines[c]);
      fprintf(stderr, "\n");
    };
    dump(g_dram); dump(g_pim);
    fprintf(stderr, "[dram] row_changes=%ld accesses=%ld (%.1f%% new-row)\n",
            g_dram->row_changes, g_dram->accesses, 100.0*g_dram->row_changes/max(1L,g_dram->accesses));
    fprintf(stderr, "[pim]  row_changes=%ld accesses=%ld (%.1f%% new-row)\n",
            g_pim->row_changes, g_pim->accesses, 100.0*g_pim->row_changes/max(1L,g_pim->accesses));
  }
  if (a.kv.count("timeline")) {
    ofstream f(a.get("timeline", ""));
    MemSide* side = (op == "p2d" || op == "memcpy") ? g_dram : g_pim;  // destination side
    f << "t_us";
    for (int c = 0; c < side->nch; c++) f << ",ch" << c;
    f << "\n";
    for (size_t w = 0; w < side->timeline.size(); w++) {
      f << (w + 0.5) * side->window / (MEM_GHZ * 1e3);
      for (int c = 0; c < side->nch; c++) f << "," << side->timeline[w][c] * 64.0 / (side->window / (MEM_GHZ * 1e9)) / 1e9;
      f << "\n";
    }
  }
  return 0;
}
