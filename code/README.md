# PIM-MMU Reproduction — Group 14

Simulation reproduction of **PIM-MMU** (Lee, Hyun, Kim, Rhu, *MICRO 2024*) plus a
ragged-transfer extension. We model DRAM↔PIM transfers on **Ramulator**'s DDR4
engine and compare a software baseline against the paper's three components: the
Data Copy Engine (DCE), the PIM-aware scheduler (PIM-MS), and the heterogeneous
memory mapping (HetMap). The transfer model (`src/pimsim.cpp`) is ours; Ramulator
is touched by a 2-line patch. All real-hardware work (PIM kernels, driver,
power/area, end-to-end PrIM times) is left out, as the assignment requires.

## Layout

```
setup.sh      clone Ramulator @214f635, apply patch, build
Makefile      builds build/pimsim
src/          pimsim.cpp — the transfer model
patches/      the 2-line Ramulator patch
scripts/      run_experiments.py, plot.py, gen_sizes.py, verify_results.py
results/      raw.jsonl (all data points), summary.txt, sizes/, timeline_*.csv
figs/         generated figures
report/       report.tex
```

## Setup

Requires Linux (WSL2/VM on Windows), `g++` (C++14), `make`, `git`, and
`python3` with `numpy`+`matplotlib`.

```bash
./setup.sh
./build/pimsim --op=d2p --total_mb=4 --engine=dce --sched=alg1 --dmap=mlp   # smoke test, ~70 GB/s
```

`setup.sh` clones Ramulator at commit `214f635`, applies the patch, and builds.
The patch fires the write-completion callback (commented out upstream) and sets
the scheduler to FR-FCFS; the Makefile passes `-include cstdint` for newer GCC.

## Running

Each run prints one `key=value` line; `GBps` is the throughput, `useful_GBps`
counts only un-padded bytes.

```bash
# full PIM-MMU vs software baseline, DRAM->PIM, 16 MB
./build/pimsim --op=d2p --total_mb=16 --engine=dce --sched=alg1 --dmap=mlp
./build/pimsim --op=d2p --total_mb=16 --engine=sw  --dmap=loc

# extension on a skewed (R-MAT) partition
python3 scripts/gen_sizes.py --dist=rmat --mean_kb=16 --out=/tmp/rmat.txt
./build/pimsim --op=d2p --sizes=/tmp/rmat.txt --engine=dce --sched=lrf --dmap=mlp --place=1
```

Ablation → flags: `Base` = `--engine=sw --dmap=loc`; `+D` = `--engine=dce
--sched=seq --dmap=loc`; `+D+H` = `--sched=seq --dmap=mlp`; `+D+H+P` (full
PIM-MMU) = `--sched=alg1 --dmap=mlp`.

Main flags: `--op` (`d2p`/`p2d`/`memcpy`/`stream`), `--engine` (`sw`/`dce`),
`--sched` (`seq`/`alg1`/`lrf`), `--dmap` (`loc`/`mlp`), `--total_mb` /
`--sizes=FILE`, `--pad`, `--place`, `--pch`/`--pra`/`--dch`/`--dra`, `--debug`.

## Reproduce figures

```bash
python3 scripts/run_experiments.py    # E1–E9, resumable (~30–60 min); skips finished runs
python3 scripts/plot.py               # writes figs/*.png and results/summary.txt
```

`results/raw.jsonl` is shipped, so `plot.py` works without re-running. Experiments
E1–E7 reproduce the paper (throughput vs size, vs PIM modules, mapping, memcpy,
contention, timelines); E8–E9 evaluate the ragged-transfer extension.

## Verify

```bash
python3 scripts/verify_results.py     # checks results/raw.jsonl; reports "all invariants hold"
```

Note: for ragged transfers (`--sizes`, no `--pad`) `dram_rd` and `pim_wr` are
*not* equal — reads move only useful bytes while a write burst covers all 8 DPUs
of a bank unit, so `pim_wr/dram_rd` is the padding ratio. The verifier applies
this correct invariant; a naive `read == write` check will wrongly flag these runs.

## Report

```bash
cd report && pdflatex report.tex && pdflatex report.tex
```
