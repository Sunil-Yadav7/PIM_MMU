#!/usr/bin/env python3
import sys
try:
    import numpy as np  # noqa
except ImportError:
    sys.exit("ERROR: this script needs NumPy. Install it with:\n"
             "    pip3 install numpy matplotlib\n"
             "(figures and results/summary.txt are already shipped if you only need to read them.)")
"""Generates every figure/table of the report from results/raw.jsonl."""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "raw.jsonl"))]
FIG = os.path.join(ROOT, "figs"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .3, "grid.linestyle": ":"})
COL = {"Base": "#7f7f7f", "Base+H": "#c7c7c7", "Base+D": "#9ecae1", "Base+D+H": "#3182bd",
       "Base+D+P": "#fdae6b", "Base+D+H+P": "#d62728"}
ORDER = ["Base", "Base+D", "Base+D+H", "Base+D+H+P", "Base+H", "Base+D+P"]
OPN = {"d2p": "DRAM$\\rightarrow$PIM", "p2d": "PIM$\\rightarrow$DRAM"}
summary = []

def arg(r, k):
    for a in r["args"]:
        if a.startswith(f"--{k}="): return a.split("=", 1)[1]
    return None

def mean_by(rs, key):
    d = collections.defaultdict(list)
    for r in rs: d[key(r)].append(float(r["GBps"]))
    return {k: (np.mean(v), np.std(v)) for k, v in d.items()}

# ---------------- E1: throughput vs transfer size ----------------
e1 = [r for r in R if r["exp"] == "E1"]
if e1:
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    M = mean_by(e1, lambda r: (r["op"], int(arg(r, "total_mb")), r["label"]))
    allsizes = sorted({int(arg(r, "total_mb")) for r in e1})
    for ax, op in zip(axs, ["d2p", "p2d"]):
        sizes = [s for s in allsizes if s <= 64 and (op, s, "Base") in M and (op, s, "Base+D+H+P") in M]
        w = 0.13
        for i, cfg in enumerate(ORDER):
            xs, ys, es = [], [], []
            for j, s in enumerate(sizes):
                if (op, s, cfg) in M:
                    base = M[(op, s, "Base")][0]
                    xs.append(j + (i - 2.5) * w); ys.append(M[(op, s, cfg)][0] / base); es.append(M[(op, s, cfg)][1] / base)
            hatch = "///" if cfg in ("Base+H", "Base+D+P") else None
            ax.bar(xs, ys, w, yerr=es, color=COL[cfg], edgecolor="k", lw=.3, hatch=hatch, label=cfg, error_kw={"lw": .5})
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"{s}MB" for s in sizes])
        ax.set_title(f"({'a' if op=='d2p' else 'b'}) {OPN[op]}"); ax.set_ylabel("Throughput (norm. to Base)")
        for j, s in enumerate(sizes):
            ax.text(j - 2.5 * w, 0.05, f"{M[(op, s, 'Base')][0]:.1f}", fontsize=5, rotation=90, va="bottom", ha="center")
            if (op, s, "Base+D+H+P") in M:
                v = M[(op, s, "Base+D+H+P")]; ax.text(j + .5 * w, v[0] / M[(op, s, 'Base')][0] + .1, f"{v[0]:.0f}", fontsize=5, ha="center")
    axs[0].legend(ncol=3, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/e1_throughput_vs_size.pdf"); fig.savefig(f"{FIG}/e1_throughput_vs_size.png")
    for op in ["d2p", "p2d"]:
        sp = [M[(op, s, "Base+D+H+P")][0] / M[(op, s, "Base")][0] for s in sizes]
        summary.append(f"E1 {op}: PIM-MMU/Base speedup per size {dict(zip(sizes, np.round(sp, 2)))} mean={np.mean(sp):.2f}")
        for cfg in ORDER:
            vals = [M[(op, s, cfg)][0] for s in sizes if (op, s, cfg) in M]
            summary.append(f"   {cfg:11s} GB/s {np.round(vals, 1).tolist()}")

# ---------------- E2: throughput vs number of PIM modules ----------------
e2 = [r for r in R if r["exp"] == "E2"]
if e2:
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    M = mean_by(e2, lambda r: (r["op"], int(r["pch"]) * int(r["pra"]) // 2, r["label"]))
    dimms = sorted({int(r["pch"]) * int(r["pra"]) // 2 for r in e2})
    lab = {1: "1\n(1C,128)", 2: "2\n(2C,256)", 4: "4\n(4C,512)", 8: "8\n(4C,1024)"}
    for ax, op in zip(axs, ["d2p", "p2d"]):
        for cfg in ["Base", "Base+H", "Base+D", "Base+D+H", "Base+D+H+P"]:
            ys = [M[(op, d, cfg)][0] for d in dimms]; es = [M[(op, d, cfg)][1] for d in dimms]
            ax.errorbar(range(len(dimms)), ys, yerr=es, marker="o", ms=3, lw=1, color=COL[cfg], label=cfg,
                        ls="--" if cfg == "Base+H" else "-")
        peak = [min(4, d // (1 if d < 4 else d // 4) if False else min(d, 4)) * 19.2 for d in dimms]
        ax.plot(range(len(dimms)), [min(d, 4) * 19.2 for d in dimms], "k:", lw=.8, label="PIM-ch. peak")
        ax.set_xticks(range(len(dimms))); ax.set_xticklabels([lab[d] for d in dimms])
        ax.set_xlabel("# PIM DIMMs (channels, DPUs)"); ax.set_ylabel("Throughput (GB/s)")
        ax.set_title(f"({'a' if op=='d2p' else 'b'}) {OPN[op]}, 32 KB/DPU")
    axs[0].legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/e2_modules.pdf"); fig.savefig(f"{FIG}/e2_modules.png")
    for op in ["d2p", "p2d"]:
        for cfg in ["Base", "Base+H", "Base+D", "Base+D+H", "Base+D+H+P"]:
            summary.append(f"E2 {op} {cfg:11s} " + " ".join(f"{d}DIMM={M[(op,d,cfg)][0]:.1f}" for d in dimms))

# ---------------- E3: Fig.8 mapping ----------------
e3 = {r["label"]: float(r["GBps"]) for r in R if r["exp"] == "E3"}
e4 = [r for r in R if r["exp"] == "E4"]
if e3 and e4:
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 1.7))
    ax = axs[0]
    labels, vals, cols = [], [], []
    for st, nm in [(16, "Stride"), (1, "Seq.")]:
        for dm in ["loc", "mlp"]:
            labels.append(f"{nm}: {'Locality' if dm=='loc' else 'MLP'}-centric")
            vals.append(e3[f"{dm}-s{st}"] / e3[f"mlp-s{st}"]); cols.append("#d62728" if dm == "loc" else "k")
    ax.barh(range(4), vals, color=cols); ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    ax.axvline(0.30, color="tab:blue", ls="--", lw=.8); ax.text(0.32, 3.3, "paper: ~0.30", color="tab:blue", fontsize=6)
    ax.set_xlabel("DRAM BW (norm. to MLP-centric)"); ax.set_title("(a) Mapping (cf. paper Fig. 8)")
    for i, v in enumerate(vals): ax.text(v + .01, i, f"{v:.2f}", va="center", fontsize=6)
    summary.append("E3 " + " ".join(f"{k}={v:.1f}GB/s" for k, v in e3.items()))
    ax = axs[1]
    M = {(int(r["dch"]), int(r["dra"]), r["label"]): float(r["GBps"]) for r in e4}
    cfgs = sorted({(c, r) for c, r, _ in M}, key=lambda x: (x[0], x[1]))
    y, labels = 0, []
    for c, rk in cfgs:
        for nm, col in [("Baseline", "#7f7f7f"), ("PIM-MMU", "#d62728")]:
            v = M[(c, rk, nm)] / M[(c, rk, "Baseline")]
            ax.barh(y, v, color=col); ax.text(v + .05, y, f"{v:.1f}x ({M[(c,rk,nm)]:.0f} GB/s)", va="center", fontsize=6)
            labels.append(f"{c}C-{c*rk}R {nm}"); y += 1
    ax.set_yticks(range(y)); ax.set_yticklabels(labels); ax.set_xlim(0, 7)
    ax.set_xlabel("DRAM memcpy throughput (norm.)"); ax.set_title("(b) HetMap memcpy (cf. paper Fig. 14)")
    fig.tight_layout(); fig.savefig(f"{FIG}/e3e4_mapping.pdf"); fig.savefig(f"{FIG}/e3e4_mapping.png")
    summary.append("E4 " + " ".join(f"{c}C{c*r}R:{M[(c,r,'PIM-MMU')]/M[(c,r,'Baseline')]:.2f}x" for c, r in cfgs))

# ---------------- E5: contention ----------------
e5 = [r for r in R if r["exp"] == "E5"]
if e5:
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    d = collections.defaultdict(list)
    for r in e5: d[(r["label"], int(r["spin"]))].append(float(r["time_us"]))
    ref = np.mean(d[("Base", 0)])
    spins = sorted({k[1] for k in d})
    for nm, col in [("Base", "k"), ("PIM-MMU", "#d62728")]:
        ys = [np.mean(d[(nm, s)]) / ref for s in spins]; es = [np.std(d[(nm, s)]) / ref for s in spins]
        ax.errorbar(spins, ys, yerr=es, marker="s", ms=3, color=col, label=nm)
        summary.append(f"E5 {nm} norm latency " + " ".join(f"{s}:{y:.2f}" for s, y in zip(spins, ys)))
    ax.set_xticks(spins); ax.set_xlabel("# compute-bound CPU contenders"); ax.set_ylabel("DRAM$\\rightarrow$PIM latency (norm.)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/e5_contention.pdf"); fig.savefig(f"{FIG}/e5_contention.png")

# ---------------- E6: timelines ----------------
names = [("Base", "Base (SW, locality map)"), ("Base_H", "Base+H (SW, MLP map)"), ("Base_D_H_P", "PIM-MMU")]
if all(os.path.exists(f"{ROOT}/results/timeline_{n}.csv") for n, _ in names):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.8), sharey=True)
    for ax, (n, t) in zip(axs, names):
        a = np.genfromtxt(f"{ROOT}/results/timeline_{n}.csv", delimiter=",", names=True)
        t_us = a["t_us"]; chs = [c for c in a.dtype.names if c.startswith("ch")]
        ax.stackplot(t_us, *[a[c] for c in chs], labels=[c.upper() for c in chs],
                     colors=["#f0f0f0", "#bdbdbd", "#636363", "#252525"])
        ax.set_title(t); ax.set_xlabel("time ($\\mu$s)")
        imb = np.mean([a[c].max() / max(1e-9, a[c].mean()) for c in chs])
    axs[0].set_ylabel("PIM write GB/s per channel\n(stacked)"); axs[0].legend(ncol=2, frameon=False, fontsize=5.5, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/e6_timeline.pdf"); fig.savefig(f"{FIG}/e6_timeline.png")
    for n, t in names:
        a = np.genfromtxt(f"{ROOT}/results/timeline_{n}.csv", delimiter=",", names=True)
        chs = [c for c in a.dtype.names if c.startswith("ch")]
        X = np.vstack([a[c] for c in chs])[:, :-1]
        cv = np.mean(X.std(0) / np.maximum(1e-9, X.mean(0)))
        summary.append(f"E6 {n}: mean per-window coefficient of variation across channels = {cv:.2f}")

# ---------------- E7: OS policy ----------------
e7 = {r["label"]: float(r["GBps"]) for r in R if r["exp"] == "E7"}
if e7: summary.append("E7 " + " ".join(f"{k}={v:.1f}" for k, v in sorted(e7.items())))

# ---------------- E8: extension ----------------
e8 = [r for r in R if r["exp"] == "E8"]
if e8:
    VAR = ["SW-pad", "MMU-pad", "+Ragged", "+Ragged+LRF", "+Ragged+Place", "+Ragged+Place+LRF"]
    VC = ["#7f7f7f", "#d62728", "#fdd0a2", "#fd8d3c", "#74c476", "#006d2c"]
    dists = ["uniform", "lognorm-0.5", "lognorm-1.0", "rmat"]
    DL = {"uniform": "Uniform", "lognorm-0.5": "LogN $\\sigma$=0.5", "lognorm-1.0": "LogN $\\sigma$=1.0", "rmat": "R-MAT SpMV"}
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    M = {(r["op"], r["dist"], r["label"]): float(r["useful_GBps"]) for r in e8}
    op = "d2p"
    w = 0.13
    vs = [v for v in VAR if all((op, d, v) in M for d in dists)]
    for i, v in enumerate(vs):
        ys = [M[(op, d, v)] for d in dists]
        ax.bar(np.arange(4) + (i - (len(vs) - 1) / 2) * w, ys, w, color=VC[VAR.index(v)], edgecolor="k", lw=.3, label=v)
    ax.set_xticks(range(4)); ax.set_xticklabels([DL[d] for d in dists]); ax.set_yscale("log")
    ax.set_ylabel("Effective throughput (GB/s)"); ax.set_title(f"{OPN[op]}, 512 DPUs, mean 16 KB/DPU")
    for d in dists:
        summary.append(f"E8 {op} {d:12s} " + " ".join(f"{v}={M[(op,d,v)]:.2f}" for v in vs))
    # also report p2d where available
    for d in dists:
        row=" ".join(f"{v}={M[('p2d',d,v)]:.2f}" for v in VAR if ('p2d',d,v) in M)
        if row: summary.append(f"E8 p2d {d:12s} "+row)
    ax.legend(ncol=2, frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIG}/e8_extension.pdf"); fig.savefig(f"{FIG}/e8_extension.png")

e9 = [r for r in R if r["exp"] == "E9"]
if e9:
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    for d, col in [("lognorm-1.0", "tab:orange"), ("rmat", "tab:green")]:
        pts = sorted((int(r["lrf_cap"]), float(r["useful_GBps"])) for r in e9 if r["dist"] == d)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, color=col, label=d)
        summary.append(f"E9 {d} " + " ".join(f"cap{c}={v:.1f}" for c, v in pts))
    ax.set_xscale("log", base=2); ax.set_xlabel("LRF in-flight cap per bank unit (chunks)"); ax.set_ylabel("Eff. GB/s")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/e9_lrfcap.pdf"); fig.savefig(f"{FIG}/e9_lrfcap.png")

open(os.path.join(ROOT, "results", "summary.txt"), "w").write("\n".join(summary) + "\n")
print("\n".join(summary))
