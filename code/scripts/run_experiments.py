#!/usr/bin/env python3
import sys
try:
    import numpy as np  # noqa
except ImportError:
    sys.exit("ERROR: this script needs NumPy. Install it with:\n"
             "    pip3 install numpy matplotlib\n"
             "(figures and results/summary.txt are already shipped if you only need to read them.)")
"""Runs every experiment of the report and appends results to results/raw.jsonl.
Already-finished runs are skipped, so the script can be interrupted/resumed.
Usage:  python3 scripts/run_experiments.py [--only E1,E2] [--quick]
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "build", "pimsim")
OUT = os.path.join(ROOT, "results", "raw.jsonl")
SZ = os.path.join(ROOT, "results", "sizes")

# Paper ablation points (Fig.15) + two extra points we add for insight
CONFIGS = {
    "Base":       ["--engine=sw", "--dmap=loc"],
    "Base+H":     ["--engine=sw", "--dmap=mlp"],                    # extra (not in paper)
    "Base+D":     ["--engine=dce", "--sched=seq", "--dmap=loc"],
    "Base+D+H":   ["--engine=dce", "--sched=seq", "--dmap=mlp"],
    "Base+D+P":   ["--engine=dce", "--sched=alg1", "--dmap=loc"],  # extra (not in paper)
    "Base+D+H+P": ["--engine=dce", "--sched=alg1", "--dmap=mlp"],  # = PIM-MMU
}

def jobs(quick):
    J = []
    seeds = [1] if quick else [1, 2]
    sizes = [1, 2, 4, 8, 16, 32, 64]
    # E1: throughput vs transfer size (Fig.15a)
    for op in ["d2p", "p2d"]:
        for mb in sizes:
            for name, cfg in CONFIGS.items():
                if name == "Base+D+P" and mb > 64: continue
                for sd in (seeds if cfg[0] == "--engine=sw" and mb <= 16 else [1]):
                    J.append(("E1", name, [f"--op={op}", f"--total_mb={mb}", f"--seed={sd}"] + cfg))
    # E2: throughput vs number of PIM modules (DIMM = 2 ranks); 32 KB per DPU
    for op in ["d2p", "p2d"]:
        for (pc, pr) in [(1, 2), (2, 2), (4, 2), (4, 4)]:
            for name in ["Base", "Base+H", "Base+D", "Base+D+H", "Base+D+H+P"]:
                cfg = CONFIGS[name]
                for sd in (seeds if cfg[0] == "--engine=sw" else [1]):
                    J.append(("E2", name, [f"--op={op}", f"--pch={pc}", f"--pra={pr}", "--per_dpu_kb=32", f"--seed={sd}"] + cfg))
    # E3: DRAM bandwidth, locality- vs MLP-centric mapping (Fig.8)
    for stride in [1, 16]:
        for dm in ["loc", "mlp"]:
            J.append(("E3", f"{dm}-s{stride}", ["--op=stream", "--engine=dce", "--sched=alg1", f"--dmap={dm}",
                                                 f"--stride={stride}", "--total_mb=64"]))
    # E4: DRAM->DRAM memcpy throughput (Fig.14)
    for (dc, dr) in [(2, 2), (4, 2), (4, 4)]:
        for dm, name in [("loc", "Baseline"), ("mlp", "PIM-MMU")]:
            J.append(("E4", name, ["--op=memcpy", "--engine=sw", f"--dmap={dm}", f"--dch={dc}", f"--dra={dr}", "--total_mb=64"]))
    # E5: co-located compute-bound contenders (Fig.13a)
    for spin in [0, 8, 16, 24]:
        for sd in seeds:
            J.append(("E5", "Base", ["--op=d2p", "--total_mb=64", f"--spin={spin}", f"--seed={sd}"] + CONFIGS["Base"]))
        J.append(("E5", "PIM-MMU", ["--op=d2p", "--total_mb=64", f"--spin={spin}"] + CONFIGS["Base+D+H+P"]))
    # E6: per-channel write-throughput timelines (Fig.6 / Fig.12)
    for name in ["Base", "Base+H", "Base+D+H+P"]:
        J.append(("E6", name, ["--op=d2p", "--total_mb=64", "--seed=1",
                               f"--timeline={ROOT}/results/timeline_{name.replace('+', '_')}.csv"] + CONFIGS[name]))
    # E7: OS-scheduling sensitivity of the SW baseline
    for dm in ["loc", "mlp"]:
        for pol in ["rand", "seq", "inter"]:
            J.append(("E7", f"{dm}-{pol}", ["--op=d2p", "--total_mb=16", "--engine=sw", f"--dmap={dm}", f"--os={pol}"]))
    # E8 (extension): skewed per-DPU sizes
    VAR = {
        "SW-pad":        ["--engine=sw", "--dmap=loc", "--pad=1"],
        "MMU-pad":       ["--engine=dce", "--sched=alg1", "--dmap=mlp", "--pad=1"],
        "+Ragged":       ["--engine=dce", "--sched=alg1", "--dmap=mlp"],
        "+Ragged+LRF":   ["--engine=dce", "--sched=lrf", "--dmap=mlp"],
        "+Ragged+Place": ["--engine=dce", "--sched=alg1", "--dmap=mlp", "--place=1"],
        "+Ragged+Place+LRF": ["--engine=dce", "--sched=lrf", "--dmap=mlp", "--place=1"],
    }
    for op in ["d2p", "p2d"]:
        for d in ["uniform", "lognorm-0.5", "lognorm-1.0", "rmat"]:
            for name, cfg in VAR.items():
                if name == "SW-pad" and op == "p2d": continue
                J.append(("E8", name, [f"--op={op}", f"--sizes={SZ}/{d}.txt", f"--dist={d}"] + cfg))
    # E9 (extension): LRF in-flight cap sensitivity
    for d in ["lognorm-1.0", "rmat"]:
        for cap in [1, 2, 4, 6, 12, 32]:
            J.append(("E9", f"cap{cap}", ["--op=d2p", f"--sizes={SZ}/{d}.txt", f"--dist={d}", "--engine=dce",
                                          "--sched=lrf", "--dmap=mlp", "--place=1", f"--lrf_cap={cap}"]))
    return J

def parse(line):
    return {k: v for k, v in (t.split("=", 1) for t in line.split())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(SZ, exist_ok=True)
    for d in ["uniform", "lognorm-0.5", "lognorm-1.0", "rmat"]:
        f = f"{SZ}/{d}.txt"
        if not os.path.exists(f):
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_sizes.py"), f"--dist={d}",
                            "--mean_kb=16", f"--out={f}"], check=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["exp"], r["label"], " ".join(r["args"])))
    only = set(a.only.split(",")) if a.only else None
    J = [j for j in jobs(a.quick) if (only is None or j[0] in only)]
    for i, (exp, label, args) in enumerate(J):
        key = (exp, label, " ".join(args))
        if key in done: continue
        simargs = [x for x in args if not x.startswith("--dist=")]
        t0 = time.time()
        p = subprocess.run([SIM] + simargs, capture_output=True, text=True)
        if p.returncode != 0:
            print("FAILED", key, p.stderr); continue
        r = parse(p.stdout.strip().splitlines()[-1])
        r.update(exp=exp, label=label, args=args, wall_s=round(time.time() - t0, 1))
        for x in args:
            if x.startswith("--dist="): r["dist"] = x.split("=", 1)[1]
            if x.startswith("--spin="): r["spin"] = x.split("=", 1)[1]
            if x.startswith("--total_mb="): r["total_mb"] = x.split("=", 1)[1]
            if x.startswith("--lrf_cap="): r["lrf_cap"] = x.split("=", 1)[1]
        with open(OUT, "a") as f: f.write(json.dumps(r) + "\n")
        print(f"[{i+1}/{len(J)}] {exp} {label:18s} {r['GBps']:>8s} GB/s  ({r['wall_s']}s)  {' '.join(args)}", flush=True)

if __name__ == "__main__":
    main()
