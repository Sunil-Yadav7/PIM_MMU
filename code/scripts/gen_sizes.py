#!/usr/bin/env python3
"""Per-DPU transfer sizes (bytes) for the ragged-transfer extension.

uniform   : every DPU gets the same size (sanity check; ragged == padded)
lognorm-s : lognormal sizes with sigma s (mean fixed)
rmat      : SpMV-style row partition of an R-MAT (a,b,c,d = .57,.19,.19,.05)
            sparse matrix: rows are split into equal contiguous blocks (as in
            PrIM SpMV / BFS), bytes per DPU = nnz * 8 (4 B col idx + 4 B value)
"""
import argparse, numpy as np

def rmat_rowcounts(scale, ef, rng, a=.57, b=.19, c=.19):
    n, m = 1 << scale, ef << scale
    rows = np.zeros(m, dtype=np.int64)
    for bit in range(scale):
        r = rng.random(m)
        # row bit is 1 for quadrants c and d
        rows |= ((r >= a + b).astype(np.int64) << (scale - 1 - bit))
    return np.bincount(rows, minlength=n)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dist", required=True)
    p.add_argument("--ndpu", type=int, default=512)
    p.add_argument("--mean_kb", type=float, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    mean = a.mean_kb * 1024
    if a.dist == "uniform":
        s = np.full(a.ndpu, mean)
    elif a.dist.startswith("lognorm"):
        sig = float(a.dist.split("-")[1])
        s = rng.lognormal(0, sig, a.ndpu); s = s / s.mean() * mean
    elif a.dist == "rmat":
        rc = rmat_rowcounts(17, 16, rng)
        blocks = rc.reshape(a.ndpu, -1).sum(1).astype(float) * 8
        s = blocks / blocks.mean() * mean
    else:
        raise SystemExit("unknown dist")
    s = np.maximum(64, np.round(s / 64) * 64).astype(np.int64)
    np.savetxt(a.out, s, fmt="%d")
    print(f"{a.dist}: mean={s.mean()/1024:.1f}KB max={s.max()/1024:.1f}KB max/mean={s.max()/s.mean():.2f} "
          f"cv={s.std()/s.mean():.2f}")

if __name__ == "__main__":
    main()
