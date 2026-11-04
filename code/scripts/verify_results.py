#!/usr/bin/env python3
"""Validate results/raw.jsonl against the model's invariants.

Runs with the standard library only (no NumPy), so it works in a minimal
environment. It checks every row for sane payload/cycles, and applies the
*correct* read/write balance invariant — which differs for fixed-size vs ragged
transfers (see README section 10).

Usage:  python3 scripts/verify_results.py [results/raw.jsonl]
Exit status is non-zero if any genuine inconsistency is found.
"""
import json, os, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw.jsonl")

def num(r, k):
    return float(r.get(k, 0) or 0)

def main():
    rows = [json.loads(l) for l in open(PATH) if l.strip()]
    n = len(rows)
    sane_fail, bal_fail, ragged_ok = [], [], 0
    for i, r in enumerate(rows):
        tag = f"row {i} exp={r.get('exp')} op={r.get('op')} label={r.get('label')} dist={r.get('dist')}"
        # 1. basic sanity: positive payload, cycles, throughput
        if num(r, "payload_B") <= 0 or num(r, "cycles") <= 0 or num(r, "GBps") <= 0:
            sane_fail.append(tag)
            continue
        op = r.get("op")
        args = r.get("args", [])
        drd, dwr = num(r, "dram_rd_GBps"), num(r, "dram_wr_GBps")
        prd, pwr = num(r, "pim_rd_GBps"), num(r, "pim_wr_GBps")
        # A transfer is "ragged" (per-DPU sizes may differ, no padding) iff it was
        # driven by a --sizes file and --pad=1 was NOT given. This covers E8 and E9.
        has_sizes = any(a.startswith("--sizes=") for a in args)
        pad_flag = any(a == "--pad=1" for a in args)
        is_ragged = has_sizes and not pad_flag
        if op in ("d2p", "p2d"):
            # For d2p: DRAM is read, PIM is written. For p2d: PIM is read, DRAM is written.
            # The PIM side always carries the per-bank-unit padding; the DRAM side is the
            # useful (un-padded) side. So the PIM side >= the DRAM side on ragged input.
            if op == "d2p":
                dram_side, pim_side = drd, pwr      # read, write
            else:
                dram_side, pim_side = dwr, prd      # write, read
            if is_ragged:
                # correct invariant: DRAM side == useful payload; PIM side >= DRAM side
                if dram_side <= 0 or pim_side + 1e-9 < dram_side * 0.999:
                    bal_fail.append(tag + f"  (ragged: dram={dram_side:.2f} pim={pim_side:.2f}, expect pim>=dram)")
                else:
                    ragged_ok += 1
            else:
                # fixed-size / padded: both sides move the same bytes, equal within 1%
                if abs(dram_side - pim_side) > 0.01 * max(dram_side, pim_side, 1):
                    bal_fail.append(tag + f"  (fixed: dram={dram_side:.2f} pim={pim_side:.2f}, expect equal)")
        elif op == "memcpy":
            if abs(drd - dwr) > 0.01 * max(drd, dwr, 1):
                bal_fail.append(tag + f"  (memcpy: rd={drd:.2f} wr={dwr:.2f})")

    print(f"checked {n} rows")
    print(f"  sanity failures (payload/cycles/throughput <= 0): {len(sane_fail)}")
    print(f"  balance failures (genuine): {len(bal_fail)}")
    print(f"  ragged rows with correct read<=write asymmetry: {ragged_ok}")
    for t in sane_fail[:10]:
        print("   SANE  ", t)
    for t in bal_fail[:10]:
        print("   BAL   ", t)
    ok = not sane_fail and not bal_fail
    print("\nRESULT:", "all invariants hold" if ok else "INCONSISTENCIES FOUND")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
