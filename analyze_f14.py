#!/usr/bin/env python3
"""F14: does a mention repel the same number, or anything sharing its digits?"""
import json
import sys

import numpy as np

LET = ("A", "B", "K", "Q")
OOR = [str(v) for v in range(20, 30)]


def main():
    for fn in sys.argv[1:]:
        rows = [json.loads(l) for l in open(fn)]
        D = {(r["prompt"], r["location"], r["mention"]): np.array(r["q"]) for r in rows}
        print("==", fn.split("/")[-1])
        for loc in ("assign_down", "prompt_nl"):
            res = {"same_1digit": [], "same_2digit": [], "d_to_1d": [], "dd_to_d_last": [], "dd_to_d_first": [],
                   "oor_to_lastdigit": []}
            fav = []
            for p in ("base", "uniform"):
                ctrl = np.mean([D[(p, loc, L)] for L in LET], 0)
                for m in range(20):
                    q = D[(p, loc, str(m))]
                    (res["same_1digit"] if m < 10 else res["same_2digit"]).append(q[m] - ctrl[m])
                    if ctrl[m] >= .05:
                        fav.append((ctrl[m], q[m]))
                    if m < 10:  # mention d → x = 10 + d (shares the last digit)
                        res["d_to_1d"].append(q[10 + m] - ctrl[10 + m])
                    else:  # mention 1d → x = d (last digit) and x = 1 (first digit)
                        res["dd_to_d_last"].append(q[m - 10] - ctrl[m - 10])
                        res["dd_to_d_first"].append(q[1] - ctrl[1])
                for m in range(20, 30):  # out-of-range 2d → in-range x = its last digit
                    q = D[(p, loc, str(m))]
                    res["oor_to_lastdigit"].append(q[m - 20] - ctrl[m - 20])
            rel = sum(b for _, b in fav) / sum(a for a, _ in fav) - 1 if fav else float("nan")
            print(f"  {loc:12s} favored-cell rel. change (same number) {rel:+.2f} (n={len(fav)})")
            for k, v in res.items():
                print(f"     {k:18s} mean Δ {np.mean(v):+.4f}  neg {np.mean(np.array(v) < 0):.2f}  n={len(v)}")
        base = D[("base", "base", None)]
        print("  base q(x):", {i: round(float(base[i]), 3) for i in np.argsort(-base)[:5]})


if __name__ == "__main__":
    main()
