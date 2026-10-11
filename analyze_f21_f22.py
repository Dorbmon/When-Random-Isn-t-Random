#!/usr/bin/env python3
"""F21 (y avoids x: do the same heads do it?) and F22 (first vs last digit)."""
import json
import sys

import numpy as np


def f21(path):
    rows = [json.loads(l) for l in open(path)]
    def pyx(key):
        return np.mean([r[key][r["x"]] if key != "q_random" else np.mean([q[r["x"]] for q in r["q_random"]]) for r in rows])
    base = np.mean([1 / r["n"] for r in rows])
    out = {k: float(pyx(k)) for k in ("q_none", "q_cand", "q_full", "q_random")}
    out["uniform"] = float(base)
    by_n = {}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        by_n[n] = {k: float(np.mean([r[k][r["x"]] if k != "q_random" else np.mean([q[r["x"]] for q in r["q_random"]]) for r in rs]))
                   for k in ("q_none", "q_cand", "q_full", "q_random")}
    print(f"F21 {rows[0]['model']}: P(y=x) none {out['q_none']:.4f} | candidate heads KO {out['q_cand']:.4f} | "
          f"random heads KO {out['q_random']:.4f} | all heads KO (x token blocked) {out['q_full']:.4f} | uniform {base:.3f}")
    for n, v in by_n.items():
        print(f"     n={n}: " + " ".join(f"{k[2:]} {x:.4f}" for k, x in v.items()))
    return out


def f22(path):
    rows = [json.loads(l) for l in open(path)]
    v = rows[0]["target"]; u = v % 10
    let = [r for r in rows if r["mention"] in ("A", "B", "K", "Q")]
    print(f"F22 {rows[0]['model']} target {v} (P1 = first digit '1', P2 = second digit '{u}')")
    res = {}
    for pos in ("P1", "P2"):
        ctrl = np.mean([r[pos]["p_target"] for r in let])
        ctrl_ko = np.mean([r[pos]["p_target_knockout"] for r in let])
        res[pos] = {"ctrl": float(ctrl)}
        line = []
        for m in (str(v), str(u), str(20 + u), "1", str(10 + (u + 3) % 10)):
            rs = [r for r in rows if r["mention"] == m]
            pt = np.mean([r[pos]["p_target"] for r in rs]); pk = np.mean([r[pos]["p_target_knockout"] for r in rs])
            att = {h: np.round(np.mean([r[pos]["heads"][h]["attn_mention_tokens"] for r in rs], 0), 3).tolist() for h in rs[0][pos]["heads"]}
            dla = {h: round(float(np.mean([r[pos]["heads"][h]["dla_target_minus_mean_digit"] for r in rs])), 3) for h in rs[0][pos]["heads"]}
            res[pos][m] = {"p": float(pt), "p_ko": float(pk), "attn": att, "dla": dla}
            line.append(f"'{m}': {pt:.3f} (KO {pk:.3f})")
        print(f"   {pos}: control {ctrl:.3f} (KO {ctrl_ko:.3f}) | " + " | ".join(line))
        for m in (str(u), "1"):
            print(f"      mention '{m}' at {pos}: attention to mention tokens {res[pos][m]['attn']}  DLA onto target {res[pos][m]['dla']}")
    return res


if __name__ == "__main__":
    out = {}
    for p in sys.argv[1:]:
        out[p] = f21(p) if "f21" in p else f22(p)
