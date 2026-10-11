#!/usr/bin/env python3
"""F25/F26 analysis. Writes results/f25.json and results/f26.json.

F25 primary: per model, Delta_x of the classic anchor about x (cmp_x) against the four letter
controls at the same slot, averaged over in-range digits, with a 95% bootstrap over its 12
templates (prompt x n x expr; B = 10,000; digits fixed) -- the same estimator as the
"Same phrase" column (robust_stats.cells / template_boot).
Secondary: cmp_answer; the paired difference cmp_x - prompt_nl (same template and digit);
favored-value change (descriptive, not cross-fitted); pipeline check of prompt_nl against the
main grid (data/<model>.jsonl).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robust_stats as rs  # noqa: E402

MODELS = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"]
LOCS = ("cmp_x", "cmp_answer", "prompt_nl")
B = 10000


def load(path, locs):
    T = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if r["location"] in locs:
            q = np.asarray(r["q_x"]); T[(r["prompt"], r["n"], r["expr"], r["location"])][r["mention"]] = q / q.sum()
    return T


def paired_boot(ca, cb, rng):
    """Template bootstrap of mean(Delta_a - Delta_b) over matched (template, digit) cells."""
    key = lambda x: (x[0][0], x[0][1], x[0][2], x[1])  # noqa: E731
    A = {key(x): x[2] - x[3] for x in ca}; Bm = {key(x): x[2] - x[3] for x in cb}
    ks = sorted(set(A) & set(Bm))
    temps = sorted({k[:3] for k in ks}); ti = {t: i for i, t in enumerate(temps)}
    s = np.zeros(len(temps)); n = np.zeros(len(temps))
    for k in ks:
        s[ti[k[:3]]] += A[k] - Bm[k]; n[ti[k[:3]]] += 1
    idx = rng.integers(0, len(temps), (B, len(temps)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(ks)


def f25():
    rng = np.random.default_rng(20260927)
    rs.rng = rng
    out = {}
    for m in MODELS:
        p = HERE / f"data/f25_{m}.jsonl"
        if not p.exists():
            continue
        T = load(p, LOCS + ("base",))
        res = {"rows": sum(1 for _ in open(p))}
        cells = {}
        for loc in LOCS:
            Tl = {t: d for t, d in T.items() if t[3] == loc}
            cs = rs.cells(Tl); cells[loc] = cs
            ci, k = rs.template_boot(cs, rs.dx)
            base = {t[:3]: d[None] for t, d in T.items() if t[3] == "base"}
            dbase = [x[2] - base[x[0][:3]][x[1]] for x in cs]
            res[loc] = {"dx": rs.dx(cs), "ci95_template": ci, "n_templates": k, "cells": len(cs),
                        "fav_rel_change": rs.R_of(cs), "d_vs_no_mention": float(np.mean(dbase))}
        for a, b in (("cmp_x", "prompt_nl"), ("cmp_answer", "prompt_nl"), ("cmp_x", "cmp_answer")):
            d, ci, nc = paired_boot(cells[a], cells[b], rng)
            res[f"{a}_minus_{b}"] = {"diff": d, "ci95_template": ci, "cells": nc}
        # pipeline check: re-scored prompt_nl vs the main grid
        main = load(HERE / f"data/{m}.jsonl", ("prompt_nl",))
        diffs = [float(np.abs(T[t][mm] - main[t][mm]).max()) for t in main for mm in main[t] if t in T and mm in T[t]]
        res["pipeline_check"] = {"contexts": len(diffs), "max_abs_qx_diff": max(diffs) if diffs else None,
                                 "main_grid_dx_prompt_nl": rs.dx(rs.cells(main))}
        out[m] = res
        print(f"{m:18s} cmp_x {res['cmp_x']['dx']:+.4f} [{res['cmp_x']['ci95_template'][0]:+.4f},{res['cmp_x']['ci95_template'][1]:+.4f}]"
              f"  cmp_answer {res['cmp_answer']['dx']:+.4f} [{res['cmp_answer']['ci95_template'][0]:+.4f},{res['cmp_answer']['ci95_template'][1]:+.4f}]"
              f"  incidental {res['prompt_nl']['dx']:+.4f}  cmp_x-inc {res['cmp_x_minus_prompt_nl']['diff']:+.4f}"
              f" [{res['cmp_x_minus_prompt_nl']['ci95_template'][0]:+.4f},{res['cmp_x_minus_prompt_nl']['ci95_template'][1]:+.4f}]"
              f"  | check max|dq| {res['pipeline_check']['max_abs_qx_diff']:.2e}")
    if out:
        per_model = {k: r for k, r in out.items() if not k.startswith("_")}
        for loc in ("cmp_x", "cmp_answer"):
            v = [r[loc] for r in per_model.values()]
            out.setdefault("_summary", {})[loc] = {
                "models": len(v), "positive": sum(x["dx"] > 0 for x in v),
                "ci_above_0": sum(x["ci95_template"][0] > 0 for x in v),
                "ci_below_0": sum(x["ci95_template"][1] < 0 for x in v)}
        d = [r["cmp_x_minus_prompt_nl"] for k, r in out.items() if not k.startswith("_")]
        out["_summary"]["cmp_x_minus_prompt_nl"] = {"positive": sum(x["diff"] > 0 for x in d),
                                                    "ci_above_0": sum(x["ci95_template"][0] > 0 for x in d)}
        print(json.dumps(out["_summary"]))
    (HERE / "results/f25.json").write_text(json.dumps(out, indent=1))


def f26():
    out = {}
    for m in ["Qwen3-8B", "Qwen3-8B-Base", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it"]:
        p = HERE / f"data/f26_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        f20 = json.loads((HERE / f"data/f20_{m}.json").read_text())["candidates"]
        # consistency with F20 (same code, same weights)
        chk = max(abs(d["candidates"][h][e]["z"] - f20[h][e]["diag_z_mean"]) for h in d["candidates"] for e in ("raw", "effective"))
        out[m] = {"H": d["H"], "candidates": d["candidates"], "max_abs_z_diff_vs_f20": chk}
        for h, r in d["candidates"].items():
            print(f"{m:15s} {h:8s} raw z {r['raw']['z']:+.2f} rank {r['raw']['z_rank']}/{r['raw']['n_heads']}"
                  f"  eff z {r['effective']['z']:+.2f} rank {r['effective']['z_rank']}/{r['effective']['n_heads']}  (F20 diff {chk:.1e})")
    (HERE / "results/f26.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    f25()
    f26()
