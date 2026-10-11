#!/usr/bin/env python3
"""Reviewer statistics, part 2: C1 minimal equivalence margin (90% CIs), MAX_RETRIES R with a
cell bootstrap, and scenario invariance on F5 (pairwise TV across constant values vs E)."""
import collections
import glob
import json
from pathlib import Path

import numpy as np

import analyze_f2 as A

rng = np.random.default_rng(7)


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# ---------------- C1 minimal margin ----------------
def c1_margin():
    P2 = {}
    for m in ("Qwen3-4B", "Qwen3-8B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"):
        P2[m] = A.part2(A.load_f2_local(f"data/f2_{m}.jsonl"))
    for spec in ("gpt-5.6-luna", "gpt-6-luna", "gpt-6-sol", "deepseek-v4.1-flash", "deepseek-v4-pro", "kimi-k3", "qwen3.8-flash"):
        for p in ("api_data/logprobs_f2.jsonl", "api_data/logprobs_f2_qwen.jsonl"):
            D = A.load_f2_lp(p, spec)
            if D:
                P2["lp:" + spec] = A.part2(D)
    ests = [P2[k]["contrasts"]["C1_output_role"] for k in P2]
    re = A.dersimonian_laird([(e["mean"], e["se"]) for e in ests])
    re90 = [re["mean"] - 1.645 * re["se"], re["mean"] + 1.645 * re["se"]]
    Bm = np.stack([e["_boots"] for e in ests]); k = len(ests)
    mi = rng.integers(0, k, (A.REPS, k)); ri = rng.integers(0, Bm.shape[1], (A.REPS, k))
    bs = Bm[mi, ri].mean(1)
    ew = float(np.mean([e["mean"] for e in ests]))
    ew90 = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    out = {"k": k, "RE": {"mean": re["mean"], "ci90": re90, "delta_min": max(abs(re90[0]), abs(re90[1]))},
           "EW": {"mean": ew, "ci90": ew90, "delta_min": max(abs(ew90[0]), abs(ew90[1]))}}
    print(f"C1 (k={k}) RE {re['mean']:+.4f} 90% [{re90[0]:+.4f},{re90[1]:+.4f}] δmin {out['RE']['delta_min']:.4f} | "
          f"EW {ew:+.4f} 90% [{ew90[0]:+.4f},{ew90[1]:+.4f}] δmin {out['EW']['delta_min']:.4f}")
    return out


# ---------------- F5: MAX_RETRIES R and scenario invariance ----------------
def load_f5(path):
    D = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        q = np.asarray(r["q_x"]); D[(r["prompt"], r["level"])][r["mention"]] = q / q.sum()
    return D


def f5_stats(path, name, main_E):
    D = load_f5(path)
    n = 8
    res = {}
    # MAX_RETRIES R (controls = out-of-range constants 8, 9)
    cells = []
    per_prompt = {}
    for p in ("base", "uniform"):
        d = D.get((p, "retries"))
        if not d or any(str(w) not in d for w in (8, 9)):
            continue
        ctrl = np.mean([d["8"], d["9"]], 0)
        pc = [(c, d[str(c)][c], ctrl[c]) for c in range(n) if str(c) in d and ctrl[c] >= 0.05]
        cells += pc
        per_prompt[p] = (sum(x[1] for x in pc) / sum(x[2] for x in pc) - 1) if pc else float("nan")
    if cells:
        R = sum(x[1] for x in cells) / sum(x[2] for x in cells) - 1
        arr = np.array([[x[1], x[2]] for x in cells])
        idx = rng.integers(0, len(cells), (10000, len(cells)))
        bs = arr[idx, 0].sum(1) / arr[idx, 1].sum(1) - 1
        res["max_retries"] = {"R": float(R), "ci95_cell": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              "per_prompt": per_prompt, "favored_cells": len(cells)}
    # scenario invariance: pairwise TV across constant values within the same template
    pw, E5 = [], []
    for (p, lvl), d in D.items():
        ms = sorted(d)
        for i in range(len(ms)):
            E5.append(tv(d[ms[i]], np.full(n, 1 / n)))
            for j in range(i + 1, len(ms)):
                pw.append(tv(d[ms[i]], d[ms[j]]))
    res["scenario"] = {"pairwise_tv": float(np.mean(pw)), "E_f5": float(np.mean(E5)), "E_main": main_E,
                       "ratio_to_E_f5": float(np.mean(pw) / np.mean(E5))}
    mr = res.get("max_retries", {})
    print(f"{name:22s} MAX_RETRIES R {mr.get('R', float('nan')):+.3f} cell-CI {[round(x,3) for x in mr.get('ci95_cell',[float('nan')]*2)]} "
          f"per-prompt {{{', '.join(f'{k}: {v:+.2f}' for k, v in mr.get('per_prompt', {}).items())}}} n_fav={mr.get('favored_cells')} | "
          f"scenario pairwise TV {res['scenario']['pairwise_tv']:.3f}  E(F5) {res['scenario']['E_f5']:.3f}  E(main) {main_E:.3f}")
    return res


def main():
    out = {"C1": c1_margin(), "F5": {}}
    rs = json.load(open("results/robust_stats.json"))
    Emain = {**{k: v["invariance"]["E"] for k, v in rs["local"].items()}, **{k: v["invariance"]["E"] for k, v in rs["api"].items()}}
    files = {"Qwen3-8B": "data/f5_Qwen3-8B.jsonl", "Qwen3-32B": "data/f5_Qwen3-32B.jsonl",
             "gemma-4-31B-it": "data/f5_gemma-4-31B-it.jsonl", "Muse-Glimmer-30B": "data/f5_Muse-Glimmer-30B.jsonl"}
    for spec in ("deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-6-luna", "gpt-5.6-luna", "qwen3.8-flash", "kimi-k3", "gpt-6-sol"):
        files[spec] = f"data/f5_lp_{spec}.jsonl"
    for name, path in files.items():
        if Path(path).exists():
            out["F5"][name] = f5_stats(path, name, Emain.get(name, float("nan")))
    Path("results/robust_stats2.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
