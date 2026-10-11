#!/usr/bin/env python3
"""F10 analysis: does a mention of an option word repel or attract its draw?
Local files have row["q"] over the options. API files (--api) have v_top first-token probs."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from f10_categorical import ALL_SETS as SETS

RNG = np.random.default_rng(10)


def load_local(path):
    D = {}
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        D[(r["prompt"], r["set"], r["location"], r["mention"])] = np.asarray(r["q"])
    return {"": D}


def load_api(path):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        if not r.get("ok") or not r.get("v_top"):
            continue
        opts = SETS[r["set"]][0]
        q = np.zeros(len(opts))
        for t, p in r["v_top"].items():
            ts = t.strip().lower()
            if not ts:
                continue
            hits = [i for i, o in enumerate(opts) if o.lower().startswith(ts)]
            if len(hits) == 1:
                q[hits[0]] += p
        if q.sum() > 0.3:
            acc[r["spec"]][(r["prompt"], r["set"], r["location"], r["mention"])].append(q / q.sum())
    return {s: {k: np.mean(v, 0) for k, v in d.items()} for s, d in acc.items()}


def analyze(D):
    cells = []
    for (p, s, loc, m) in D:
        if loc == "base" or m not in SETS[s][0]:
            continue
        opts, ctrls = SETS[s]
        ctrl = [D.get((p, s, loc, c)) for c in ctrls]
        if any(x is None for x in ctrl):
            continue
        o = opts.index(m)
        base = np.mean(ctrl, 0)
        cells.append({"opt": f"{s}:{m}", "loc": loc, "d": D[(p, s, loc, m)][o] - base[o],
                      "ctrl": base[o], "ment": D[(p, s, loc, m)][o]})
    by = collections.defaultdict(list)
    for c in cells:
        by[c["opt"]].append(c["d"])
    ks = sorted(by)
    s_ = np.array([sum(by[k]) for k in ks]); n_ = np.array([len(by[k]) for k in ks])
    idx = RNG.integers(0, len(ks), (5000, len(ks)))
    bs = s_[idx].sum(1) / n_[idx].sum(1)
    fav = [c for c in cells if c["ctrl"] >= .05]
    rel = sum(c["ment"] for c in fav) / sum(c["ctrl"] for c in fav) - 1 if fav else float("nan")
    byloc = {l: float(np.mean([c["d"] for c in cells if c["loc"] == l])) for l in ("assign_down", "comment_down", "prompt_nl")}
    return {"mean": float(np.mean([c["d"] for c in cells])), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
            "frac_negative": float(np.mean([c["d"] < 0 for c in cells])), "fav_rel": rel, "cells": len(cells), "by_loc": byloc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs="*", default=[])
    ap.add_argument("--api", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    runs = [(Path(p).stem, load_local(p)[""]) for p in a.local]
    for p in a.api:
        runs += [("lp:" + s, D) for s, D in load_api(p).items()]
    print(f"{'model':26s} {'Δ(option) [95% CI]':>26s} {'neg':>5s} {'fav rel':>8s}  by location (assign / comment / prompt)")
    for name, D in runs:
        r = analyze(D)
        out[name] = r
        bl = r["by_loc"]
        print(f"{name:26s} {r['mean']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] {r['frac_negative']:5.2f} {r['fav_rel']:+8.2f}  "
              f"{bl['assign_down']:+.3f} / {bl['comment_down']:+.3f} / {bl['prompt_nl']:+.3f}")
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
