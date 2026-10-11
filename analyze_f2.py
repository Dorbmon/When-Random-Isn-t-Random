#!/usr/bin/env python3
"""F2 analysis.

Part 1: paired D_out − D_irr on the main grid, plus a TOST on the irrelevant-mention Δx.
Part 2: minimal-pair contrasts on the F2 grid.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from analyze_repulsion import LETTERS, LOCS, cells_from_dists, load_local, load_logprobs

RNG = np.random.default_rng(20260926)
REPS = 5000
MARGIN = 0.025


def boot(cells, key):
    """Digit-cluster bootstrap: mean, SE, 95% and 90% CIs."""
    cells = [c for c in cells if c.get(key) is not None and np.isfinite(c[key])]
    by = collections.defaultdict(list)
    for c in cells:
        by[c["c"]].append(c[key])
    ds = sorted(by)
    sums = np.array([np.sum(by[d]) for d in ds]); cnts = np.array([len(by[d]) for d in ds])
    idx = RNG.integers(0, len(ds), (REPS, len(ds)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    m = float(np.mean([c[key] for c in cells]))
    return {"_boots": bs, "mean": m, "se": float(bs.std(ddof=1)), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
            "ci90": [float(x) for x in np.percentile(bs, [5, 95])], "cells": len(cells)}


def dersimonian_laird(ests):
    """Random-effects pooling of [(mean, se)]."""
    y = np.array([e[0] for e in ests]); v = np.array([max(e[1], 1e-6) ** 2 for e in ests])
    w = 1 / v
    fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - fixed) ** 2)
    tau2 = max(0.0, (q - (len(y) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if len(y) > 1 else 0.0
    ws = 1 / (v + tau2)
    mu = np.sum(ws * y) / np.sum(ws); se = np.sqrt(1 / np.sum(ws))
    return {"mean": float(mu), "se": float(se), "ci95": [float(mu - 1.96 * se), float(mu + 1.96 * se)],
            "tau2": float(tau2), "k": len(y)}


# ---------------------------------------------------------------- Part 1

def part1(D):
    prim = cells_from_dists(D, "qx")  # letter-controlled, all n
    irr = collections.defaultdict(list); cd = collections.defaultdict(list)
    for r in prim:
        irr[(r["prompt"], r["n"], r["c"])].append(r["delta"])
        if r["loc"] == "comment_down":
            cd[(r["prompt"], r["n"], r["c"])].append(r["delta"])
    cells = []
    for p in ("base", "uniform"):
        for n in (6, 8):
            pc = {w: D.get((p, n, None, "print_const", str(w))) for w in range(10)}
            if any(v is None for v in pc.values()):
                continue
            for c in range(n):
                d_out = pc[c]["qx"][c] - np.mean([pc[w]["qx"][c] for w in range(n, 10)])
                k = (p, n, c)
                if k not in irr:
                    continue
                d_irr = float(np.mean(irr[k])); d_cd = float(np.mean(cd[k])) if cd.get(k) else None
                cells.append({"prompt": p, "n": n, "c": c, "d_out": float(d_out), "d_irr": d_irr,
                              "diff": float(d_out - d_irr),
                              "diff_cd": float(d_out - d_cd) if d_cd is not None else None})
    res = {k: boot(cells, k) for k in ("d_out", "d_irr", "diff", "diff_cd")}
    tost = boot(prim, "delta")
    tost["margin"] = MARGIN
    tost["equivalent"] = bool(tost["ci90"][0] > -MARGIN and tost["ci90"][1] < MARGIN)
    tost["min_margin"] = float(max(abs(tost["ci90"][0]), abs(tost["ci90"][1])))
    res["tost_irrelevant"] = tost
    return res


# ---------------------------------------------------------------- Part 2

def load_f2_local(path):
    D = {}
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        if r["legal_mass_x"] < 0.5:
            continue
        D[(r["prompt"], r["n"], r["cond"], r["mention"])] = np.asarray(r["q_x"])
    return D


def load_f2_lp(path, spec):
    acc = collections.defaultdict(list)
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        if r["spec"] != spec or not r.get("ok"):
            continue
        p = np.asarray(r["x_lp"]["digit_probs"][:r["n"]])
        if p.sum() > 0:
            acc[(r["prompt"], r["n"], r["cond"], r["mention"])].append(p / p.sum())
    return {k: np.mean(v, 0) for k, v in acc.items()}


CONDS = ("var_print", "var_noprint", "if_true", "if_false", "commented_print",
         "print_before_x", "print_after_x")
CONTRASTS = {"C1_output_role": ("var_print", "var_noprint"),
             "C2_execution": ("if_true", "if_false"),
             "C3_commented_vs_dead": ("commented_print", "if_false"),
             "C4_order": ("print_before_x", "print_after_x")}


def part2(D):
    """Δ_OOR per (prompt, n in {6,8}, cond, c), then contrasts matched on (prompt, n, c)."""
    per = {}
    for p in ("base", "uniform"):
        for n in (6, 8):
            for cond in CONDS:
                M = {w: D.get((p, n, cond, str(w))) for w in range(10)}
                if any(v is None for v in M.values()):
                    continue
                for c in range(n):
                    per[(p, n, cond, c)] = float(M[c][c] - np.mean([M[w][c] for w in range(n, 10)]))
    res = {"conditions": {}, "contrasts": {}}
    for cond in CONDS:
        cells = [{"c": c, "v": v} for (p, n, cc, c), v in per.items() if cc == cond]
        if cells:
            res["conditions"][cond] = boot(cells, "v")
    for name, (a, b) in CONTRASTS.items():
        cells = [{"c": c, "v": per[(p, n, a, c)] - per[(p, n, b, c)]}
                 for (p, n, cc, c) in per if cc == a and (p, n, b, c) in per]
        if cells:
            res["contrasts"][name] = boot(cells, "v")
    # letter-controlled Δ where letters exist (secondary)
    for cond in ("var_noprint", "if_false", "commented_print"):
        cells = []
        for p in ("base", "uniform"):
            for n in (6, 8, 10):
                ctrl = [D.get((p, n, cond, L)) for L in LETTERS]
                if any(x is None for x in ctrl):
                    continue
                base = np.mean(ctrl, 0)
                for c in range(n):
                    q = D.get((p, n, cond, str(c)))
                    if q is not None:
                        cells.append({"c": c, "v": float(q[c] - base[c])})
        if cells:
            res["conditions"][cond + "_vs_letters"] = boot(cells, "v")
    return res


def fmt(b, ci="ci95"):
    return f"{b['mean']:+.3f} [{b[ci][0]:+.3f},{b[ci][1]:+.3f}]"


def equal_weight(ests):
    """Equal-weight mean over models; two-stage bootstrap (models, then each model's digit bootstrap)."""
    k = len(ests)
    B = np.stack([e["_boots"] for e in ests])  # k x REPS
    mi = RNG.integers(0, k, (REPS, k)); ri = RNG.integers(0, B.shape[1], (REPS, k))
    bs = B[mi, ri].mean(1)
    mu = float(np.mean([e["mean"] for e in ests]))
    return {"mean": mu, "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "k": k}


def pooled_table(results, key_fn, groups):
    out = {}
    for g, names in groups.items():
        ests = [key_fn(results[n]) for n in names if n in results and key_fn(results[n])]
        if ests:
            out[g] = dersimonian_laird([(e["mean"], e["se"]) for e in ests])
            out[g]["equal_weight"] = equal_weight(ests)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs="*", default=[])
    ap.add_argument("--logprobs", nargs="*", default=[], help="main-grid logprob jsonl files")
    ap.add_argument("--lp-specs", nargs="*", default=[])
    ap.add_argument("--f2-local", nargs="*", default=[])
    ap.add_argument("--f2-lp", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    P1, P2 = {}, {}
    for path in a.local:
        P1[Path(path).stem] = part1(load_local(path)[0])
    for spec in a.lp_specs:
        for path in a.logprobs:
            D, st = load_logprobs(path, spec)
            if D:
                P1["lp:" + spec] = part1(D)
    for path in a.f2_local:
        P2[Path(path).stem.replace("f2_", "")] = part2(load_f2_local(path))
    for spec in a.lp_specs:
        for path in a.f2_lp:
            D = load_f2_lp(path, spec)
            if D:
                P2["lp:" + spec] = part2(D)
    local = [k for k in P1 if not k.startswith("lp:")]
    api = [k for k in P1 if k.startswith("lp:")]
    groups = {"local": local, "api": api, "all": local + api}
    out = {"part1": P1, "part2": P2, "pooled": {}}
    if P1:
        print("PART 1: output constant vs irrelevant mention (n in {6,8}, matched cells)")
        print(f"{'model':26s} {'D_out':>22s} {'D_irr':>22s} {'D_out−D_irr':>22s} {'dist-matched diff':>22s} | TOST(±{MARGIN}) 90%CI  min-margin")
        for k, r in P1.items():
            t = r["tost_irrelevant"]
            print(f"{k:26s} {fmt(r['d_out']):>22s} {fmt(r['d_irr']):>22s} {fmt(r['diff']):>22s} {fmt(r['diff_cd']):>22s} | "
                  f"{'EQUIV' if t['equivalent'] else 'no   '} {fmt(t, 'ci90')} {t['min_margin']:.3f}")
        for key in ("d_out", "d_irr", "diff", "diff_cd"):
            pt = pooled_table(P1, lambda r: r[key], groups)
            out["pooled"]["part1_" + key] = pt
            print(f"  pooled {key:8s}", {g: f"RE {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] | EW {v['equal_weight']['mean']:+.3f} [{v['equal_weight']['ci95'][0]:+.3f},{v['equal_weight']['ci95'][1]:+.3f}] k={v['k']}" for g, v in pt.items()})
    if P2:
        print("\nPART 2: minimal pairs, Δ_OOR (n in {6,8})")
        conds = list(CONDS)
        print(f"{'model':22s} " + " ".join(f"{c[:14]:>21s}" for c in conds))
        for k, r in P2.items():
            print(f"{k:22s} " + " ".join(f"{fmt(r['conditions'][c]) if c in r['conditions'] else '-':>21s}" for c in conds))
        print(f"\n{'model':22s} " + " ".join(f"{c:>24s}" for c in CONTRASTS))
        for k, r in P2.items():
            print(f"{k:22s} " + " ".join(f"{fmt(r['contrasts'][c]) if c in r['contrasts'] else '-':>24s}" for c in CONTRASTS))
        l2 = [k for k in P2 if not k.startswith("lp:")]; a2 = [k for k in P2 if k.startswith("lp:")]
        g2 = {"local": l2, "api": a2, "all": l2 + a2}
        for name in list(CONTRASTS) + [c for c in conds]:
            if name in CONTRASTS:
                pt = pooled_table(P2, lambda r, n=name: r["contrasts"].get(n), g2)
            else:
                pt = pooled_table(P2, lambda r, n=name: r["conditions"].get(n), g2)
            out["pooled"]["part2_" + name] = pt
            print(f"  pooled {name:22s}", {g: f"RE {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] | EW {v['equal_weight']['mean']:+.3f} [{v['equal_weight']['ci95'][0]:+.3f},{v['equal_weight']['ci95'][1]:+.3f}] k={v['k']}" for g, v in pt.items()})
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "_boots"}
        return o
    a.out.write_text(json.dumps(strip(out), indent=1))


if __name__ == "__main__":
    main()
