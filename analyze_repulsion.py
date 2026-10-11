#!/usr/bin/env python3
"""Pre-registered analysis for the context-repulsion experiment."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

LETTERS = ("A", "B", "K", "Q")
LOCS = ("comment_down", "assign_down", "varname_down", "comment_up", "prompt_nl")
FLOOR = 1e-9
RNG = np.random.default_rng(20260925)


# ---------------------------------------------------------------- data loading

def load_local(path):
    """Return {(prompt, n, expr, loc, mention): dict(qx, qy, qxy)} plus exclusion count."""
    D, excluded = {}, 0
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["legal_mass_x"] < 0.5:
            excluded += 1
            continue
        q = np.asarray(r["q_xy"])
        D[(r["prompt"], r["n"], r["expr"], r["location"], r["mention"])] = {
            "qx": q.sum(1), "qy": q.sum(0), "qxy": q, "legal_mass_x": r["legal_mass_x"]}
    return D, excluded


def load_api(path, spec):
    """Return {key: np.array of x values} and {key: list of (x, y)} for one spec."""
    X = collections.defaultdict(list)
    XY = collections.defaultdict(list)
    seen = set()
    stats = collections.Counter()
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["spec"] != spec or r["http_status"] != 200 or not r["raw"]:
            continue
        k = (r["prompt"], r["cid"], r["sample_index"])
        if k in seen:
            continue
        seen.add(k)
        stats["responses"] += 1
        stats["strict"] += bool(r["strict"])
        key = (r["prompt"], r["n"], r["expr"], r["location"], r["mention"])
        if r["x"] is not None and 0 <= r["x"] < r["n"]:
            X[key].append(r["x"])
            stats["valid_x"] += 1
            if r["y"] is not None and 0 <= r["y"] < r["n"]:
                XY[key].append((r["x"], r["y"]))
    return {k: np.asarray(v) for k, v in X.items()}, XY, dict(stats)


def load_logprobs(path, spec):
    """API logprob arm: q(x) from top_logprobs, renormalized over 0..n-1, averaged over repeats."""
    acc = collections.defaultdict(list)
    stats = collections.Counter()
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["spec"] != spec:
            continue
        stats["rows"] += 1
        if not r.get("ok"):
            stats["excluded"] += 1
            continue
        p = np.asarray(r["x_lp"]["digit_probs"][:r["n"]])
        if p.sum() <= 0:
            stats["excluded"] += 1
            continue
        stats["legal_mass_sum"] += r["x_lp"]["digit_mass"]
        acc[(r["prompt"], r["n"], r["expr"], r["location"], r["mention"])].append(p / p.sum())
    D = {k: {"qx": np.mean(v, axis=0), "reps": v} for k, v in acc.items()}
    return D, dict(stats)


def analyze_logprobs(path, spec):
    D, stats = load_logprobs(path, spec)
    prim = cells_from_dists(D, "qx")
    noise = letter_noise(D)
    # Repeat reliability: TV between the two repeats of the same context.
    rep_tv = [0.5 * np.abs(v["reps"][0] - v["reps"][1]).sum() for v in D.values() if len(v["reps"]) >= 2]
    res = {"spec": spec, "stats": stats, "contexts": len(D),
           "primary_delta_x": cluster_boot(prim, "delta"),
           "secondary_logratio_x": cluster_boot(prim, "logratio"),
           "diagonal_contrast_x": cluster_boot(diagonal_contrast(D), "delta"),
           "letter_noise": {"mean_abs": float(np.mean(np.abs(noise))) if len(noise) else None},
           "repeat_tv": {"mean": float(np.mean(rep_tv)) if rep_tv else None, "n": len(rep_tv)},
           "by_location": breakdown(prim, "loc"), "by_prompt": breakdown(prim, "prompt"),
           "by_n": breakdown(prim, "n")}
    if res["letter_noise"]["mean_abs"]:
        res["abs_delta_over_letter_noise"] = float(np.mean([abs(r["delta"]) for r in prim]) / res["letter_noise"]["mean_abs"])
    pc = print_const_cells(D)
    res["print_const_oor"] = cluster_boot([r for r in pc if "delta_oor" in r], "delta_oor")
    res["print_const_vs_base"] = cluster_boot([r for r in pc if "delta_base" in r], "delta_base")
    bases = [v["qx"] for (p, n, e, l, m), v in D.items() if l == "base"]
    res["collapse_norm_entropy_x"] = float(np.mean([-(q[q > 0] * np.log(q[q > 0])).sum() / np.log(len(q)) for q in bases])) if bases else None
    res["confirmed_H1"] = bool(res["primary_delta_x"] and res["primary_delta_x"]["ci95"][1] < 0)
    return res, prim


# ---------------------------------------------------------------- cell construction

def cells_from_dists(D, var="qx"):
    """Primary cells: one per (prompt, n, expr, loc, in-range digit c)."""
    out = []
    groups = {(p, n, e, l) for (p, n, e, l, m) in D if l in LOCS}
    for (p, n, e, l) in sorted(groups):
        ctrl = [D.get((p, n, e, l, ell)) for ell in LETTERS]
        if any(c is None for c in ctrl):
            continue
        for c in range(n):
            d = D.get((p, n, e, l, str(c)))
            if d is None:
                continue
            base = np.mean([cc[var][c] for cc in ctrl])
            val = d[var][c]
            out.append({"prompt": p, "n": n, "expr": e, "loc": l, "c": c,
                        "q_mention": float(val), "q_ctrl": float(base),
                        "delta": float(val - base),
                        "logratio": float(np.log(max(val, FLOOR)) - np.log(max(base, FLOOR)))})
    return out


def either_dist(d, c):
    return d["qx"][c] + d["qy"][c] - d["qxy"][c, c]


def cells_either(D):
    out = []
    groups = {(p, n, e, l) for (p, n, e, l, m) in D if l in LOCS}
    for (p, n, e, l) in sorted(groups):
        ctrl = [D.get((p, n, e, l, ell)) for ell in LETTERS]
        if any(c is None for c in ctrl):
            continue
        for c in range(n):
            d = D.get((p, n, e, l, str(c)))
            if d is None:
                continue
            base = np.mean([either_dist(cc, c) for cc in ctrl])
            out.append({"prompt": p, "n": n, "expr": e, "loc": l, "c": c,
                        "delta": float(either_dist(d, c) - base)})
    return out


def letter_noise(D):
    """Δ^ℓ(v) = q_ℓ(x=v) − mean_{ℓ'≠ℓ} q_ℓ'(x=v) for every letter, value."""
    vals = []
    groups = {(p, n, e, l) for (p, n, e, l, m) in D if l in LOCS}
    for (p, n, e, l) in sorted(groups):
        ctrl = {ell: D.get((p, n, e, l, ell)) for ell in LETTERS}
        if any(v is None for v in ctrl.values()):
            continue
        for ell in LETTERS:
            others = np.mean([ctrl[o]["qx"] for o in LETTERS if o != ell], axis=0)
            vals.extend((ctrl[ell]["qx"] - others).tolist())
    return np.asarray(vals)


def diagonal_contrast(D):
    """D = mean_v (M[v,v] − mean_{c≠v} M[c,v]) with M over digit mentions 0..9."""
    out = []
    groups = {(p, n, e, l) for (p, n, e, l, m) in D if l in LOCS}
    for (p, n, e, l) in sorted(groups):
        M = [D.get((p, n, e, l, str(c))) for c in range(10)]
        if any(m is None for m in M):
            continue
        M = np.asarray([m["qx"] for m in M])  # 10 x n
        for v in range(n):
            off = np.mean([M[c, v] for c in range(10) if c != v])
            out.append({"prompt": p, "n": n, "expr": e, "loc": l, "c": v,
                        "delta": float(M[v, v] - off)})
    return out


def print_const_cells(D):
    out = []
    for p in ("base", "uniform"):
        for n in (6, 8, 10):
            pc = {c: D.get((p, n, None, "print_const", str(c))) for c in range(10)}
            if any(v is None for v in pc.values()):
                continue
            bases = [D.get((p, n, e, "base", None)) for e in ("e1", "e2")]
            bases = [b for b in bases if b is not None]
            for c in range(n):
                row = {"prompt": p, "n": n, "c": c, "q_mention": float(pc[c]["qx"][c])}
                if n < 10:
                    ctrl = np.mean([pc[w]["qx"][c] for w in range(n, 10)])
                    row["delta_oor"] = float(pc[c]["qx"][c] - ctrl)
                    row["logratio_oor"] = float(np.log(max(pc[c]["qx"][c], FLOOR))
                                                - np.log(max(ctrl, FLOOR)))
                if bases:
                    cb = np.mean([b["qx"][c] for b in bases])
                    row["delta_base"] = float(pc[c]["qx"][c] - cb)
                out.append(row)
    return out


# ---------------------------------------------------------------- statistics

def cluster_boot(cells, key="delta", reps=10000):
    """Mean with a 95% CI from resampling digit identities c."""
    if not cells:
        return None
    by = collections.defaultdict(list)
    for r in cells:
        by[r["c"]].append(r[key])
    digits = sorted(by)
    sums = np.array([np.sum(by[d]) for d in digits])
    cnts = np.array([len(by[d]) for d in digits])
    idx = RNG.integers(0, len(digits), size=(reps, len(digits)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    vals = np.array([r[key] for r in cells])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)],
            "frac_negative": float(np.mean(vals < 0)), "cells": len(vals),
            "median": float(np.median(vals))}


def breakdown(cells, field, key="delta"):
    out = {}
    for v in sorted({r[field] for r in cells}, key=str):
        out[str(v)] = cluster_boot([r for r in cells if r[field] == v], key, reps=4000)
    return out


def collapse(D):
    rows = []
    for (p, n, e, l, m), d in D.items():
        if l != "base":
            continue
        qx = d["qx"]
        h = -np.sum(qx[qx > 0] * np.log(qx[qx > 0])) / np.log(n)
        rows.append({"prompt": p, "n": n, "expr": e, "norm_entropy_x": float(h),
                     "max_mass_x": float(qx.max()), "argmax_x": int(qx.argmax()),
                     "max_pair_mass": float(d["qxy"].max()),
                     "p_x_eq_y": float(np.trace(d["qxy"]))})
    return rows


def analyze_local(path):
    D, excl = load_local(path)
    prim = cells_from_dists(D, "qx")
    ycells = cells_from_dists(D, "qy")
    noise = letter_noise(D)
    res = {
        "n_rows": len(D), "excluded_low_legal_mass": excl,
        "primary_delta_x": cluster_boot(prim, "delta"),
        "secondary_logratio_x": cluster_boot(prim, "logratio"),
        "delta_y": cluster_boot(ycells, "delta"),
        "delta_either": cluster_boot(cells_either(D), "delta"),
        "diagonal_contrast_x": cluster_boot(diagonal_contrast(D), "delta"),
        "letter_noise": {"mean_abs": float(np.mean(np.abs(noise))) if len(noise) else None,
                         "sd": float(np.std(noise)) if len(noise) else None},
        "by_location": breakdown(prim, "loc"), "by_n": breakdown(prim, "n"),
        "by_prompt": breakdown(prim, "prompt"), "by_expr": breakdown(prim, "expr"),
        "by_location_logratio": breakdown(prim, "loc", "logratio"),
        "collapse_base": collapse(D),
    }
    if res["letter_noise"]["mean_abs"]:
        res["abs_delta_over_letter_noise"] = float(
            np.mean([abs(r["delta"]) for r in prim]) / res["letter_noise"]["mean_abs"])
    pc = print_const_cells(D)
    res["print_const_oor"] = cluster_boot([r for r in pc if "delta_oor" in r], "delta_oor")
    res["print_const_oor_logratio"] = cluster_boot([r for r in pc if "delta_oor" in r],
                                                   "logratio_oor")
    res["print_const_vs_base"] = cluster_boot([r for r in pc if "delta_base" in r],
                                              "delta_base")
    res["confirmed_H1"] = bool(res["primary_delta_x"]["ci95"][1] < 0)
    return res, prim


# ---------------------------------------------------------------- API arm

def api_dists(X, n=8):
    return {k: {"qx": np.bincount(v, minlength=n)[:n] / len(v)} for k, v in X.items() if len(v)}


def analyze_api(path, spec, reps=2000):
    X, XY, stats = load_api(path, spec)
    point = cells_from_dists(api_dists(X), "qx")
    pc_point = print_const_cells_api(api_dists(X))
    # Two-stage bootstrap: responses are resampled within each context, then digits are
    # resampled as clusters.
    boots, pc_boots = [], []
    keys = list(X)
    for _ in range(reps):
        Xb = {k: X[k][RNG.integers(0, len(X[k]), len(X[k]))] for k in keys}
        cells = cells_from_dists(api_dists(Xb), "qx")
        digs = RNG.integers(0, 8, 8)
        byd = collections.defaultdict(list)
        for r in cells:
            byd[r["c"]].append(r["delta"])
        boots.append(np.mean(np.concatenate([byd[d] for d in digs if byd[d]])))
        pcc = print_const_cells_api(api_dists(Xb))
        byd = collections.defaultdict(list)
        for r in pcc:
            byd[r["c"]].append(r["delta_oor"])
        pc_boots.append(np.mean(np.concatenate([byd[d] for d in digs if byd[d]])))
    def summ(cells, key, bs):
        vals = np.array([r[key] for r in cells])
        return {"mean": float(vals.mean()), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                "frac_negative": float(np.mean(vals < 0)), "cells": len(vals)}
    res = {"spec": spec, "stats": stats,
           "samples_per_context_median": float(np.median([len(v) for v in X.values()])),
           "primary_delta_x": summ(point, "delta", boots),
           "print_const_oor": summ(pc_point, "delta_oor", pc_boots),
           "by_location": {l: {"mean": float(np.mean([r["delta"] for r in point if r["loc"] == l])),
                               "frac_negative": float(np.mean([r["delta"] < 0 for r in point if r["loc"] == l]))}
                           for l in sorted({r["loc"] for r in point})},
           "by_prompt": {p: float(np.mean([r["delta"] for r in point if r["prompt"] == p]))
                         for p in ("base", "uniform")}}
    # P(x=y) from the empirical pairs, over base contexts
    for p in ("base", "uniform"):
        pairs = [xy for k, v in XY.items() if k[0] == p for xy in v]
        res[f"p_x_eq_y_{p}"] = float(np.mean([a == b for a, b in pairs])) if pairs else None
    res["confirmed_H1"] = bool(res["primary_delta_x"]["ci95"][1] < 0)
    return res, point


def print_const_cells_api(D, n=8):
    out = []
    for p in ("base", "uniform"):
        pc = {c: D.get((p, n, None, "print_const", str(c))) for c in range(10)}
        if any(v is None for v in pc.values()):
            continue
        for c in range(n):
            ctrl = np.mean([pc[w]["qx"][c] for w in range(n, 10)])
            out.append({"prompt": p, "n": n, "c": c, "delta_oor": float(pc[c]["qx"][c] - ctrl)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs="*", default=[])
    ap.add_argument("--api", type=Path)
    ap.add_argument("--api-specs", nargs="*", default=[])
    ap.add_argument("--logprobs", type=Path)
    ap.add_argument("--lp-specs", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {"local": {}, "api": {}}
    cells_dump = []
    for path in a.local:
        name = Path(path).stem
        res, cells = analyze_local(path)
        out["local"][name] = res
        cells_dump += [{"model": name, **c} for c in cells]
        p = res["primary_delta_x"]
        print(f"{name:20s} Δx={p['mean']:+.4f} CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] "
              f"neg={p['frac_negative']:.2f} logr={res['secondary_logratio_x']['mean']:+.2f} "
              f"H1={res['confirmed_H1']} print_oor={res['print_const_oor']['mean']:+.4f}", flush=True)
    for spec in a.api_specs:
        res, cells = analyze_api(a.api, spec)
        out["api"][spec] = res
        cells_dump += [{"model": "api:" + spec, **c} for c in cells]
        p = res["primary_delta_x"]
        print(f"{spec:26s} Δx={p['mean']:+.4f} CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] "
              f"neg={p['frac_negative']:.2f} H1={res['confirmed_H1']} "
              f"print_oor={res['print_const_oor']['mean']:+.4f} n/ctx={res['samples_per_context_median']}",
              flush=True)
    out["logprobs"] = {}
    for spec in a.lp_specs:
        res, cells = analyze_logprobs(a.logprobs, spec)
        out["logprobs"][spec] = res
        cells_dump += [{"model": "lp:" + spec, **c} for c in cells]
        p = res["primary_delta_x"]
        if p:
            print(f"lp:{spec:24s} Δx={p['mean']:+.4f} CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] "
                  f"neg={p['frac_negative']:.2f} H1={res['confirmed_H1']} ctx={res['contexts']} "
                  f"print_oor={res['print_const_oor']['mean'] if res['print_const_oor'] else float('nan'):+.4f} "
                  f"repTV={res['repeat_tv']['mean']}", flush=True)
    local_conf = [v["confirmed_H1"] for v in out["local"].values()]
    k = sum(local_conf)
    out["overall_local_decision"] = ("general" if k >= 4 else "partial" if k >= 2 else
                                     "not confirmed") + f" ({k}/{len(local_conf)})"
    a.out.write_text(json.dumps(out, indent=1))
    (a.out.parent / (a.out.stem + "_cells.jsonl")).write_text(
        "\n".join(json.dumps(c) for c in cells_dump) + "\n")
    print("DECISION", out["overall_local_decision"])


if __name__ == "__main__":
    main()
