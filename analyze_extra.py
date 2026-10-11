#!/usr/bin/env python3
"""Analyses for the exploratory follow-ups F3 (knockout), F4/F5 (distance, dose, realistic)
and F7 (thinking). F6 reuses analyze_repulsion.analyze_local on the f6raw files."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(7)
LETTERS = ("A", "B", "K", "Q")


def cboot(cells, key="v", reps=4000):
    cells = [c for c in cells if np.isfinite(c[key])]
    if not cells:
        return None
    by = collections.defaultdict(list)
    for c in cells:
        by[c["c"]].append(c[key])
    ds = sorted(by)
    s = np.array([np.sum(by[d]) for d in ds]); n = np.array([len(by[d]) for d in ds])
    idx = RNG.integers(0, len(ds), (reps, len(ds)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return {"mean": float(np.mean([c[key] for c in cells])),
            "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "cells": len(cells)}


def fmt(b):
    return "—" if not b else f"{b['mean']:+.3f} [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]"


def fav_rel(cells):
    fav = [c for c in cells if c["ctrl"] >= .05]
    return (sum(c["ment"] for c in fav) / sum(c["ctrl"] for c in fav) - 1) if fav else float("nan")


# ------------------------------------------------------------------ F3

def f3(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    n = rows[0]["n"]
    ks = list(rows[0]["q_x"])
    Q = {(r["prompt"], r["location"], r["mention"]): r["q_x"] for r in rows}
    out = {}
    for k in ks:
        irr, pc = [], []
        for p in ("base", "uniform"):
            for loc in ("assign_down", "comment_down", "prompt_nl", "varname_down"):
                ctrl = np.mean([Q[(p, loc, L)][k] for L in LETTERS], 0)
                for c in range(n):
                    q = Q[(p, loc, str(c))][k]
                    irr.append({"c": c, "v": q[c] - ctrl[c], "ctrl": ctrl[c], "ment": q[c], "loc": loc})
            oor = np.mean([Q[(p, "print_const", str(w))][k] for w in range(n, 10)], 0)
            for c in range(n):
                q = Q[(p, "print_const", str(c))][k]
                pc.append({"c": c, "v": q[c] - oor[c], "ctrl": oor[c], "ment": q[c]})
        out[k] = {"irrelevant": cboot(irr), "print_const": cboot(pc),
                  "irr_fav_rel": fav_rel(irr), "pc_fav_rel": fav_rel(pc),
                  "by_loc": {l: cboot([c for c in irr if c["loc"] == l]) for l in
                             ("assign_down", "comment_down", "prompt_nl", "varname_down")}}
    base_i, base_p = out["none"]["irrelevant"]["mean"], out["none"]["print_const"]["mean"]
    print(f"== F3 {rows[0]['model']}  (recovery = 1 − Δ_k/Δ_none)")
    print(f"{'knockout':22s} {'irrelevant Δx':>24s} {'recov':>6s} {'fav rel':>8s} | {'print_const Δx':>24s} {'recov':>6s} {'fav rel':>8s}")
    for k, v in out.items():
        ri = 1 - v["irrelevant"]["mean"] / base_i if base_i else float("nan")
        rp = 1 - v["print_const"]["mean"] / base_p if base_p else float("nan")
        v["recovery_irrelevant"], v["recovery_print_const"] = ri, rp
        print(f"{k:22s} {fmt(v['irrelevant']):>24s} {ri:6.2f} {v['irr_fav_rel']:+8.2f} | {fmt(v['print_const']):>24s} {rp:6.2f} {v['pc_fav_rel']:+8.2f}")
    return out


# ------------------------------------------------------------------ F4 / F5

def load_grid(path):
    D = {}
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        if r["legal_mass_x"] < .5:
            continue
        D[(r["prompt"], r["family"], r["level"], r["mention"])] = np.asarray(r["q_x"])
    return D


def f4(path):
    D = load_grid(path)
    n = 8
    out = {}
    fams = sorted({(f, lv) for (_, f, lv, _) in D}, key=lambda t: (t[0], str(t[1])))
    print(f"== F4 {Path(path).stem}")
    for fam, lv in fams:
        cells = []
        for p in ("base", "uniform"):
            ctrl = [D.get((p, fam, lv, L)) for L in LETTERS]
            if any(x is None for x in ctrl):
                continue
            ctrl = np.mean(ctrl, 0)
            for c in range(n):
                q = D.get((p, fam, lv, str(c)))
                if q is not None:
                    cells.append({"c": c, "v": q[c] - ctrl[c], "ctrl": ctrl[c], "ment": q[c]})
        out[f"{fam}|{lv}"] = {"delta": cboot(cells), "fav_rel": fav_rel(cells)}
        print(f"  {fam:9s} level={str(lv):3s} Δx={fmt(out[f'{fam}|{lv}']['delta']):>24s} fav rel={out[f'{fam}|{lv}']['fav_rel']:+.2f}")
    return out


def f5(path):
    D = load_grid(path)
    n = 8
    out = {}
    print(f"== F5 {Path(path).stem}")
    for lv in sorted({lv for (_, f, lv, _) in D}):
        cells = []
        for p in ("base", "uniform"):
            M = {w: D.get((p, "realistic", lv, str(w))) for w in range(10)}
            if any(v is None for v in M.values()):
                continue
            oor = np.mean([M[w] for w in (8, 9)], 0)
            for c in range(n):
                cells.append({"c": c, "v": M[c][c] - oor[c], "ctrl": oor[c], "ment": M[c][c]})
        out[lv] = {"delta_oor": cboot(cells), "fav_rel": fav_rel(cells)}
        print(f"  {lv:12s} Δ_OOR={fmt(out[lv]['delta_oor']):>24s} fav rel={out[lv]['fav_rel']:+.2f}")
    allc = []
    return out


# ------------------------------------------------------------------ F8

def f8(path, top=15):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    n = 8
    R = {(r["location"], r["mention"]): r for r in rows}
    layers = sorted(int(L) for L in rows[0]["q_head_knock"])
    H = len(rows[0]["q_head_knock"][str(layers[0])])
    irr_locs = ("assign_down", "comment_down", "prompt_nl", "varname_down")
    score_irr = np.zeros((len(layers), H)); score_pc = np.zeros((len(layers), H))
    base_irr = []; base_pc = []
    for li, L in enumerate(layers):
        for h in range(H):
            si, sp = [], []
            for loc in irr_locs:
                for c in range(n):
                    r = R[(loc, str(c))]
                    rec = r["q_head_knock"][str(L)][h][c] - r["q_none"][c]
                    ctl = np.mean([R[(loc, Lt)]["q_head_knock"][str(L)][h][c] - R[(loc, Lt)]["q_none"][c] for Lt in LETTERS])
                    si.append(rec - ctl)
            for c in range(n):
                r = R[("print_const", str(c))]
                rec = r["q_head_knock"][str(L)][h][c] - r["q_none"][c]
                ctl = np.mean([R[("print_const", str(w))]["q_head_knock"][str(L)][h][c] - R[("print_const", str(w))]["q_none"][c] for w in (8, 9)])
                sp.append(rec - ctl)
            score_irr[li, h] = np.mean(si); score_pc[li, h] = np.mean(sp)
    # total repulsion to recover (Δ_none, same definition as F3 on the base prompt)
    for loc in irr_locs:
        for c in range(n):
            base_irr.append(R[(loc, str(c))]["q_none"][c] - np.mean([R[(loc, Lt)]["q_none"][c] for Lt in LETTERS]))
    for c in range(n):
        base_pc.append(R[("print_const", str(c))]["q_none"][c] - np.mean([R[("print_const", str(w))]["q_none"][c] for w in (8, 9)]))
    tot_i, tot_p = -np.mean(base_irr), -np.mean(base_pc)
    print(f"== F8 {rows[0]['model']}: repulsion to recover irr={tot_i:.3f} print={tot_p:.3f}")
    order = np.dstack(np.unravel_index(np.argsort(-score_irr, axis=None), score_irr.shape))[0][:top]
    print("  top heads (irrelevant):", [(f"L{layers[a]}H{b}", round(score_irr[a, b] / tot_i, 3), round(score_pc[a, b] / tot_p, 3)) for a, b in order])
    order_p = np.dstack(np.unravel_index(np.argsort(-score_pc, axis=None), score_pc.shape))[0][:top]
    print("  top heads (print_const):", [(f"L{layers[a]}H{b}", round(score_pc[a, b] / tot_p, 3), round(score_irr[a, b] / tot_i, 3)) for a, b in order_p])
    print("  per-layer summed share irr:", {layers[i]: round(score_irr[i].sum() / tot_i, 2) for i in range(len(layers))})
    print("  corr(head score irr, print):", round(float(np.corrcoef(score_irr.ravel(), score_pc.ravel())[0, 1]), 3))
    return {"layers": layers, "score_irr": score_irr.tolist(), "score_pc": score_pc.tolist(),
            "total_irr": tot_i, "total_pc": tot_p,
            "top_irr": [[layers[a], int(b)] for a, b in order], "top_pc": [[layers[a], int(b)] for a, b in order_p]}


# ------------------------------------------------------------------ F7

def f7(path, exact_path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    n = 8
    X = collections.defaultdict(list)
    closed = np.mean([r["thinking_closed"] for r in rows])
    for r in rows:
        if r["x"] is not None and 0 <= r["x"] < n:
            X[(r["location"], r["mention"])].append(r["x"])
    valid = sum(len(v) for v in X.values()) / len(rows)
    E = {}
    for l in Path(exact_path).read_text().splitlines():
        r = json.loads(l)
        if r["prompt"] == "base" and r["n"] == n and r["expr"] in ("e1", None):
            E[(r["location"], r["mention"])] = np.asarray(r["q_x"])

    def dist(xs):
        return np.bincount(xs, minlength=n)[:n] / max(len(xs), 1)

    def cells(get):
        irr, pc = [], []
        for loc in ("assign_down", "comment_down"):
            ctrl = np.mean([get((loc, L)) for L in LETTERS], 0)
            for c in range(n):
                q = get((loc, str(c)))
                irr.append({"c": c, "v": q[c] - ctrl[c], "ctrl": ctrl[c], "ment": q[c]})
        oor = np.mean([get(("print_const", str(w))) for w in (8, 9)], 0)
        for c in range(n):
            q = get(("print_const", str(c)))
            pc.append({"c": c, "v": q[c] - oor[c], "ctrl": oor[c], "ment": q[c]})
        return irr, pc

    ti, tp = cells(lambda k: dist(X[k]))
    ei, ep = cells(lambda k: E[k])
    # two-stage bootstrap for the thinking arm: resample responses, then digits
    bi, bp = [], []
    for _ in range(1000):
        Xb = {k: list(RNG.choice(v, len(v))) for k, v in X.items()}
        i2, p2 = cells(lambda k: dist(Xb[k]))
        digs = RNG.integers(0, n, n)
        bi.append(np.mean([c["v"] for d in digs for c in i2 if c["c"] == d]))
        bp.append(np.mean([c["v"] for d in digs for c in p2 if c["c"] == d]))
    res = {"thinking_closed": float(closed), "valid_x_rate": float(valid),
           "samples_per_context": float(np.median([len(v) for v in X.values()])),
           "thinking_irrelevant": {"mean": float(np.mean([c["v"] for c in ti])), "ci95": list(np.percentile(bi, [2.5, 97.5]))},
           "thinking_print_const": {"mean": float(np.mean([c["v"] for c in tp])), "ci95": list(np.percentile(bp, [2.5, 97.5]))},
           "exact_nothink_irrelevant": cboot(ei), "exact_nothink_print_const": cboot(ep),
           "thinking_fav_rel_irr": fav_rel(ti), "thinking_fav_rel_pc": fav_rel(tp),
           "nothink_fav_rel_irr": fav_rel(ei), "nothink_fav_rel_pc": fav_rel(ep),
           "thinking_base_qx": dist(X[("base", None)]).tolist(), "nothink_base_qx": E[("base", None)].tolist()}
    print(f"== F7 {rows[0]['model']}: closed={closed:.3f} valid={valid:.3f} n/ctx={res['samples_per_context']}")
    for k in ("irrelevant", "print_const"):
        t = res[f"thinking_{k}"]; e = res[f"exact_nothink_{k}"]
        print(f"  {k:12s} thinking Δx={t['mean']:+.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}]   no-think exact Δx={fmt(e)}")
    print("  base q(x) thinking:", np.round(res["thinking_base_qx"], 3), " no-think:", np.round(res["nothink_base_qx"], 3))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f3", nargs="*", default=[])
    ap.add_argument("--f4", nargs="*", default=[])
    ap.add_argument("--f5", nargs="*", default=[])
    ap.add_argument("--f8", nargs="*", default=[])
    ap.add_argument("--f7", nargs="*", default=[], help="pairs thinking.jsonl:exact_main.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {"f3": {}, "f4": {}, "f5": {}, "f7": {}, "f8": {}}
    for p in a.f8:
        out["f8"][Path(p).stem] = f8(p)
    for p in a.f3:
        out["f3"][Path(p).stem] = f3(p)
    for p in a.f4:
        out["f4"][Path(p).stem] = f4(p)
    for p in a.f5:
        out["f5"][Path(p).stem] = f5(p)
    for pair in a.f7:
        t, e = pair.split(":")
        out["f7"][Path(t).stem] = f7(t, e)
    a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
