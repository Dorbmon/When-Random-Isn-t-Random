#!/usr/bin/env python3
"""Analyses for the reviewer requests F16 (free vs exact), F17 (Gemma DLA), F18 (anchoring), F19 (flip)."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(19)
LET = ("A", "B", "K", "Q")


def cboot(vals_by_key, reps=4000):
    ks = sorted(vals_by_key)
    s = np.array([np.sum(vals_by_key[k]) for k in ks]); n = np.array([len(vals_by_key[k]) for k in ks])
    idx = RNG.integers(0, len(ks), (reps, len(ks)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    allv = [v for k in ks for v in vals_by_key[k]]
    return float(np.mean(allv)), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


# ---------------- F16 ----------------
def f16(free_path, exact_path, n=8):
    ex = {}
    for l in open(exact_path):
        r = json.loads(l)
        ex[(r["prompt"], r["cid"])] = np.array(r["q_x"])
    fg = collections.defaultdict(list)
    starts = []
    for l in open(free_path):
        r = json.loads(l)
        fg[(r["prompt"], r["cid"])].append(r["x"])
        starts.append(bool(r.get("starts_with_x")))
    tvs, noise, valid = [], [], []
    E, X = {}, {}
    for k, xs in fg.items():
        v = [x for x in xs if x is not None and 0 <= x < n]
        valid.append(len(v) / len(xs))
        if not v or k not in ex:
            continue
        emp = np.bincount(v, minlength=n)[:n] / len(v)
        q = ex[k] / ex[k].sum()
        tvs.append(0.5 * np.abs(emp - q).sum())
        noise.append(0.5 * np.abs(RNG.multinomial(len(v), q) / len(v) - q).sum())
        E[k], X[k] = emp, q

    def delta(get):
        out = []
        for p in ("base", "uniform"):
            for loc in ("comment_down", "prompt_nl"):
                ctrl = np.mean([get((p, f"n8|e1|{loc}|{L}")) for L in LET], 0)
                for c in range(n):
                    out.append(get((p, f"n8|e1|{loc}|{c}"))[c] - ctrl[c])
        return float(np.mean(out))
    return {"contexts": len(tvs), "samples_per_ctx": int(np.median([len(v) for v in fg.values()])),
            "valid_x_rate": float(np.mean(valid)), "starts_with_x_rate": float(np.mean(starts)),
            "tv_mean": float(np.mean(tvs)), "tv_max": float(np.max(tvs)), "noise_tv_mean": float(np.mean(noise)),
            "delta_exact": delta(X.get), "delta_free": delta(E.get)}


# ---------------- F17 ----------------
def f17(path, n=8):
    rows = [json.loads(l) for l in open(path)]
    var = {int(d): [str(t) for t in ts] for d, ts in rows[0]["digit_variants"].items()}
    # dominant variant per digit: highest mean probability at x= over contexts
    dom = {}
    for d in range(n):
        means = {t: np.mean([r["variant_prob"][t] for r in rows]) for t in var[d]}
        dom[d] = max(means, key=means.get)
    share = {d: float(np.mean([r["variant_prob"][dom[d]] / max(sum(r["variant_prob"][t] for t in var[d]), 1e-12) for r in rows])) for d in range(n)}
    out = {"dominant_variant": dom, "dominant_share_within_digit": share, "heads": {}}
    heads = list(rows[0]["heads"])
    for h in heads:
        dd_dom, dd_first, dl = [], [], []
        for r in rows:
            H = r["heads"][h]
            m = r["mention"]
            vd = np.array([H["dla_by_token"][dom[d]] for d in range(n)])
            vf = np.array([H["dla_by_token"][var[d][0]] for d in range(n)])
            if m and m.isdigit() and int(m) < n:
                c = int(m)
                dd_dom.append(vd[c] - np.mean(np.delete(vd, c)))
                dd_first.append(vf[c] - np.mean(np.delete(vf, c)))
            elif m in LET:
                dl.append(np.mean([vd[c] - np.mean(np.delete(vd, c)) for c in range(n)]))
        out["heads"][h] = {"candidate": rows[0]["heads"][h]["candidate"],
                           "dla_at_c_dominant": float(np.mean(dd_dom)), "dla_at_c_first_variant": float(np.mean(dd_first)),
                           "dla_letters_dominant": float(np.mean(dl)),
                           "attn_digit": float(np.mean([r["heads"][h]["attn_mention"] for r in rows if r["mention"] and r["mention"].isdigit() and int(r["mention"]) < n])),
                           "attn_letter": float(np.mean([r["heads"][h]["attn_mention"] for r in rows if r["mention"] in LET]))}
    return out


# ---------------- F18 ----------------
def f18(path, sampling_path=None):
    rows = [json.loads(l) for l in open(path)]
    D = {(r["q"], r["kind"], r["mention"]): np.array(r["q_answer"]) for r in rows}
    nq = max(r["q"] for r in rows) + 1
    inc = collections.defaultdict(list); comp = collections.defaultdict(list); slope_x, slope_y = [], []
    comp_x, comp_y = [], []
    for q in range(nq):
        ctrl = np.mean([D[(q, "incidental", L)] for L in LET], 0)
        none = D[(q, "none", None)]
        e0 = float(np.dot(np.arange(10), none))
        ectrl = float(np.dot(np.arange(10), ctrl))
        for c in range(10):
            qi = D[(q, "incidental", str(c))]
            inc[c].append(qi[c] - ctrl[c])
            slope_x.append(c - ectrl); slope_y.append(float(np.dot(np.arange(10), qi)) - ectrl)
            qc = D[(q, "comparative", str(c))]
            others = np.mean([D[(q, "comparative", str(o))][c] for o in range(10) if o != c])
            comp[c].append(qc[c] - others)
            comp_x.append(c - e0); comp_y.append(float(np.dot(np.arange(10), qc)) - e0)
    fav = [(np.mean([D[(q, "incidental", L)] for L in LET], 0)[c], D[(q, "incidental", str(c))][c]) for q in range(nq) for c in range(10)]
    fav = [x for x in fav if x[0] >= .05]
    res = {"incidental_delta": cboot(inc), "comparative_diag": cboot(comp),
           "incidental_slope": float(np.polyfit(slope_x, slope_y, 1)[0]),
           "comparative_slope": float(np.polyfit(comp_x, comp_y, 1)[0]),
           "incidental_fav_rel": (sum(b for _, b in fav) / sum(a for a, _ in fav) - 1) if fav else float("nan"),
           "digit_mass": float(np.mean([r["digit_mass"] for r in rows]))}
    return res


# ---------------- F19 ----------------
def f19(path):
    acc = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r.get("q"):
            acc[(r["spec"], r["prompt"], r["cell"], r["mention"])].append(np.array(r["q"]))
    Q = {k: np.mean(v, 0) for k, v in acc.items()}
    specs = sorted({k[0] for k in Q}); cells = ["M", "C1", "C2", "C4", "C3"]
    out = {}
    for s in specs:
        out[s] = {}
        for cell in cells:
            vals = collections.defaultdict(list); fav = []
            for p in ("base", "uniform"):
                ctrl = [Q.get((s, p, cell, L)) for L in LET]
                if any(x is None for x in ctrl):
                    continue
                ctrl = np.mean(ctrl, 0)
                for c in range(8):
                    q = Q.get((s, p, cell, str(c)))
                    if q is None:
                        continue
                    vals[c].append(q[c] - ctrl[c])
                    if ctrl[c] >= .05:
                        fav.append((ctrl[c], q[c]))
            if vals:
                m, ci = cboot(vals, 2000)
                rel = (sum(b for _, b in fav) / sum(a for a, _ in fav) - 1) if fav else float("nan")
                out[s][cell] = {"delta": m, "ci95": ci, "fav_rel": rel, "cells": sum(len(v) for v in vals.values())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f16", nargs="*", default=[], help="free.jsonl:exact.jsonl")
    ap.add_argument("--f17")
    ap.add_argument("--f18", nargs="*", default=[])
    ap.add_argument("--f19")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for pair in a.f16:
        fr, ex = pair.split(":")
        r = f16(fr, ex); out.setdefault("f16", {})[Path(fr).stem] = r
        print(f"F16 {Path(fr).stem}: ctx={r['contexts']} n/ctx={r['samples_per_ctx']} starts x={r['starts_with_x_rate']:.3f} valid={r['valid_x_rate']:.3f} "
              f"TV={r['tv_mean']:.3f} (max {r['tv_max']:.3f}) noise={r['noise_tv_mean']:.3f} Δx exact={r['delta_exact']:+.4f} free={r['delta_free']:+.4f}")
    if a.f17:
        r = f17(a.f17); out["f17"] = r
        print("F17 dominant variants:", r["dominant_variant"], "share", {k: round(v, 3) for k, v in r["dominant_share_within_digit"].items()})
        for h, v in r["heads"].items():
            print(f"   {h:8s} cand={v['candidate']!s:5s} attn {v['attn_digit']:.3f}/{v['attn_letter']:.3f}  DLA(c) dominant {v['dla_at_c_dominant']:+.3f}  first-variant {v['dla_at_c_first_variant']:+.3f}  letters {v['dla_letters_dominant']:+.3f}")
    for p in a.f18:
        r = f18(p); out.setdefault("f18", {})[Path(p).stem] = r
        print(f"F18 {Path(p).stem:24s} incidental Δ={r['incidental_delta'][0]:+.4f} [{r['incidental_delta'][1][0]:+.4f},{r['incidental_delta'][1][1]:+.4f}] "
              f"fav {r['incidental_fav_rel']:+.2f} slope {r['incidental_slope']:+.3f} | comparative diag={r['comparative_diag'][0]:+.4f} "
              f"[{r['comparative_diag'][1][0]:+.4f},{r['comparative_diag'][1][1]:+.4f}] slope {r['comparative_slope']:+.3f} | digit mass {r['digit_mass']:.3f}")
    if a.f19:
        r = f19(a.f19); out["f19"] = r
        for s, cells in r.items():
            print(f"F19 {s:22s} " + "  ".join(f"{c}:{v['delta']:+.3f}[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] fav{v['fav_rel']:+.2f}" for c, v in cells.items()))
    a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
