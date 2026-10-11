#!/usr/bin/env python3
"""F23 (real-code counterfactuals) and F24 (natural-language dice) analysis.

Per unit (snippet or dice scenario), C = the varied constant (legal values plus out-of-range digits,
uniform) and X = the draw restricted to its legal values. Dependence: I(C; X), Cramér's V, and N for
80% power of a chi-square independence test (alpha 0.05), as in spurious_assoc.py; API models use the
excess over the repeat-noise floor. Diagonal: q(c | C = c) against the mean of q(c | C = c') over the
other values. F23 endpoint 1: for units whose original constant c0 is legal,
Δ0 = q(c0 | original) − mean over out-of-range variants of q(c0)."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

from spurious_assoc import assoc, n80

rng = np.random.default_rng(2324)
LOCAL = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"]
API = ["deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-6-luna", "gpt-6-sol"]


def restrict(q, legal):
    v = np.array([q[d] for d in legal], float)
    return v / v.sum() if v.sum() > 0 else np.full(len(legal), 1 / len(legal))


def load_local(path):
    U = collections.defaultdict(dict); meta = {}
    for l in open(path):
        r = json.loads(l)
        U[r["sid"]][r["c"]] = restrict(r["q"], r["legal"]); meta[r["sid"]] = r
    return {s: ({c: {0: v} for c, v in d.items()}, meta[s]) for s, d in U.items()}


def load_api(path, spec):
    U = collections.defaultdict(lambda: collections.defaultdict(dict)); meta = {}
    for l in open(path):
        r = json.loads(l)
        if r["spec"] != spec or not r.get("ok"):
            continue
        U[r["sid"]][r["c"]][r["repeat"]] = restrict(r["lp"]["digit_probs"], r["legal"]); meta[r["sid"]] = r
    return {s: (d, meta[s]) for s, d in U.items()}


def unit_stats(d, meta, api):
    legal = meta["legal"]; cvals = sorted(legal + meta["oor"])
    if any(c not in d for c in cvals) or (api and any(0 not in d[c] or 1 not in d[c] for c in cvals)):
        return None
    if api:
        Q0 = np.array([d[c][0] for c in cvals]); Q1 = np.array([d[c][1] for c in cvals]); Q = (Q0 + Q1) / 2
        Qn = np.clip(Q.mean(0, keepdims=True) + (Q0 - Q1) / 2, 1e-12, None); Qn /= Qn.sum(1, keepdims=True)
    else:
        Q = np.array([d[c][0] for c in cvals]); Qn = None
    k, n = Q.shape
    mi, w2, V = assoc(Q)
    r = {"sid": meta["sid"], "k": k, "n": n, "MI": mi, "w2": w2, "V": V}
    if api:
        mi0, w20, _ = assoc(Qn)
        r.update({"MI_floor": mi0, "MI": max(0.0, mi - mi0), "w2": max(0.0, w2 - w20)})
        r["V"] = float(np.sqrt(r["w2"] / min(k - 1, n - 1)))
    r["N80"] = n80(r["w2"], k, n)
    col = {v: j for j, v in enumerate(legal)}
    dd, on, off = [], 0.0, 0.0
    for i, c in enumerate(cvals):
        if c in col:
            o = np.mean([Q[j, col[c]] for j in range(k) if j != i])
            dd.append(Q[i, col[c]] - o); on += Q[i, col[c]]; off += o
    r["diag"] = float(np.mean(dd)); r["diag_rel"] = float(on / off - 1) if off > 0 else float("nan")
    r["diag_on"], r["diag_off"] = float(on), float(off)
    c0 = meta.get("c0")
    if c0 is not None and c0 in col:
        qo = Q[cvals.index(c0), col[c0]]
        qc = float(np.mean([Q[cvals.index(v), col[c0]] for v in meta["oor"]]))
        r.update({"c0": c0, "q_orig": float(qo), "q_oor": qc, "d0": float(qo - qc)})
    return r


def summarize(rows):
    N = np.array([r["N80"] for r in rows])
    s = {"units": len(rows), "MI_mean": float(np.mean([r["MI"] for r in rows])),
         "V_mean": float(np.mean([r["V"] for r in rows])), "N80_median": float(np.median(N)),
         "share_N_le_100": float(np.mean(N <= 100)), "share_N_le_1000": float(np.mean(N <= 1000)),
         "share_N_le_10000": float(np.mean(N <= 10000)),
         "diag_mean": float(np.mean([r["diag"] for r in rows])),
         "diag_share_negative": float(np.mean([r["diag"] < 0 for r in rows]))}
    oo = np.array([[r["diag_on"], r["diag_off"]] for r in rows])
    s["diag_rel_pooled"] = float(oo[:, 0].sum() / oo[:, 1].sum() - 1)  # ratio of sums over units
    ix = rng.integers(0, len(rows), (10000, len(rows)))
    s["diag_rel_ci95"] = np.percentile(oo[ix, 0].sum(1) / oo[ix, 1].sum(1) - 1, [2.5, 97.5]).tolist()
    e = [r for r in rows if "d0" in r]
    if e:
        d0 = np.array([r["d0"] for r in e])
        bs = d0[rng.integers(0, len(d0), (10000, len(d0)))].mean(1)
        fav = [r for r in e if r["q_oor"] >= 0.05]
        R = (sum(r["q_orig"] for r in fav) / sum(r["q_oor"] for r in fav) - 1) if fav else float("nan")
        if fav:
            fa = np.array([[r["q_orig"], r["q_oor"]] for r in fav]); ix = rng.integers(0, len(fav), (10000, len(fav)))
            Rb = fa[ix, 0].sum(1) / fa[ix, 1].sum(1) - 1
        s.update({"e1_units": len(e), "d0_mean": float(d0.mean()), "d0_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                  "d0_negative": int((d0 < 0).sum()),
                  "wilcoxon_p": float(stats.wilcoxon(d0).pvalue) if np.any(d0 != 0) else float("nan"),
                  "fav_units": len(fav), "R_fav": float(R),
                  "R_fav_ci95": np.percentile(Rb, [2.5, 97.5]).tolist() if fav else None})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=("f23", "f24"), required=True)
    a = ap.parse_args()
    out = {}
    for m in LOCAL + API:
        if m in LOCAL:
            p = Path(f"data/{a.exp}_{m}.jsonl")
            if not p.exists():
                continue
            U = load_local(p)
        else:
            p = Path(f"api_data/{a.exp}.jsonl")
            if not p.exists():
                continue
            U = load_api(p, m)
        rows = [x for x in (unit_stats(d, meta, m in API) for d, meta in U.values()) if x]
        if not rows:
            continue
        out[m] = {"summary": summarize(rows), "units": rows}
        if a.exp == "f23":
            cat = json.load(open("f23/categories.json"))
            for k in ("compare_with_draw_var", "other"):
                sub = [r for r in rows if cat.get(r["sid"]) == k]
                if sub:
                    out[m]["summary_" + k] = summarize(sub)
        s = out[m]["summary"]
        line = (f"{m:20s} units {s['units']:3d} | MI {s['MI_mean']:.3f} V {s['V_mean']:.2f} | median N80 {s['N80_median']:,.0f} | "
                f"N<=100 {s['share_N_le_100']:.0%} <=1000 {s['share_N_le_1000']:.0%} <=10000 {s['share_N_le_10000']:.0%} | "
                f"diag {s['diag_mean']:+.3f} ({s['diag_rel_pooled']:+.0%} [{s['diag_rel_ci95'][0]:+.0%}, {s['diag_rel_ci95'][1]:+.0%}], neg {s['diag_share_negative']:.0%})")
        if "d0_mean" in s:
            line += (f" | E1 n={s['e1_units']} Δ0 {s['d0_mean']:+.4f} {np.round(s['d0_ci95'], 4).tolist()} neg {s['d0_negative']} "
                     f"p={s['wilcoxon_p']:.2g} | R_fav({s['fav_units']}) {s['R_fav']:+.0%} {np.round(s['R_fav_ci95'], 2).tolist() if s['R_fav_ci95'] else ''}")
        print(line)
        for k in ("compare_with_draw_var", "other"):
            t = out[m].get("summary_" + k)
            if t:
                print(f"    {k:22s} units {t['units']:3d} MI {t['MI_mean']:.3f} median N80 {t['N80_median']:,.0f} N<=1000 {t['share_N_le_1000']:.0%} "
                      f"diag {t['diag_rel_pooled']:+.0%} [{t['diag_rel_ci95'][0]:+.0%}, {t['diag_rel_ci95'][1]:+.0%}] neg {t['diag_share_negative']:.0%}" + (f" | E1 n={t['e1_units']} Δ0 {t['d0_mean']:+.4f} {np.round(t['d0_ci95'], 4).tolist()} "
                      f"neg {t['d0_negative']} | R_fav({t['fav_units']}) {t['R_fav']:+.0%}" if "d0_mean" in t else ""))
    Path(f"results/{a.exp}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()


def pooled_f23(models, label):
    """Pool across models with snippets as the unit: per snippet, average over models; bootstrap snippets."""
    r = json.load(open("results/f23.json")); cat = json.load(open("f23/categories.json"))
    per = collections.defaultdict(dict)
    for m in models:
        if m not in r:
            return None
        for u in r[m]["units"]:
            per[u["sid"]][m] = u
    sids = [s for s in per if len(per[s]) == len(models)]
    out = {"models": models, "label": label}
    for k in ("all", "compare_with_draw_var", "other"):
        S = [s for s in sids if k == "all" or cat.get(s) == k]
        on = np.array([np.mean([per[s][m]["diag_on"] for m in models]) for s in S])
        off = np.array([np.mean([per[s][m]["diag_off"] for m in models]) for s in S])
        ix = rng.integers(0, len(S), (10000, len(S)))
        res = {"units": len(S), "diag_rel": float(on.sum() / off.sum() - 1),
               "diag_ci95": np.percentile(on[ix].sum(1) / off[ix].sum(1) - 1, [2.5, 97.5]).tolist(),
               "share_N_le_1000": float(np.mean([np.mean([per[s][m]["N80"] <= 1000 for m in models]) for s in S]))}
        E = [s for s in S if "d0" in per[s][models[0]]]
        if E:
            d0 = np.array([np.mean([per[s][m]["d0"] for m in models]) for s in E])
            ixe = rng.integers(0, len(E), (10000, len(E)))
            res.update({"e1_units": len(E), "d0": float(d0.mean()), "d0_ci95": np.percentile(d0[ixe].mean(1), [2.5, 97.5]).tolist(),
                        "wilcoxon_p": float(stats.wilcoxon(d0).pvalue)})
        out[k] = res
        print(f"{label:10s} {k:22s} n={res['units']:3d} diag {res['diag_rel']:+.0%} [{res['diag_ci95'][0]:+.0%}, {res['diag_ci95'][1]:+.0%}] "
              f"detectable<=1000 {res['share_N_le_1000']:.0%}" + (f" | Δ0 {res['d0']:+.4f} [{res['d0_ci95'][0]:+.4f}, {res['d0_ci95'][1]:+.4f}] "
              f"p={res['wilcoxon_p']:.2g} (n={res['e1_units']})" if E else ""))
    return out
