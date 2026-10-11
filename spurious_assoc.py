#!/usr/bin/env python3
"""Spurious association between an irrelevant constant C and the simulated draw X.

For each model and construct, C is taken as uniform over its values and the joint is
p(c, x) = q(x | C = c) / k. Reported per (model, construct):
  - mutual information I(C; X) in bits, and Cramér's V;
  - w^2 = sum (p_cx - p_c p_x)^2 / (p_c p_x) and the number of simulated records N at which a
    chi-square independence test (alpha = 0.05) reaches 80% power: noncentral chi-square with
    df = (k - 1)(n - 1) and lambda = N w^2;
  - the diagonal: P(x = c | C = c) minus the mean of P(x = c | C = c') over c' != c.
API models: q is averaged over 2 repeats. A noise floor comes from a null joint with no
dependence but the same call noise, q0(x) + (q_r0(x|c) - q_r1(x|c)) / 2 (the noise of a
2-repeat mean); only the excess over that floor is reported.
Data: F5 realistic constants (main), main grid (supplement). Analysis only."""
import collections
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, stats

import robust_stats as RS

LOCAL = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"]
F5_LOCAL = ["Qwen3-8B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"]
API = ["deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-6-luna", "gpt-5.6-luna", "qwen3.8-flash", "kimi-k3", "gpt-6-sol"]
LEVELS = ["retries", "timeout", "default_arg", "config_dict", "loop_bound"]


def assoc(Q):
    """Q: k x n matrix, row c = q(x | C = c). Returns MI (bits), w2, V."""
    Q = np.clip(Q, 0, None); Q = Q / Q.sum(1, keepdims=True)
    k, n = Q.shape
    P = Q / k; pc = P.sum(1, keepdims=True); px = P.sum(0, keepdims=True)
    E = pc * px
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(np.where(P > 0, P * np.log2(P / E), 0.0))
        w2 = np.nansum(np.where(E > 0, (P - E) ** 2 / E, 0.0))
    return float(mi), float(w2), float(np.sqrt(w2 / min(k - 1, n - 1)))


@lru_cache(maxsize=None)
def lam80(df):
    crit = stats.chi2.ppf(0.95, df)
    return optimize.brentq(lambda l: stats.ncx2.sf(crit, df, l) - 0.8, 1e-6, 1e5)


def n80(w2, k, n):
    return float(lam80((k - 1) * (n - 1)) / w2) if w2 > 0 else float("inf")


def diagonal(Q, cvals):
    """Mean over in-range c of P(x=c | C=c) - mean_{c' != c} P(x=c | C=c'), and its relative form."""
    n = Q.shape[1]; d, on, off = [], 0.0, 0.0
    for i, c in enumerate(cvals):
        if c < n:
            o = np.mean([Q[j, c] for j in range(len(cvals)) if j != i])
            d.append(Q[i, c] - o); on += Q[i, c]; off += o
    return float(np.mean(d)), float(on / off - 1) if off > 0 else float("nan"), float(on), float(off)


def summarize(Q, cvals, Qnull=None):
    k, n = Q.shape
    mi, w2, V = assoc(Q)
    r = {"k": k, "n": n, "MI_bits": mi, "w2": w2, "V": V}
    if Qnull is not None:
        mi0, w20, _ = assoc(Qnull)
        r.update({"MI_floor": mi0, "w2_floor": w20, "MI_excess": max(0.0, mi - mi0), "w2_excess": max(0.0, w2 - w20)})
        w2 = r["w2_excess"]; r["V_excess"] = float(np.sqrt(w2 / min(k - 1, n - 1)))
    r["N80"] = n80(w2, k, n)
    r["diag"], r["diag_rel"], r["diag_on"], r["diag_off"] = diagonal(Q, cvals)
    return r


def f5_local(m):
    D = collections.defaultdict(dict)
    for l in open(f"data/f5_{m}.jsonl"):
        r = json.loads(l); q = np.asarray(r["q_x"]); D[(r["prompt"], r["level"])][int(r["mention"])] = q / q.sum()
    return {k: (np.array([v[c] for c in sorted(v)]), sorted(v), None) for k, v in D.items()}


def f5_api(spec):
    D = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in ("api_data/logprobs_f5.jsonl", "api_data/logprobs_f5_qwen.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if r["spec"] != spec or not r.get("ok"):
                continue
            q = np.asarray(r["x_lp"]["digit_probs"][:r["n"]])
            if q.sum() > 0:
                D[(r["prompt"], r["level"])][int(r["mention"])][r["repeat"]] = q / q.sum()
    return {k: null_pair(v) for k, v in D.items()}


def null_pair(v):
    cs = sorted(c for c in v if 0 in v[c] and 1 in v[c])
    if len(cs) < 8:  # need (nearly) all constant values with both repeats
        return None
    Q0 = np.array([v[c][0] for c in cs]); Q1 = np.array([v[c][1] for c in cs])
    Q = (Q0 + Q1) / 2
    Qn = np.clip(Q.mean(0, keepdims=True) + (Q0 - Q1) / 2, 0, None)
    return Q, cs, Qn / Qn.sum(1, keepdims=True)


def main_grid(model, api=False):
    if api:
        T0 = RS.load_api("api_data/logprobs.jsonl", model, repeat=0); T1 = RS.load_api("api_data/logprobs.jsonl", model, repeat=1)
    else:
        T0 = RS.load_local(f"data/{model}.jsonl"); T1 = None
    out = []
    for t, d in T0.items():
        cs = sorted(int(m) for m in d if m.isdigit() and (T1 is None or (t in T1 and m in T1[t])))
        if len(cs) < 10:
            continue
        if api:
            npair = null_pair({c: {0: d[str(c)], 1: T1[t][str(c)]} for c in cs})
            if npair is not None:
                out.append(summarize(*npair))
        else:
            out.append(summarize(np.array([d[str(c)] for c in cs]), cs))
    return out


def agg(rows):
    key = "w2_excess" if "w2_excess" in rows[0] else "w2"
    w2 = float(np.mean([r[key] for r in rows])); k, n = rows[0]["k"], rows[0]["n"]
    res = {"MI_bits": float(np.mean([r["MI_bits"] for r in rows])), "V": float(np.mean([r["V"] for r in rows])),
           "w2": w2, "N80": n80(w2, k, n), "N80_median_over_templates": float(np.median([r["N80"] for r in rows])),
           "diag": float(np.mean([r["diag"] for r in rows])),
           "diag_rel": float(sum(r["diag_on"] for r in rows) / sum(r["diag_off"] for r in rows) - 1),  # ratio of sums
           "k": k, "n": n, "n_templates": len(rows)}
    if "MI_floor" in rows[0]:
        res.update({"MI_floor": float(np.mean([r["MI_floor"] for r in rows])),
                    "MI_excess": float(np.mean([r["MI_excess"] for r in rows])),
                    "V_excess": float(np.mean([r["V_excess"] for r in rows]))})
    return res


def main():
    out = {"f5": {}, "main_grid": {}}
    for m in F5_LOCAL + API:
        D = f5_local(m) if m in F5_LOCAL else f5_api(m)
        out["f5"][m] = {}
        for lvl in LEVELS:
            rows = [summarize(*D[(p, lvl)]) for p in ("base", "uniform") if (p, lvl) in D and D[(p, lvl)] is not None]
            if not rows:
                continue
            out["f5"][m][lvl] = agg(rows)
    for m in LOCAL + API:
        rows = main_grid(m, api=m in API)
        if rows:
            out["main_grid"][m] = agg(rows)
    # print
    print("F5 realistic constants: MI bits (excess over noise floor for API) | V | N80 | diag Δ (rel)")
    for m, d in out["f5"].items():
        print(f"{m:20s} " + " | ".join(
            f"{l[:6]}: {v.get('MI_excess', v['MI_bits']):.3f}b V{v.get('V_excess', v['V']):.2f} N{v['N80']:.0f} {v['diag']:+.3f}({v['diag_rel']:+.0%})"
            for l, v in d.items()))
    print("\nMain grid (per template, averaged): MI | V | N80 (from mean w2) | median N80 | diag")
    for m, v in out["main_grid"].items():
        print(f"{m:20s} MI {v.get('MI_excess', v['MI_bits']):.3f} (raw {v['MI_bits']:.3f}, floor {v.get('MI_floor', 0):.3f}) "
              f"V {v.get('V_excess', v['V']):.2f} N80 {v['N80']:.0f} median {v['N80_median_over_templates']:.0f} "
              f"diag {v['diag']:+.3f} ({v['diag_rel']:+.0%}) templates {v['n_templates']}")
    Path("results/spurious_assoc.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
