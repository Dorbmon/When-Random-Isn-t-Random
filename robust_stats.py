#!/usr/bin/env python3
"""Reviewer statistics: template-cluster CIs, diagnostics, C1 minimal margin, invariance index,
and the API noise-corrected letter-swap ratio. Analysis only (no new data)."""
import collections
import json
from pathlib import Path

import numpy as np

B = 10000
LET = ("A", "B", "K", "Q")
LOCS = ("comment_down", "assign_down", "varname_down", "comment_up", "prompt_nl")
rng = np.random.default_rng(20260926)


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# ---------------- loaders: {template: {mention: q_x}} ; template = (prompt, n, expr, loc)
def load_local(path):
    T = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if r["location"] in LOCS:
            q = np.asarray(r["q_x"]); T[(r["prompt"], r["n"], r["expr"], r["location"])][r["mention"]] = q / q.sum()
    return T


def load_api(path, spec, repeat=None):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(path):
        r = json.loads(l)
        if r["spec"] != spec or not r.get("ok") or r["location"] not in LOCS:
            continue
        if repeat is not None and r["repeat"] != repeat:
            continue
        p = np.asarray(r["x_lp"]["digit_probs"][:r["n"]])
        if p.sum() > 0:
            acc[(r["prompt"], r["n"], r["expr"], r["location"])][r["mention"]].append(p / p.sum())
    return {t: {m: np.mean(v, 0) for m, v in d.items()} for t, d in acc.items()}


def cells(T):
    """(template, c, q_mention(c), q_ctrl(c)) for in-range digit c with all letter controls present."""
    out = []
    for t, d in T.items():
        if any(L not in d for L in LET):
            continue
        ctrl = np.mean([d[L] for L in LET], 0)
        n = t[1]
        for c in range(n):
            if str(c) in d:
                out.append((t, c, float(d[str(c)][c]), float(ctrl[c])))
    return out


def R_of(cs):
    fav = [x for x in cs if x[3] >= 0.05]
    return (sum(x[2] for x in fav) / sum(x[3] for x in fav) - 1) if fav else float("nan")


def template_boot(cs, stat):
    """Cluster bootstrap over templates, vectorized via per-template sums. stat in {R_of, dx}."""
    temps = sorted({x[0] for x in cs})
    ti = {t: i for i, t in enumerate(temps)}
    k = len(temps)
    a = np.zeros(k); b = np.zeros(k); dsum = np.zeros(k); cnt = np.zeros(k)
    for (t, c, qm, qc) in cs:
        i = ti[t]
        if qc >= 0.05:
            a[i] += qm; b[i] += qc
        dsum[i] += qm - qc; cnt[i] += 1
    idx = rng.integers(0, k, (B, k))
    if stat is R_of:
        num = a[idx].sum(1); den = b[idx].sum(1)
        vals = np.where(den > 0, num / np.where(den > 0, den, 1) - 1, np.nan)
    else:
        vals = dsum[idx].sum(1) / cnt[idx].sum(1)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], k


def dx(cs):
    return float(np.mean([x[2] - x[3] for x in cs]))


def digit_diag(cs):
    """(a) share of digit-bootstrap resamples with no favored digit; (b) mean dx of non-favored digits."""
    by = collections.defaultdict(list)
    for x in cs:
        by[x[1]].append(x)
    digits = sorted(by)
    fav_digits = {c for c in digits if np.mean([x[3] for x in by[c]]) >= 0.05}
    idx = rng.integers(0, len(digits), (B, len(digits)))
    none = np.mean([not any(digits[i] in fav_digits for i in row) for row in idx])
    nonfav = [x[2] - x[3] for x in cs if x[1] not in fav_digits]
    return float(none), float(np.mean(nonfav)) if nonfav else float("nan"), sorted(fav_digits)


def invariance(T):
    I, I0, E = [], [], []
    for t, d in T.items():
        if any(L not in d for L in LET):
            continue
        n = t[1]
        ctrl = np.mean([d[L] for L in LET], 0)
        for c in range(n):
            if str(c) in d:
                I.append(tv(d[str(c)], ctrl))
        for i in range(4):
            for j in range(i + 1, 4):
                I0.append(tv(d[LET[i]], d[LET[j]]))
        E.append(tv(ctrl, np.full(n, 1 / n)))
    I, I0, E = np.mean(I), np.mean(I0), np.mean(E)
    return {"I": float(I), "I0": float(I0), "E": float(E), "ratio": float((I - I0) / E)}


def main():
    out = {"local": {}, "api": {}}
    local = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"]
    api = ["deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-6-luna", "gpt-5.6-luna", "qwen3.8-flash", "kimi-k3", "gpt-6-sol"]
    sets = [("local", m, load_local(f"data/{m}.jsonl")) for m in local] + \
           [("api", s, load_api("api_data/logprobs.jsonl", s)) for s in api]
    print(f"{'model':22s} {'R':>7s} {'R 95% CI (template)':>22s} {'Δx':>8s} {'Δx CI template':>20s} | diag: no-fav share, non-fav Δx, fav digits | (I−I0)/E  I  I0  E")
    for grp, m, T in sets:
        cs = cells(T)
        R = R_of(cs); Rci, nt = template_boot(cs, R_of)
        d = dx(cs); dci, _ = template_boot(cs, dx)
        none, nonfav, favd = digit_diag(cs)
        inv = invariance(T)
        out[grp][m] = {"R": R, "R_ci95_template": Rci, "n_templates": nt, "dx": d, "dx_ci95_template": dci,
                       "digitboot_no_favored_share": none, "nonfavored_dx": nonfav, "favored_digits": favd,
                       "invariance": inv}
        print(f"{m:22s} {R:+7.3f} [{Rci[0]:+.3f},{Rci[1]:+.3f}] {d:+8.4f} [{dci[0]:+.4f},{dci[1]:+.4f}] | {none:.3f} {nonfav:+.5f} {favd} | "
              f"{inv['ratio']:.3f} {inv['I']:.3f} {inv['I0']:.4f} {inv['E']:.3f}")
    # ---- API noise-corrected letter-swap ratio (single repeats for TV_digit/TV_letter; F between repeats)
    print("\nAPI noise-corrected letter-swap ratio")
    for s in api:
        T0 = load_api("api_data/logprobs.jsonl", s, repeat=0); T1 = load_api("api_data/logprobs.jsonl", s, repeat=1)
        inv0 = invariance(T0)
        F = np.mean([tv(T0[t][m], T1[t][m]) for t in T0 if t in T1 for m in T0[t] if m in T1[t]])
        raw = inv0["I"] / inv0["I0"]
        corr = (inv0["I"] - F) / (inv0["I0"] - F) if inv0["I0"] > F else float("nan")
        out["api"][s]["noise_corrected"] = {"TV_digit": inv0["I"], "TV_letter": inv0["I0"], "F": float(F), "raw_ratio": raw, "corrected_ratio": corr}
        print(f"  {s:22s} TV_digit {inv0['I']:.3f} TV_letter {inv0['I0']:.3f} F {F:.3f}  raw {raw:.2f}  corrected {corr:.2f}")
    Path("results/robust_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
