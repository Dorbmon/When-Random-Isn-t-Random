#!/usr/bin/env python3
"""Reviewer statistics, part 4: inference that treats template AND digit identity as sources of
variation, plus a favored-value conditional analysis with favored digits as the unit.

  A. Two-way (pigeonhole) cluster bootstrap: templates and digits resampled independently;
     each (template, digit) cell gets weight w_t * w_c (Owen 2007). B = 10,000.
  B. Two-way cluster-robust SE (Cameron, Gelbach & Miller 2011) for the mean Δ, with t(G_min - 1).
  C. Crossed random-effects model, REML: Δ_tc = β + u_t + v_c + ε (statsmodels MixedLM).
  D. Favored-value conditional effect, cross-fitted: favored digits are chosen with letters
     {A, B} and the effect is measured against {K, Q}, and vice versa (averaged), so selection
     and estimation use disjoint controls (no regression to the mean). Unit = (model, favored digit).
Analysis only; no new data."""
import collections
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

import robust_stats as RS

B = 10000
rng = np.random.default_rng(4242)
LOCAL = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"]
API = ["deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-6-luna", "gpt-5.6-luna", "qwen3.8-flash", "kimi-k3", "gpt-6-sol"]
SPLITS = ((("A", "B"), ("K", "Q")), (("K", "Q"), ("A", "B")))


def grid(T, ctrl=RS.LET):
    """Matrices over (template, digit): q_mention, q_ctrl, mask."""
    temps = sorted(t for t, d in T.items() if all(L in d for L in RS.LET))
    Qm = np.full((len(temps), 10), np.nan); Qc = np.full((len(temps), 10), np.nan)
    for i, t in enumerate(temps):
        d = T[t]; c_ = np.mean([d[L] for L in ctrl], 0)
        for c in range(t[1]):
            if str(c) in d:
                Qm[i, c] = d[str(c)][c]; Qc[i, c] = c_[c]
    keep = ~np.all(np.isnan(Qm), 0)
    return temps, Qm[:, keep], Qc[:, keep], np.arange(10)[keep]


def pigeonhole(Qm, Qc):
    M = ~np.isnan(Qm); D = np.where(M, Qm - Qc, 0.0)
    fav = M & (np.nan_to_num(Qc) >= 0.05)
    A = np.where(fav, Qm, 0.0); Bf = np.where(fav, Qc, 0.0)
    kt, kc = Qm.shape
    Wt = rng.multinomial(kt, np.full(kt, 1 / kt), B).astype(float)
    Wc = rng.multinomial(kc, np.full(kc, 1 / kc), B).astype(float)
    num = np.einsum("bt,tc,bc->b", Wt, D, Wc); den = np.einsum("bt,tc,bc->b", Wt, M.astype(float), Wc)
    dx = num / den
    a = np.einsum("bt,tc,bc->b", Wt, A, Wc); b = np.einsum("bt,tc,bc->b", Wt, Bf, Wc)
    R = np.where(b > 0, a / np.where(b > 0, b, 1) - 1, np.nan)
    return {"dx_ci95": np.percentile(dx, [2.5, 97.5]).tolist(),
            "R_ci95": np.nanpercentile(R, [2.5, 97.5]).tolist(), "R_undefined_share": float(np.isnan(R).mean())}


def cgm(Qm, Qc):
    M = ~np.isnan(Qm); d = (Qm - Qc)[M]; ti, ci = np.nonzero(M)
    N = d.size; e = d - d.mean()
    St = np.bincount(ti, e); Sc = np.bincount(ci, e)
    Vt = (St ** 2).sum() / N ** 2; Vc = (Sc ** 2).sum() / N ** 2; V0 = (e ** 2).sum() / N ** 2
    V = Vt + Vc - V0
    if V <= 0:
        V = Vt + Vc
    G = min(len(St), len(Sc)); se = float(np.sqrt(V)); tq = stats.t.ppf(0.975, G - 1)
    return {"mean": float(d.mean()), "se": se, "df": G - 1, "ci95": [float(d.mean() - tq * se), float(d.mean() + tq * se)],
            "se_template_only": float(np.sqrt(Vt)), "se_digit_only": float(np.sqrt(Vc))}


def mixed(Qm, Qc, digits):
    M = ~np.isnan(Qm); ti, ci = np.nonzero(M)
    df = pd.DataFrame({"d": (Qm - Qc)[M], "t": ti, "c": digits[ci], "g": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("d ~ 1", df, groups="g", vc_formula={"t": "0 + C(t)", "c": "0 + C(c)"}, re_formula="0")
        r = m.fit(reml=True, method=["lbfgs", "powell"])
    b, se = float(r.fe_params["Intercept"]), float(r.bse_fe["Intercept"])
    vc = dict(zip(r.model.exog_vc.names, np.asarray(r.vcomp).tolist()))
    tq = stats.t.ppf(0.975, len(set(df.c)) - 1)  # conservative df: number of digits − 1
    return {"beta": b, "se": se, "ci95_z": [b - 1.96 * se, b + 1.96 * se], "ci95_t_digits": [b - tq * se, b + tq * se],
            "var_template": vc.get("t"), "var_digit": vc.get("c"), "var_resid": float(r.scale), "converged": bool(r.converged)}


def favored_units(T):
    """Cross-fitted per-favored-digit relative change. Returns {digit: {...}}."""
    per = collections.defaultdict(list)
    for sel, est in SPLITS:
        _, _, Qs, digs = grid(T, sel)
        _, Qm, Qe, _ = grid(T, est)
        favd = [j for j in range(len(digs)) if np.nanmean(Qs[:, j]) >= 0.05]
        for j in favd:
            ok = ~np.isnan(Qm[:, j])
            qm, qe = Qm[ok, j], Qe[ok, j]
            idx = rng.integers(0, ok.sum(), (B, ok.sum()))
            bs = qm[idx].sum(1) / qe[idx].sum(1) - 1
            per[int(digs[j])].append((qm.sum() / qe.sum() - 1, bs, int(ok.sum())))
    out = {}
    for c, lst in per.items():  # average over the splits in which c was selected
        R = float(np.mean([x[0] for x in lst])); bs = np.mean([x[1] for x in lst], 0)
        out[c] = {"R": R, "ci95_template": np.percentile(bs, [2.5, 97.5]).tolist(), "se": float(bs.std()),
                  "n_splits": len(lst), "n_templates": lst[0][2]}
    return out


def main():
    sets = [(m, RS.load_local(f"data/{m}.jsonl")) for m in LOCAL] + \
           [(s, RS.load_api("api_data/logprobs.jsonl", s)) for s in API]
    out = {"models": {}}
    print(f"{'model':20s} | pigeonhole Δx CI      R CI            | CGM Δx CI (t df)        | mixed β [CI z] (var_t, var_c) | favored units")
    units = []
    for m, T in sets:
        temps, Qm, Qc, digs = grid(T)
        ph = pigeonhole(Qm, Qc); cg = cgm(Qm, Qc); mx = mixed(Qm, Qc, digs); fu = favored_units(T)
        out["models"][m] = {"pigeonhole": ph, "cgm": cg, "mixed": mx, "favored_units": fu}
        for c, v in fu.items():
            units.append((m, c, v["R"], v["se"]))
        print(f"{m:20s} | [{ph['dx_ci95'][0]:+.4f},{ph['dx_ci95'][1]:+.4f}] [{ph['R_ci95'][0]:+.2f},{ph['R_ci95'][1]:+.2f}] "
              f"(undef {ph['R_undefined_share']:.3f}) | {cg['mean']:+.4f} [{cg['ci95'][0]:+.4f},{cg['ci95'][1]:+.4f}] df{cg['df']} | "
              f"{mx['beta']:+.4f} [{mx['ci95_z'][0]:+.4f},{mx['ci95_z'][1]:+.4f}] t9 [{mx['ci95_t_digits'][0]:+.4f},{mx['ci95_t_digits'][1]:+.4f}] "
              f"({mx['var_template']:.1e},{mx['var_digit']:.1e},{mx['var_resid']:.1e}) conv={mx['converged']} | "
              + ", ".join(f"{c}: {v['R']:+.2f} [{v['ci95_template'][0]:+.2f},{v['ci95_template'][1]:+.2f}]" for c, v in sorted(fu.items())))
    # ---- pooled favored-unit inference
    Rs = np.array([u[2] for u in units]); ses = np.array([u[3] for u in units])
    models = sorted({u[0] for u in units})
    by_m = {mm: np.array([u[2] for u in units if u[0] == mm]) for mm in models}
    neg = int((Rs < 0).sum()); sig = int(sum(1 for u in units
                                             if out["models"][u[0]]["favored_units"][u[1]]["ci95_template"][1] < 0))
    sign_p = float(stats.binomtest(neg, len(Rs), 0.5, alternative="greater").pvalue)
    # model-level sign test (each model's mean over its favored units), 13 independent units
    mmeans = np.array([by_m[mm].mean() for mm in models])
    sign_p_models = float(stats.binomtest(int((mmeans < 0).sum()), len(mmeans), 0.5, alternative="greater").pvalue)
    # two-stage bootstrap: models, then favored digits within model
    bs = np.empty(B)
    for b in range(B):
        ms = rng.integers(0, len(models), len(models))
        bs[b] = np.mean([rng.choice(by_m[models[i]], len(by_m[models[i]])).mean() for i in ms])
    ew = float(mmeans.mean())
    # random-effects meta-analysis over units (DerSimonian–Laird), unit SE from template bootstrap
    w = 1 / ses ** 2; mu_f = (w * Rs).sum() / w.sum(); Q = (w * (Rs - mu_f) ** 2).sum(); k = len(Rs)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (ses ** 2 + tau2); mu = (ws * Rs).sum() / ws.sum(); se_mu = np.sqrt(1 / ws.sum())
    pooled = {"n_units": k, "n_negative": neg, "n_ci_below_0": sig, "sign_p_units": sign_p,
              "n_models_negative": int((mmeans < 0).sum()), "sign_p_models": sign_p_models,
              "equal_weight_by_model": ew, "ew_ci95_two_stage": np.percentile(bs, [2.5, 97.5]).tolist(),
              "RE_mean": float(mu), "RE_ci95": [float(mu - 1.96 * se_mu), float(mu + 1.96 * se_mu)], "tau": float(np.sqrt(tau2)),
              "local_only": {"n_units": int(sum(u[0] in LOCAL for u in units)),
                             "n_negative": int(sum(u[0] in LOCAL and u[2] < 0 for u in units))}}
    out["pooled_favored"] = pooled
    print("\nFavored-value units (cross-fitted):", json.dumps(pooled, indent=1, default=float))
    Path("results/robust_stats4.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()


# ---------------- pooled across models (appended) ----------------
def dl(est, se):
    est, se = np.asarray(est), np.asarray(se)
    w = 1 / se ** 2; mf = (w * est).sum() / w.sum(); Q = (w * (est - mf) ** 2).sum(); k = len(est)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2); mu = (ws * est).sum() / ws.sum(); s = np.sqrt(1 / ws.sum())
    return {"mean": float(mu), "ci95": [float(mu - 1.96 * s), float(mu + 1.96 * s)], "tau": float(np.sqrt(tau2)), "k": k}


def pooled_models():
    res = json.load(open("results/robust_stats4.json"))
    M = res["models"]; out = {}
    for name, grp in (("all13", LOCAL + API), ("local6", LOCAL), ("api7", API)):
        out[name] = {"cgm_DL": dl([M[m]["cgm"]["mean"] for m in grp], [M[m]["cgm"]["se"] for m in grp]),
                     "mixed_DL": dl([M[m]["mixed"]["beta"] for m in grp], [M[m]["mixed"]["se"] for m in grp]),
                     "n_negative_dx": int(sum(M[m]["cgm"]["mean"] < 0 for m in grp)),
                     "sign_p": float(stats.binomtest(int(sum(M[m]["cgm"]["mean"] < 0 for m in grp)), len(grp), 0.5,
                                                     alternative="greater").pvalue)}
    # one crossed mixed model over all 13 models: digit identity crossed across models. The
    # model:template component is left in the residual: per-model fits put it at <= 2e-3, far below
    # the digit component, and 780 extra columns make REML intractably slow here.
    rows = []
    sets = [(m, RS.load_local(f"data/{m}.jsonl")) for m in LOCAL] + \
           [(s, RS.load_api("api_data/logprobs.jsonl", s)) for s in API]
    for m, T in sets:
        temps, Qm, Qc, digs = grid(T)
        Mk = ~np.isnan(Qm); ti, ci = np.nonzero(Mk)
        for d, t, c in zip((Qm - Qc)[Mk], ti, digs[ci]):
            rows.append((d, m, f"{m}|{t}", int(c), f"{m}|{c}"))
    df = pd.DataFrame(rows, columns=["d", "m", "mt", "c", "mc"]); df["g"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = smf.mixedlm("d ~ 1", df, groups="g", re_formula="0",
                        vc_formula={"m": "0 + C(m)", "c": "0 + C(c)", "mc": "0 + C(mc)"}).fit(reml=True, method=["lbfgs"])
    b, se = float(r.fe_params["Intercept"]), float(r.bse_fe["Intercept"])
    out["pooled_crossed_mixed"] = {"beta": b, "se": se, "ci95_z": [b - 1.96 * se, b + 1.96 * se],
                                   "ci95_t9": [b - stats.t.ppf(.975, 9) * se, b + stats.t.ppf(.975, 9) * se],
                                   "vcomp": dict(zip(r.model.exog_vc.names, np.asarray(r.vcomp).tolist())),
                                   "resid": float(r.scale), "n": len(df), "converged": bool(r.converged)}
    res["pooled_models"] = out
    Path("results/robust_stats4.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    pooled_models()
