#!/usr/bin/env python3
"""Values that the paper (when-random-isnt-random-paper/) left as placeholders, recomputed from
the saved experiment outputs. Analysis only, no new data. Writes results/paper_missing_values.json.

  1. Favored-digit table: exact template CI of Kimi K3, digit 2        (results/robust_stats4.json)
  2. Noise-ratio table: TV_letter - F for Qwen3.8 Flash                  (results/robust_stats.json)
  3. Figure 1 / anchoring table: 95% template CI of Delta_x for the prompt-text phrase
     "This is test case c." in the sampling grid (local models)          (data/<model>.jsonl)
  4. Thinking table: CIs of the cells that had none                      (data/f7_*, data/<model>.jsonl)
  5. Minimal-pair table: RE CIs of the two print-order conditions        (local_analysis/results/f2_part2_final.json)
  6. Weight-only OV table: random-head means and Gemma L41H11            (data/f20_*.json)
  7. Patching: R and the random-head control, with the random-set sizes  (data/f8c*_*.jsonl)
  8. Digit-token table (F14): check that every cell uses the plain instruction
  9. API models: OpenRouter identifiers, pinned providers, top-k, reasoning setting
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "local_analysis"))
LOCAL = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it", "Muse-Glimmer-30B"]
out = {}

# 1. Kimi K3 favored digits (cross-fitted, template CI)
rs4 = json.loads((HERE / "results/robust_stats4.json").read_text())
out["kimi_k3_favored"] = rs4["models"]["kimi-k3"]["favored_units"]

# 2. Noise-corrected letter-swap ratio, Qwen3.8 Flash
nc = json.loads((HERE / "results/robust_stats.json").read_text())["api"]
out["noise_ratio"] = {s: dict(v["noise_corrected"], TVl_minus_F=v["noise_corrected"]["TV_letter"] - v["noise_corrected"]["F"],
                              TVd_minus_F=v["noise_corrected"]["TV_digit"] - v["noise_corrected"]["F"])
                      for s, v in ((s, nc[s]) for s in nc)}

# 3. Prompt-text Delta_x with the template bootstrap of robust_stats.py (B = 10,000)
import robust_stats as rs  # noqa: E402
rs.rng = np.random.default_rng(20260926)
out["prompt_text_dx"] = {}
for m in LOCAL:
    cs = rs.cells(rs.load_local(HERE / f"data/{m}.jsonl"))
    allci, _ = rs.template_boot(cs, rs.dx)
    pt = [x for x in cs if x[0][3] == "prompt_nl"]
    ci, k = rs.template_boot(pt, rs.dx)
    out["prompt_text_dx"][m] = {"dx": rs.dx(pt), "ci95_template": ci, "n_templates": k,
                                "all_locations_dx": rs.dx(cs), "all_locations_ci95": allci}

# 4. Thinking mode (F7): each model on a fresh RNG, as in the report's separate runs
import analyze_extra as ax  # noqa: E402
out["thinking"] = {}
for m in ("Qwen3-8B", "Qwen3-32B"):
    ax.RNG = np.random.default_rng(7)
    r = ax.f7(str(HERE / f"data/f7_{m}.jsonl"), str(HERE / f"data/{m}.jsonl"))
    out["thinking"][m] = {k: r[k] for k in ("thinking_irrelevant", "thinking_print_const",
                                            "exact_nothink_irrelevant", "exact_nothink_print_const")}

# 5. Minimal pairs, pooled over 12 models
f2 = json.loads((HERE / "local_analysis/results/f2_part2_final.json").read_text())["pooled"]
out["minimal_pairs_pooled"] = {k.replace("part2_", ""): v["all"] for k, v in f2.items()}

# 6. Weight-only OV check
out["ov"] = {}
for m in ("Qwen3-8B", "Qwen3-8B-Base", "Qwen3-14B", "Qwen3-32B", "gemma-4-31B-it"):
    d = json.loads((HERE / f"data/f20_{m}.json").read_text())
    R = d["random"].values()
    out["ov"][m] = {
        "candidates": {h: {e: {"z": v[e]["diag_z_mean"], "own_lowest": v[e]["frac_diag_min"]} for e in ("raw", "effective")}
                       for h, v in d["candidates"].items()},
        "random": {"n": len(R),
                   **{f"{e}_{s}": float(np.mean([v[e][key] for v in R]))
                      for e in ("raw", "effective") for s, key in (("z", "diag_z_mean"), ("own_lowest", "frac_diag_min"))},
                   "effective_z_range": [min(v["effective"]["diag_z_mean"] for v in R), max(v["effective"]["diag_z_mean"] for v in R)]}}

# 7. Activation patching (ratio of mean differences over contexts, letters and digits c < 8)
out["patching"] = {}
for f in ("f8c_Qwen3-8B", "f8c_Qwen3-32B", "f8c10_Qwen3-32B"):
    rows = [json.loads(l) for l in (HERE / f"data/{f}.jsonl").read_text().splitlines()]
    full = np.mean([r["q_digit"][r["c"]] - r["q_letter"][r["c"]] for r in rows])
    pat = np.mean([r["q_patch"][r["c"]] - r["q_letter"][r["c"]] for r in rows])
    rnd = np.mean([np.mean([q[r["c"]] for q in r["q_patch_random"]]) - r["q_letter"][r["c"]] for r in rows])
    out["patching"][f] = {"rows": len(rows), "R_heads": pat / full, "R_random": rnd / full,
                          "random_sets": len(rows[0]["q_patch_random"]),
                          "locations": sorted({r["location"] for r in rows}), "prompts": sorted({r["prompt"] for r in rows})}

# 8. F14 digit-token table: plain instruction ('base'), control = mean of the four letter mentions
spec = {"Qwen3-8B": 17, "Qwen3-14B": 14, "Qwen3-32B": 13, "gemma-4-31B-it": 12, "Qwen3-8B-Base": 15, "Muse-Glimmer-30B": 13}
out["f14_plain"] = {}
for m, t in spec.items():
    f = HERE / (f"data/f14raw_{m}.jsonl" if m.endswith("Base") else f"data/f14_{m}.jsonl")
    D = {}
    for l in f.read_text().splitlines():
        r = json.loads(l); q = np.asarray(r["q"]); D[(r["prompt"], r["cid"])] = q / q.sum()
    row = {}
    for p in ("base", "uniform"):
        g = lambda cid: D[(p, cid)][t]  # noqa: E731
        row[p] = {"control": float(np.mean([g(f"assign_down|{L}") for L in "ABKQ"])), "full": float(g(f"assign_down|{t}")),
                  "last": float(g(f"assign_down|{t % 10}")), "last_out": float(g(f"assign_down|{20 + t % 10}")),
                  "first": float(g(f"assign_down|{t // 10}"))}
    out["f14_plain"][m] = {"target": t, **row}

# 9. API models of the logprob arm (main grid)
api = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for l in (HERE / "api_data/logprobs.jsonl").read_text().splitlines():
    r = json.loads(l)
    for k in ("model", "provider", "top_logprobs"):
        api[r["spec"]][k][str(r.get(k))] += 1
    api[r["spec"]]["reasoning_cfg"][json.dumps(r.get("reasoning_cfg"))] += 1
    api[r["spec"]]["ok"][str(r.get("ok"))] += 1
out["api_models"] = {s: {k: dict(v) for k, v in d.items()} for s, d in api.items()}

(HERE / "results/paper_missing_values.json").write_text(json.dumps(out, indent=1, default=float))
print("wrote results/paper_missing_values.json")
