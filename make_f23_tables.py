#!/usr/bin/env python3
"""Markdown tables for F23 (from results/f23.json)."""
import json
import sys
NM = {"Qwen3-4B": "Qwen3-4B", "Qwen3-8B": "Qwen3-8B", "Qwen3-14B": "Qwen3-14B", "Qwen3-32B": "Qwen3-32B", "gemma-4-31B-it": "gemma-4-31B-it",
      "Muse-Glimmer-30B": "Muse-Glimmer-30B", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash", "deepseek-v4-pro": "DeepSeek V4 Pro",
      "gpt-6-luna": "GPT-6 Luna", "gpt-6-sol": "GPT-6 Sol"}
r = json.load(open("results/f23.json"))
pct = lambda x: f"{x * 100:+.0f}%".replace("-", "−")
ci = lambda c: f"[{c[0] * 100:+.0f}, {c[1] * 100:+.0f}]".replace("-", "−")
f4 = lambda x: f"{x:+.3f}".replace("-", "−")
ci4 = lambda c: f"[{f4(c[0])}, {f4(c[1])}]"
print("| Model | Median N for 80% power | Snippets with N ≤ 100 / 1,000 / 10,000 | Diagonal, all constants | Diagonal, compared with the drawn variable (33) | Diagonal, other constants (55) |")
print("| --- | --- | --- | --- | --- | --- |")
for m, v in r.items():
    s, c, o = v["summary"], v["summary_compare_with_draw_var"], v["summary_other"]
    print(f"| {NM[m]} | {s['N80_median']:,.0f} | {s['share_N_le_100']:.0%} / {s['share_N_le_1000']:.0%} / {s['share_N_le_10000']:.0%} | "
          f"{pct(s['diag_rel_pooled'])} {ci(s['diag_rel_ci95'])} | {pct(c['diag_rel_pooled'])} {ci(c['diag_rel_ci95'])} | {pct(o['diag_rel_pooled'])} {ci(o['diag_rel_ci95'])} |")
print()
print("| Model | Δ0, other constants (n = 29) | Wilcoxon p | Δ0, compared constants (n = 32) | Favored c0 (control q ≥ 0.05), all: change [95% CI] |")
print("| --- | --- | --- | --- | --- |")
from scipy import stats
import numpy as np
for m, v in r.items():
    o, c, s = v["summary_other"], v["summary_compare_with_draw_var"], v["summary"]
    d0o = np.array([u["d0"] for u in v["units"] if "d0" in u and u["sid"] in {x for x in json.load(open("f23/categories.json")) if json.load(open("f23/categories.json"))[x] == "other"}])
    p = stats.wilcoxon(d0o).pvalue if np.any(d0o != 0) else float("nan")
    print(f"| {NM[m]} | {f4(o['d0_mean'])} {ci4(o['d0_ci95'])} | {p:.2g} | {f4(c['d0_mean'])} {ci4(c['d0_ci95'])} | "
          f"{pct(s['R_fav'])} {ci(s['R_fav_ci95'])} (n = {s['fav_units']}) |")
