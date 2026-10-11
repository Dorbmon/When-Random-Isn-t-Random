#!/usr/bin/env python3
"""Markdown tables for the spurious-association section (from results/spurious_assoc.json)."""
import json
NM = {"Qwen3-4B": "Qwen3-4B", "Qwen3-8B": "Qwen3-8B", "Qwen3-14B": "Qwen3-14B", "Qwen3-32B": "Qwen3-32B", "gemma-4-31B-it": "gemma-4-31B-it",
      "Muse-Glimmer-30B": "Muse-Glimmer-30B", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash", "deepseek-v4-pro": "DeepSeek V4 Pro",
      "gpt-6-luna": "GPT-6 Luna", "gpt-5.6-luna": "GPT-5.6 Luna", "qwen3.8-flash": "Qwen3.8 Flash", "kimi-k3": "Kimi K3", "gpt-6-sol": "GPT-6 Sol"}
sp = json.load(open("results/spurious_assoc.json"))
f = lambda x: f"{x:+.3f}".replace("-", "−")
pct = lambda x: f"{x * 100:+.0f}%".replace("-", "−")
LV = ["retries", "timeout", "default_arg", "config_dict", "loop_bound"]
print("| Model | MAX_RETRIES = d | TIMEOUT = d | bonus=d default arg | {'workers': d} | range(d) loop |\n| --- | --- | --- | --- | --- | --- |")
for m, d in sp["f5"].items():
    if not d:
        print(f"| {NM[m]} | n/a | n/a | n/a | n/a | n/a |"); continue
    cells = []
    for l in LV:
        v = d[l]; N = v["N80"]
        cells.append(f"{v.get('MI_excess', v['MI_bits']):.3f} / {v.get('V_excess', v['V']):.2f} / {'∞' if N == float('inf') else f'{N:,.0f}'} / {pct(v['diag_rel'])}")
    print(f"| {NM[m]} | " + " | ".join(cells) + " |")
print("\n| Model | I(C; X), bits | Noise floor (API) | Cramér's V | N for 80% power (mean w²) | N, median template | Diagonal Δ (relative) | Templates |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
for m, v in sp["main_grid"].items():
    fl = v.get("MI_floor")
    print(f"| {NM[m]} | {v.get('MI_excess', v['MI_bits']):.3f} | {'—' if fl is None else f'{fl:.3f} (raw {v[chr(77)+chr(73)+chr(95)+chr(98)+chr(105)+chr(116)+chr(115)]:.3f})'} | "
          f"{v.get('V_excess', v['V']):.2f} | {v['N80']:,.0f} | {v['N80_median_over_templates']:,.0f} | {f(v['diag'])} ({pct(v['diag_rel'])}) | {v['n_templates']} |")
