#!/usr/bin/env python3
"""Reviewer statistics, part 3: F5 scenario invariance for API models with a noise correction
from F5's own repeat calls. Pairwise TV across constant values uses repeat 0 only (single call vs
single call), so its noise level matches F = mean TV(repeat 0, repeat 1) of the same prompt."""
import collections
import json
from pathlib import Path

import numpy as np

SPECS = ("deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-6-luna", "gpt-5.6-luna", "qwen3.8-flash", "kimi-k3", "gpt-6-sol")


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def load(spec):
    D = collections.defaultdict(lambda: collections.defaultdict(dict))  # (prompt, level) -> mention -> repeat -> q
    for path in ("api_data/logprobs_f5.jsonl", "api_data/logprobs_f5_qwen.jsonl"):
        for l in open(path):
            r = json.loads(l)
            if r["spec"] != spec or not r.get("ok"):
                continue
            p = np.asarray(r["x_lp"]["digit_probs"][:r["n"]])
            if p.sum() > 0:
                D[(r["prompt"], r["level"])][r["mention"]][r["repeat"]] = p / p.sum()
    return D


def main():
    out = {}
    for s in SPECS:
        D = load(s)
        pw, F, E = [], [], []
        for key, d in D.items():
            ms = sorted(m for m in d if 0 in d[m])
            n = len(next(iter(d[ms[0]].values())))
            for i, m in enumerate(ms):
                E.append(tv(d[m][0], np.full(n, 1 / n)))
                if 1 in d[m]:
                    F.append(tv(d[m][0], d[m][1]))
                for m2 in ms[i + 1:]:
                    pw.append(tv(d[m][0], d[m2][0]))
        pw, F, E = map(np.mean, (pw, F, E))
        out[s] = {"pairwise_tv_rep0": float(pw), "F_f5": float(F), "E_rep0": float(E),
                  "corrected_ratio": float((pw - F) / E)}
        print(f"{s:22s} pairwise TV (rep0) {pw:.3f}  F(F5) {F:.3f}  E {E:.3f}  (TV−F)/E {(pw - F) / E:+.3f}")
    Path("results/robust_stats3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
