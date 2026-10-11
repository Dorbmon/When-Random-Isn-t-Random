#!/usr/bin/env python3
"""F21 (exploratory, reviewer request): do the same heads cause sequential repulsion (y avoids x)?

For the no-mention base programs (n in {6, 8, 10}, both expressions, both prompts) and every
value of x, score q(y) at the 'y=' position after the model's own 'x=<x>'. At that final query,
block attention from the listed heads to the x digit token and measure P(y = x).
Controls: random head sets of the same size and layers, and blocking every head in every layer
from that token (the full direct route).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from natural_factorization import chat, digit_variants  # noqa: E402
from repulsion_conditions import body, conditions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--x-separator", default="\\n")
    ap.add_argument("--n-random-sets", type=int, default=3)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sep = bytes(a.x_separator, "utf8").decode("unicode_escape")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    variants = digit_variants(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="sdpa", device_map="cuda").eval()
    render = (lambda t: t + "\n\nOutput:\n") if a.raw else (lambda t: chat(tok, t, "qwen"))
    layers = model.model.layers
    L, H = len(layers), model.config.num_attention_heads
    cand = [tuple(map(int, s.split(":"))) for s in a.heads.split(",")]
    rng = np.random.default_rng(3)
    lays = sorted({l for l, _ in cand})
    rsets = []
    for _ in range(a.n_random_sets):
        s = []
        while len(s) < len(cand):
            l = int(rng.choice(lays)); h = int(rng.integers(H))
            if (l, h) not in cand and (l, h) not in s:
                s.append((l, h))
        rsets.append(s)
    st = {"masks": None}

    def pre(m, args, kw):
        if st["masks"] is not None and m._li in st["masks"]:
            kw["attention_mask"] = st["masks"][m._li]
        return args, kw

    for i, l in enumerate(layers):
        l.self_attn._li = i
        l.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    base_ctx = [c for c in conditions() if c["location"] == "base"]
    x_tok = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)}
    with a.out.open("w") as f:
        for p in ("base", "uniform"):
            for c in base_ctx:
                n = c["n"]
                legal = [(d, t) for d in range(n) for t in variants[d]]
                ids_l = [t for _, t in legal]
                pre_ids = tok.encode(render(body(c, p)), add_special_tokens=False) + tok.encode("x=", add_special_tokens=False)
                for x in range(n):
                    ids = pre_ids + [x_tok[x]] + tok.encode(sep + "y=", add_special_tokens=False)
                    T = len(ids); xpos = len(pre_ids)
                    inp = torch.tensor([ids], device=model.device)
                    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=model.device))

                    def run(heads=None, full=False):
                        if heads is None and not full:
                            st["masks"] = None
                        elif full:
                            m = causal.clone(); m[T - 1, xpos] = False
                            st["masks"] = {l: m[None, None] for l in range(L)}
                        else:
                            ms = {}
                            for (l, h) in heads:
                                m = ms.get(l, causal[None, None].expand(1, H, T, T).clone())
                                m[0, h, T - 1, xpos] = False
                                ms[l] = m
                            st["masks"] = ms
                        with torch.inference_mode():
                            lo = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                        st["masks"] = None
                        pr = torch.softmax(lo[ids_l], -1).cpu().numpy()
                        q = np.zeros(n)
                        for j, (d, _) in enumerate(legal):
                            q[d] += pr[j]
                        return q.tolist()
                    row = {"model": a.model_name, "prompt": p, "cid": c["cid"], "n": n, "x": x,
                           "q_none": run(), "q_cand": run(cand), "q_full": run(full=True),
                           "q_random": [run(s) for s in rsets]}
                    f.write(json.dumps(row) + "\n")
                f.flush()
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
