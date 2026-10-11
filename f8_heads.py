#!/usr/bin/env python3
"""F8 (exploratory): head-level knockout for context repulsion (Qwen3).

Each layer's query heads are scored in one batch. Batch element i stops head i at the final
query position from attending to the mention token(s).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_knockout import contexts, mention_positions  # noqa: E402
from natural_factorization import chat, digit_variants  # noqa: E402
from repulsion_conditions import body  # noqa: E402

N = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True, help="exclusive")
    ap.add_argument("--prompts", nargs="+", default=["base"])
    ap.add_argument("--head-batch", type=int, default=0, help="0 = all heads of a layer at once")
    ap.add_argument("--raw", action="store_true", help="plain-completion prompt (for base models)")
    a = ap.parse_args()
    if a.raw:
        global chat
        chat = lambda tk, text, family: text + "\n\nOutput:\n"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    variants = digit_variants(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="sdpa", device_map="cuda").eval()
    H = model.config.num_attention_heads
    hb = a.head_batch or H
    layers = model.model.layers
    state = {"mask": None, "layer": None}

    def pre_hook(module, args, kwargs):
        if state["mask"] is not None and module.layer_idx == state["layer"]:
            kwargs["attention_mask"] = state["mask"]
        return args, kwargs

    for i, layer in enumerate(layers):
        layer.self_attn.layer_idx = i
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    legal = [(d, t) for d in range(N) for t in variants[d]]
    ids_legal = [t for _, t in legal]

    def qx_from(logits):
        pr = torch.softmax(logits[:, ids_legal].float(), -1).cpu().numpy()
        q = np.zeros((pr.shape[0], N))
        for j, (d, _) in enumerate(legal):
            q[:, d] += pr[:, j]
        return q

    with a.out.open("w") as f:
        for p in a.prompts:
            for c in contexts():
                rendered = chat(tok, body(c, p), "qwen")
                ids, mpos = mention_positions(tok, rendered, c)
                ids = ids + tok.encode("x=", add_special_tokens=False)
                T = len(ids)
                inp1 = torch.tensor([ids], device=model.device)
                with torch.inference_mode():
                    state["mask"] = None
                    q_none = qx_from(model(input_ids=inp1, use_cache=False, logits_to_keep=1).logits[:, -1])[0]
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=model.device))
                res = {}
                for L in range(a.layer_start, a.layer_end):
                    rows = []
                    for h0 in range(0, H, hb):
                        hs = list(range(h0, min(H, h0 + hb)))
                        m = causal[None, None].expand(len(hs), H, T, T).clone()
                        for bi, h in enumerate(hs):
                            for k in mpos:
                                m[bi, h, T - 1, k] = False
                        state["mask"], state["layer"] = m, L
                        inp = inp1.expand(len(hs), T)
                        with torch.inference_mode():
                            rows.append(qx_from(model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[:, -1]))
                    state["mask"] = None
                    res[L] = np.concatenate(rows).round(6).tolist()  # H x N
                f.write(json.dumps({"model": a.model_name, "prompt": p, "cid": c["cid"], "location": c["location"],
                                    "mention": c["mention"], "q_none": q_none.round(6).tolist(),
                                    "q_head_knock": res}) + "\n")
                f.flush()
                print("DONE", c["cid"], flush=True)
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
