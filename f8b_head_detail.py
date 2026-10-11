#!/usr/bin/env python3
"""F8b (exploratory): copy-suppression signature of the candidate heads.

For each context and each candidate head (plus random control heads), at the final query:
  attn   the attention probability on the mention token(s) (eager pass for the chosen layers)
  dla    the direct logit attribution of the head's output onto every digit 0..n-1. The head's
         o_proj slice goes through the final RMSNorm scale (frozen at its value on that
         forward) and then lm_head.
The script also computes joint knockouts of the top-k candidate heads (k = 1..K).
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
    ap.add_argument("--heads", required=True, help="L:H,L:H,... ranked candidates")
    ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
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
                                                 attn_implementation="eager", device_map="cuda").eval()
    cfg = model.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    cand = [tuple(map(int, s.split(":"))) for s in a.heads.split(",")]
    rng = np.random.default_rng(0)
    layers_c = sorted({L for L, _ in cand})
    rand = []
    while len(rand) < a.n_random:
        L = int(rng.choice(layers_c)); h = int(rng.integers(H))
        if (L, h) not in cand and (L, h) not in rand:
            rand.append((L, h))
    heads = cand + rand
    mods = model.model.layers
    cap = {}
    state = {"mask": None, "layers": set()}

    def pre_hook(module, args, kwargs):
        if state["mask"] is not None and module.layer_idx in state["layers"]:
            kwargs["attention_mask"] = state["mask"][module.layer_idx]
        return args, kwargs

    def attn_out_hook(module, args, kwargs, output):
        # output = (attn_output, attn_weights); attn_weights: (b, H, q, k)
        if output[1] is not None:
            cap[("w", module.layer_idx)] = output[1][0, :, -1].float().cpu()
        return output

    def o_in_hook(module, args):
        cap[("o_in", module.layer_idx)] = args[0][0, -1].float()  # (H*D,)

    for i, layer in enumerate(mods):
        layer.self_attn.layer_idx = i
        layer.self_attn.o_proj.layer_idx = i
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        if i in layers_c:
            layer.self_attn.register_forward_hook(attn_out_hook, with_kwargs=True)
            layer.self_attn.o_proj.register_forward_pre_hook(o_in_hook)
    final_norm = model.model.norm
    W_U = model.lm_head.weight  # (V, hidden)
    legal = [(d, t) for d in range(N) for t in variants[d]]
    ids_legal = [t for _, t in legal]
    canon = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(N)}

    def qx(logits):
        pr = torch.softmax(logits[ids_legal].float(), -1).cpu().numpy()
        q = np.zeros(N)
        for j, (d, _) in enumerate(legal):
            q[d] += pr[j]
        return q

    resid = {}

    def norm_in_hook(module, args):
        resid["x"] = args[0][0, -1].float()
    final_norm.register_forward_pre_hook(norm_in_hook)

    with a.out.open("w") as f:
        for p in ("base",):
            for c in contexts():
                rendered = chat(tok, body(c, p), "qwen")
                ids, mpos = mention_positions(tok, rendered, c)
                ids = ids + tok.encode("x=", add_special_tokens=False)
                T = len(ids)
                inp = torch.tensor([ids], device=model.device)
                state["mask"] = None
                cap.clear()
                with torch.inference_mode():
                    out = model(input_ids=inp, use_cache=False, output_attentions=True, logits_to_keep=1)
                q_none = qx(out.logits[0, -1])
                # RMSNorm scale at the final position, frozen
                x = resid["x"]
                rec = {"heads": {}}
                with torch.inference_mode():
                  scale = torch.rsqrt(x.pow(2).mean() + cfg.rms_norm_eps) * final_norm.weight.float()
                  for (L, h) in heads:
                      w = cap[("w", L)][h]  # (T,)
                      o_in = cap[("o_in", L)][h * D:(h + 1) * D]
                      Wo = mods[L].self_attn.o_proj.weight[:, h * D:(h + 1) * D].float()
                      contrib = (Wo @ o_in) * scale  # (hidden,)
                      dl = (W_U[[canon[d] for d in range(N)]].float() @ contrib).cpu().numpy()
                      rec["heads"][f"L{L}H{h}"] = {"attn_mention": float(w[mpos].sum()), "dla": dl.round(4).tolist(),
                                                   "candidate": (L, h) in cand}
                # joint knockouts of the top-k candidates at the final query
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=model.device))
                joint = {}
                for k in range(1, len(cand) + 1):
                    masks = {}
                    for (L, h) in cand[:k]:
                        m = masks.get(L, causal[None, None].expand(1, H, T, T).clone())
                        for t in mpos:
                            m[0, h, T - 1, t] = False
                        masks[L] = m
                    neg = torch.finfo(torch.bfloat16).min
                    fmasks = {L: torch.zeros(m.shape, dtype=torch.bfloat16, device=model.device).masked_fill(~m, neg)
                              for L, m in masks.items()}
                    state["mask"], state["layers"] = fmasks, set(fmasks)
                    with torch.inference_mode():
                        lo = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1]
                    joint[f"top{k}"] = qx(lo).round(6).tolist()
                state["mask"] = None
                f.write(json.dumps({"model": a.model_name, "prompt": p, "cid": c["cid"], "location": c["location"],
                                    "mention": c["mention"], "q_none": q_none.round(6).tolist(),
                                    "joint": joint, **rec}) + "\n")
                f.flush()
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
