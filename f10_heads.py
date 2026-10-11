#!/usr/bin/env python3
"""F10b (exploratory): are the copy-suppression heads number-selective?

For the F10 categorical contexts (`tag = 'w'`, `# case w`, prompt text), the script records:
the candidate heads' attention from the final 'v=' query to the mention word's first token,
and the DLA onto each option's first token. A joint knockout of those heads at the final
query shows whether they shape word draws at all.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f10_categorical import ALL_SETS as SETS, body, grid, grid12  # noqa: E402
from natural_factorization import chat  # noqa: E402

MARK = {"assign_down": "tag = '{m}'", "comment_down": "# case {m}", "prompt_nl": "test case {m}."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--f12", action="store_true")
    a = ap.parse_args()
    if a.raw:
        global chat
        chat = lambda tk, text, family: text + "\n\nOutput:\n"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="eager", device_map="cuda").eval()
    cfg = model.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    heads = [tuple(map(int, s.split(":"))) for s in a.heads.split(",")]
    layers_c = sorted({L for L, _ in heads})
    mods = model.model.layers
    cap, st = {}, {"mask": None}

    def pre_hook(module, args, kwargs):
        if st["mask"] is not None and module.layer_idx in st["mask"]:
            kwargs["attention_mask"] = st["mask"][module.layer_idx]
        return args, kwargs

    def attn_hook(module, args, kwargs, output):
        if output[1] is not None:
            cap[("w", module.layer_idx)] = output[1][0, :, -1].float().cpu()
        return output

    def o_hook(module, args):
        cap[("o", module.layer_idx)] = args[0][0, -1].float()

    for i, layer in enumerate(mods):
        layer.self_attn.layer_idx = i
        layer.self_attn.o_proj.layer_idx = i
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        if i in layers_c:
            layer.self_attn.register_forward_hook(attn_hook, with_kwargs=True)
            layer.self_attn.o_proj.register_forward_pre_hook(o_hook)
    resid = {}
    model.model.norm.register_forward_pre_hook(lambda m, a_: resid.__setitem__("x", a_[0][0, -1].float()))
    W_U = model.lm_head.weight
    vpre = tok.encode("v=", add_special_tokens=False)
    with a.out.open("w") as f:
        for c in (grid12() if a.f12 else grid()):
            if c["location"] == "base":
                continue
            opts = SETS[c["set"]][0]
            first = [tok.encode(o, add_special_tokens=False)[0] for o in opts]  # "=" is fed as its own token
            rendered = chat(tok, body(c, "base"), "qwen")
            enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
            i0 = rendered.find(MARK[c["location"]].format(m=c["mention"])) + MARK[c["location"]].index("{m}")
            mpos = [t for t, (s0, s1) in enumerate(enc["offset_mapping"]) if s0 <= i0 < s1]
            ids = enc["input_ids"] + vpre
            T = len(ids)
            inp = torch.tensor([ids], device=model.device)
            st["mask"] = None
            with torch.inference_mode():
                out = model(input_ids=inp, use_cache=False, output_attentions=True, logits_to_keep=1)
                lp = torch.log_softmax(out.logits[0, -1].float(), -1)
                x = resid["x"]
                scale = torch.rsqrt(x.pow(2).mean() + cfg.rms_norm_eps) * model.model.norm.weight.float()
                rec = {}
                for (L, h) in heads:
                    Wo = mods[L].self_attn.o_proj.weight[:, h * D:(h + 1) * D].float()
                    contrib = (Wo @ cap[("o", L)][h * D:(h + 1) * D]) * scale
                    dla = (W_U[first].float() @ contrib).cpu().numpy()
                    rec[f"L{L}H{h}"] = {"attn": float(cap[("w", L)][h][mpos].sum()), "dla": dla.round(4).tolist()}
                q_none = torch.softmax(lp[first], -1).cpu().numpy()
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=model.device))
                masks = {}
                for (L, h) in heads:
                    m = masks.get(L, causal[None, None].expand(1, H, T, T).clone())
                    for t in mpos:
                        m[0, h, T - 1, t] = False
                    masks[L] = m
                neg = torch.finfo(torch.bfloat16).min
                st["mask"] = {L: torch.zeros(m.shape, dtype=torch.bfloat16, device=model.device).masked_fill(~m, neg) for L, m in masks.items()}
                lk = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                st["mask"] = None
                q_knock = torch.softmax(lk[first], -1).cpu().numpy()
            f.write(json.dumps({"model": a.model_name, **{k: c[k] for k in ("cid", "set", "location", "mention", "kind")},
                                "options": opts, "mention_tok": mpos, "q_none": q_none.tolist(), "q_knock": q_knock.tolist(),
                                "heads": rec}) + "\n")
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
