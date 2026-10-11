#!/usr/bin/env python3
"""F22 (exploratory, reviewer request): why does sharing the LAST digit repel a two-digit value
while sharing the FIRST digit barely matters?

Program: x = random.randrange(20); tag = 'm'; print(x). The target is the model's favored
two-digit value v = 1u (e.g. 17). Two answer positions are examined:
  P1  after 'x='  -> the first digit '1'
  P2  after 'x=1' -> the second digit 'u'
Per mention (v, u, 2u, 1, letters), the script records the probability of the target token at
P1 and P2, the candidate heads' attention from P1 and P2 to each mention digit token, their
DLA onto the target token, and the effect of jointly knocking the candidates out at P1 or P2.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f14_multidigit import body  # noqa: E402
from natural_factorization import chat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--target", type=int, required=True, help="favored two-digit value, e.g. 17")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="eager", device_map="cuda").eval()
    cfg = model.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    layers = model.model.layers
    cand = [tuple(map(int, s.split(":"))) for s in a.heads.split(",")]
    lays = sorted({l for l, _ in cand})
    v = a.target; u = v % 10
    tok_d = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)}
    mentions = [str(v), str(u), str(20 + u), "1", str(10 + (u + 3) % 10), "A", "B", "K", "Q"]
    cap, st, resid = {}, {"masks": None}, {}

    def pre(m, args, kw):
        if st["masks"] is not None and m._li in st["masks"]:
            kw["attention_mask"] = st["masks"][m._li]
        return args, kw

    def post(m, args, kw, out):
        if out[1] is not None:
            cap[("w", m._li)] = out[1][0, :, -1].float().cpu()
        return out

    def o_pre(m, args):
        cap[("o", m._li)] = args[0][0, -1].float()

    for i, l in enumerate(layers):
        l.self_attn._li = i; l.self_attn.o_proj._li = i
        l.self_attn.register_forward_pre_hook(pre, with_kwargs=True)
        if i in lays:
            l.self_attn.register_forward_hook(post, with_kwargs=True)
            l.self_attn.o_proj.register_forward_pre_hook(o_pre)
    model.model.norm.register_forward_pre_hook(lambda m, ar: resid.__setitem__("x", ar[0][0, -1]))
    W_U = model.lm_head.weight
    neg = torch.finfo(torch.bfloat16).min

    with a.out.open("w") as f:
        for p in ("base", "uniform"):
            for m in mentions:
                c = {"code": f"x = random.randrange(20)\ntag = '{m}'\nprint(x)", "nl": ""}
                text = chat(tok, body(c, p), "qwen")
                ids0 = tok.encode(text, add_special_tokens=False)
                # mention digit token positions (inside "tag = 'm'")
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                i0 = text.find(f"tag = '{m}'") + len("tag = '")
                mpos = [t for t, (s0, s1) in enumerate(enc["offset_mapping"]) if i0 <= s0 < i0 + len(m)]
                rec = {"model": a.model_name, "prompt": p, "mention": m, "target": v, "mention_tokens": mpos}
                for pos_name, suffix, target_tok in (("P1", "x=", tok_d[1]), ("P2", "x=1", tok_d[u])):
                    ids = ids0 + tok.encode(suffix, add_special_tokens=False)
                    T = len(ids)
                    inp = torch.tensor([ids], device=model.device)
                    cap.clear(); st["masks"] = None
                    with torch.inference_mode():
                        out = model(input_ids=inp, use_cache=False, output_attentions=True, logits_to_keep=1)
                        pr = torch.softmax(out.logits[0, -1].float(), -1)
                        p_target = float(pr[target_tok])
                        x = resid["x"].float()
                        scale = torch.rsqrt(x.pow(2).mean() + cfg.rms_norm_eps) * model.model.norm.weight.float()
                        heads = {}
                        for (l, h) in cand:
                            w = cap[("w", l)][h]
                            contrib = (layers[l].self_attn.o_proj.weight[:, h * D:(h + 1) * D].float() @ cap[("o", l)][h * D:(h + 1) * D]) * scale
                            dla_t = float(W_U[target_tok].float() @ contrib)
                            dla_mean = float((W_U[[tok_d[d] for d in range(10)]].float() @ contrib).mean())
                            heads[f"L{l}H{h}"] = {"attn_mention_tokens": [float(w[t]) for t in mpos],
                                                  "dla_target_minus_mean_digit": dla_t - dla_mean}
                        # joint knockout of the candidates at this position (block attention to mention tokens)
                        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=model.device))
                        ms = {}
                        for (l, h) in cand:
                            mm = ms.get(l, causal[None, None].expand(1, H, T, T).clone())
                            for t in mpos:
                                mm[0, h, T - 1, t] = False
                            ms[l] = mm
                        st["masks"] = {l: torch.zeros(mm.shape, dtype=torch.bfloat16, device=model.device).masked_fill(~mm, neg) for l, mm in ms.items()}
                        lk = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                        st["masks"] = None
                        p_target_ko = float(torch.softmax(lk, -1)[target_tok])
                    rec[pos_name] = {"p_target": p_target, "p_target_knockout": p_target_ko, "heads": heads}
                f.write(json.dumps(rec) + "\n")
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
