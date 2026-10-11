#!/usr/bin/env python3
"""F3 (exploratory): attention knockout of the mention token(s) in Qwen3 models.

For each context we score q(x) at the final 'x=' position under several masks. Each mask
stops some query positions from attending to the mention token(s) in some layers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from natural_factorization import chat, digit_variants  # noqa: E402
from repulsion_conditions import EXPRS, body, draws, located  # noqa: E402

N = 8
MARK = {"assign_down": "tag = '{m}'", "comment_down": "# case {m}", "prompt_nl": "test case {m}.",
        "varname_down": "z{m}", "print_const": "print({m})"}


def contexts():
    out = []
    for loc in ("assign_down", "comment_down", "prompt_nl", "varname_down"):
        for m in [str(d) for d in range(10)] + ["A", "B", "K", "Q"]:
            code, nl = located(loc, N, EXPRS["e1"], m)
            out.append({"cid": f"n{N}|e1|{loc}|{m}", "n": N, "expr": "e1", "location": loc,
                        "mention": m, "code": code, "nl": nl})
    for d in range(10):
        out.append({"cid": f"n{N}|None|print_const|{d}", "n": N, "expr": None,
                    "location": "print_const", "mention": str(d),
                    "code": draws(N) + f"print({d})", "nl": ""})
    return out


def mention_positions(tok, rendered, c):
    """Token indices that cover the mention character(s) inside the marker text."""
    marker = MARK[c["location"]].format(m=c["mention"])
    offset = MARK[c["location"]].index("{m}")
    pos, start = [], 0
    while (i := rendered.find(marker, start)) >= 0:
        pos.extend(i + offset + k for k in range(len(c["mention"])))
        start = i + 1
    if c["location"] == "varname_down":  # z{m} appears twice: assignment and print
        assert len(pos) == 2, (c["cid"], pos)
    else:
        assert len(pos) == 1, (c["cid"], pos)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    idx = sorted({t for t, (a, b) in enumerate(enc["offset_mapping"]) for ch in pos if a <= ch < b})
    assert idx, c["cid"]
    for t in idx:
        assert c["mention"] in tok.decode([enc["input_ids"][t]]), (c["cid"], tok.decode([enc["input_ids"][t]]))
    return enc["input_ids"], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
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
    layers = model.model.layers
    L = len(layers)
    state = {"mask": None, "active": set()}

    def pre_hook(module, args, kwargs):
        if state["mask"] is not None and module.layer_idx in state["active"]:
            kwargs["attention_mask"] = state["mask"]
        return args, kwargs

    for i, layer in enumerate(layers):
        layer.self_attn.layer_idx = i
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    legal = [(d, t) for d in range(N) for t in variants[d]]
    ids_legal = [t for _, t in legal]
    windows = [list(range(s, min(L, s + a.window))) for s in range(0, L, a.window)]
    knockouts = {"none": (None, []), "causal_only": ("none", range(L)), "all": ("all", range(L)), "last_only": ("last", range(L)),
                 "others_only": ("others", range(L)),
                 "early_half_last": ("last", range(L // 2)), "late_half_last": ("last", range(L // 2, L))}
    for w in windows:
        knockouts[f"win{w[0]:02d}-{w[-1]:02d}_last"] = ("last", w)
    with a.out.open("w") as f:
        for p in ("base", "uniform"):
            for c in (contexts()[:a.limit] if a.limit else contexts()):
                rendered = chat(tok, body(c, p), "qwen")
                ids, mpos = mention_positions(tok, rendered, c)
                assert ids == tok.encode(rendered, add_special_tokens=False)
                ids = ids + tok.encode("x=", add_special_tokens=False)
                T = len(ids)
                inp = torch.tensor([ids], device=model.device)
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=model.device))
                res = {}
                for name, (who, lays) in knockouts.items():
                    if who is None:
                        state["mask"] = None
                    else:
                        m = causal.clone()
                        q = {"all": slice(None), "last": slice(T - 1, T), "others": slice(0, T - 1),
                             "none": slice(0, 0)}[who]
                        for k in mpos:
                            col = m[:, k].clone()
                            col_block = torch.zeros(T, dtype=torch.bool, device=model.device)
                            col_block[q] = True
                            col_block[:k + 1] = False  # the token itself and earlier queries are unchanged
                            m[:, k] = col & ~col_block
                        state["mask"] = m[None, None]
                    state["active"] = set(lays)
                    with torch.inference_mode():
                        logits = model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    pr = torch.softmax(logits[ids_legal], -1).cpu().numpy()
                    qx = np.zeros(N)
                    for (d, _), v in zip(legal, pr):
                        qx[d] += v
                    res[name] = qx.tolist()
                state["mask"] = None
                f.write(json.dumps({"model": a.model_name, "prompt": p, **{k: c[k] for k in ("cid", "n", "location", "mention")},
                                    "mention_token_idx": mpos, "seq_len": T, "n_layers": L, "q_x": res}) + "\n")
                f.flush()
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
