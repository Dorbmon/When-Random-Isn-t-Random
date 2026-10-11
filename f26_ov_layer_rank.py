#!/usr/bin/env python3
"""F26 (exploratory, stricter baseline for F20): rank of each candidate head's OV signature
among ALL heads of its layer.

Same computation as f20_ov.py (head_matrix and stats are copied unchanged): each digit's raw token
embedding or effective embedding (hidden state entering the layer for a one-token input) is
passed through the head's value path and o_proj slice, the final norm and the unembedding,
giving a 10 x 10 digit matrix. For every head of every candidate layer we record diagonal z and
the own-lowest share, with both inputs. A candidate's rank is 1 + the number of other heads in its
layer with a more negative diagonal z (1 = most copy-suppressive).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch import text_parts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    layers, final_norm, lm_head, cfg = text_parts(model)
    H = cfg.num_attention_heads
    cand = [tuple(map(int, s.split(":"))) for s in a.heads.split(",")]
    lays = sorted({l for l, _ in cand})
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    dev = next(model.parameters()).device
    hs_by_layer = {}
    with torch.inference_mode():
        for d, t in enumerate(digits):
            out = model(input_ids=torch.tensor([[t]], device=dev), output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            for l in lays:
                hs_by_layer.setdefault(l, []).append(hs[l][0, -1])
        emb = model.get_input_embeddings()
        raw = emb(torch.tensor(digits, device=dev))
    W_U = lm_head.weight

    def head_matrix(l, h, X):  # unchanged from f20_ov.py
        att = layers[l].self_attn
        with torch.inference_mode():
            x = layers[l].input_layernorm(X)
            if getattr(att, "v_proj", None) is not None:
                v = att.v_proj(x)
            else:
                v = att.k_proj(x)
            n_kv = v.shape[-1] // att.head_dim
            v = v.view(X.shape[0], n_kv, att.head_dim)
            if hasattr(att, "v_norm"):
                v = att.v_norm(v)
            kv = h // (H // n_kv)
            vh = v[:, kv, :].float()
            D = att.o_proj.weight.shape[1] // H
            assert D == att.head_dim
            contrib = vh @ att.o_proj.weight[:, h * D:(h + 1) * D].float().T
            normed = final_norm(contrib.to(X.dtype)).float()
            logits = normed @ W_U[digits].float().T
        M = logits.cpu().numpy()
        M = M - M.mean(axis=1, keepdims=True)
        return M

    def stats(M):  # unchanged from f20_ov.py
        diag = np.diag(M)
        z = (diag - M.mean(1)) / (M.std(1) + 1e-9)
        ranks = [int((M[i] < M[i, i]).sum()) + 1 for i in range(10)]
        return {"diag_z_mean": float(z.mean()), "frac_diag_min": float(np.mean([r == 1 for r in ranks])),
                "mean_rank": float(np.mean(ranks)), "diag_minus_offdiag": float(diag.mean() - (M.sum() - diag.sum()) / 90)}

    heads = {}
    for l in lays:
        eff = torch.stack(hs_by_layer[l])
        for h in range(H):
            heads[f"L{l}H{h}"] = {"layer": l, "head": h,
                                  "effective": stats(head_matrix(l, h, eff)), "raw": stats(head_matrix(l, h, raw))}
    ranks = {}
    for (l, h) in cand:
        me = heads[f"L{l}H{h}"]
        same = [v for v in heads.values() if v["layer"] == l]
        r = {}
        for emb_kind in ("raw", "effective"):
            z = me[emb_kind]["diag_z_mean"]; f = me[emb_kind]["frac_diag_min"]
            r[emb_kind] = {"z": z, "own_lowest": f,
                           "z_rank": 1 + sum(v[emb_kind]["diag_z_mean"] < z for v in same if v is not me),
                           "n_heads": len(same),
                           "share_heads_own_lowest_ge": float(np.mean([v[emb_kind]["frac_diag_min"] >= f for v in same if v is not me])),
                           "layer_z_median": float(np.median([v[emb_kind]["diag_z_mean"] for v in same])),
                           "layer_own_lowest_mean": float(np.mean([v[emb_kind]["frac_diag_min"] for v in same]))}
        ranks[f"L{l}H{h}"] = r
        print(f"  L{l}H{h}", {k: (round(v['z'], 2), f"rank {v['z_rank']}/{v['n_heads']}", v['own_lowest']) for k, v in r.items()}, flush=True)
    a.out.write_text(json.dumps({"model": a.model_name, "H": H, "layers": lays, "candidates": ranks, "all_heads": heads}, indent=1))
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
