#!/usr/bin/env python3
"""F20 (exploratory, reviewer request): prompt-free OV check of the copy-suppression heads.

For each digit token d, the input to a head at layer L is the "effective embedding": the
residual stream just before layer L when the model reads d as a one-token sequence (no
prompt). It passes through layer L's input norm, the head's value path and its o_proj slice,
then through the final norm and the unembedding, and the logits for digits 0..9 are read off.
The result is a 10 x 10 matrix M[d_in, d_out]. Copy suppression predicts a negative diagonal:
each digit's own logit is pushed down the most. Random heads from the same layers are the null.
The raw token embedding (with no layers applied) is reported as a second input variant.
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
    ap.add_argument("--n-random", type=int, default=30)
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
    rng = np.random.default_rng(0)
    lays = sorted({l for l, _ in cand})
    rand = []
    while len(rand) < a.n_random:
        l = int(rng.choice(lays)); h = int(rng.integers(H))
        if (l, h) not in cand and (l, h) not in rand:
            rand.append((l, h))
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    dev = next(model.parameters()).device
    # effective embeddings: hidden state entering each needed layer for a 1-token input
    hs_by_layer = {}
    with torch.inference_mode():
        for d, t in enumerate(digits):
            out = model(input_ids=torch.tensor([[t]], device=dev), output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # tuple, index i = input to layer i
            for l in lays:
                hs_by_layer.setdefault(l, []).append(hs[l][0, -1])
        emb = model.get_input_embeddings()
        raw = emb(torch.tensor(digits, device=dev))
    W_U = lm_head.weight

    def head_matrix(l, h, X):
        att = layers[l].self_attn
        with torch.inference_mode():
            x = layers[l].input_layernorm(X)
            if getattr(att, "v_proj", None) is not None:
                v = att.v_proj(x)
            else:  # key-equals-value layers (Gemma global layers)
                v = att.k_proj(x)
            n_kv = v.shape[-1] // att.head_dim
            v = v.view(X.shape[0], n_kv, att.head_dim)
            if hasattr(att, "v_norm"):
                v = att.v_norm(v)
            kv = h // (H // n_kv)
            vh = v[:, kv, :].float()
            D = att.o_proj.weight.shape[1] // H
            assert D == att.head_dim
            contrib = vh @ att.o_proj.weight[:, h * D:(h + 1) * D].float().T  # (10, hidden)
            normed = final_norm(contrib.to(X.dtype)).float()
            logits = normed @ W_U[digits].float().T  # (10 in, 10 out)
        M = logits.cpu().numpy()
        M = M - M.mean(axis=1, keepdims=True)
        return M

    def stats(M):
        diag = np.diag(M)
        z = (diag - M.mean(1)) / (M.std(1) + 1e-9)
        ranks = [int((M[i] < M[i, i]).sum()) + 1 for i in range(10)]  # 1 = most negative
        return {"diag_z_mean": float(z.mean()), "frac_diag_min": float(np.mean([r == 1 for r in ranks])),
                "mean_rank": float(np.mean(ranks)), "diag_minus_offdiag": float(diag.mean() - (M.sum() - diag.sum()) / 90)}

    res = {"model": a.model_name, "candidates": {}, "random": {}}
    for kind, hl in (("candidates", cand), ("random", rand)):
        for (l, h) in hl:
            eff = torch.stack(hs_by_layer[l])
            Me = head_matrix(l, h, eff)
            Mr = head_matrix(l, h, raw)
            res[kind][f"L{l}H{h}"] = {"effective": stats(Me), "raw": stats(Mr),
                                      "matrix_effective": np.round(Me, 3).tolist()}
    a.out.write_text(json.dumps(res, indent=1))
    for kind in ("candidates", "random"):
        z = [v["effective"]["diag_z_mean"] for v in res[kind].values()]
        print(kind, "effective diag z: mean", round(float(np.mean(z)), 2), "min", round(float(np.min(z)), 2), "max", round(float(np.max(z)), 2), flush=True)
    for k, v in res["candidates"].items():
        print("  ", k, "eff", {x: round(y, 2) for x, y in v["effective"].items()}, "raw z", round(v["raw"]["diag_z_mean"], 2), flush=True)
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
