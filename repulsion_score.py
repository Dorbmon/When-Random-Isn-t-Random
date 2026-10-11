#!/usr/bin/env python3
"""Exact two-draw kernel for the context-repulsion grid."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from natural_factorization import chat, digit_variants  # noqa: E402
from repulsion_conditions import body, conditions  # noqa: E402

Y_PAD = 32


def forward(model, seqs, pad_to):
    import torch
    assert len({len(s) for s in seqs}) == 1 and len(seqs) <= max(pad_to, 1)
    k = len(seqs)
    if pad_to:
        seqs = seqs + [seqs[-1]] * (pad_to - k)
    toks = torch.tensor(seqs, dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=toks, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    return logits[:k]


def cond_probs(logits, ids):
    import torch
    chosen = logits[:, ids]
    p = torch.softmax(chosen, dim=-1).cpu().numpy().astype(np.float64)
    mass = torch.exp(torch.logsumexp(chosen, -1) - torch.logsumexp(logits, -1)).cpu().numpy()
    return p / p.sum(1, keepdims=True), np.clip(mass.astype(np.float64), 0, 1)


def score(model, tok, family, variants, text, n, x_sep):
    prefix = tok.encode(chat(tok, text, family), add_special_tokens=False) + \
        tok.encode("x=", add_special_tokens=False)
    sep_y = tok.encode(x_sep + "y=", add_special_tokens=False)
    all10 = [(d, t) for d in range(10) for t in variants[d]]
    legal = [(d, t) for d, t in all10 if d < n]
    lx = forward(model, [prefix], 0)
    p10, m10 = cond_probs(lx, [t for _, t in all10])
    px, mx = cond_probs(lx, [t for _, t in legal])
    ypre = [prefix + [t] + sep_y for _, t in legal]
    assert len(ypre) <= Y_PAD, len(ypre)
    ly = forward(model, ypre, Y_PAD)
    py, my = cond_probs(ly, [t for _, t in legal])
    qx10 = np.zeros(10)
    for (d, _), p in zip(all10, p10[0]):
        qx10[d] += p
    q = np.zeros((n, n))
    for i, (x, _) in enumerate(legal):
        for j, (y, _) in enumerate(legal):
            q[x, y] += px[0, i] * py[i, j]
    assert abs(q.sum() - 1) < 1e-6
    return {"q_xy": q.tolist(), "q_x": q.sum(1).tolist(), "q_x_all10": qx10.tolist(),
            "legal_mass_x": float(mx[0]), "digit_mass_x_all10": float(m10[0]),
            "legal_mass_y_given_legal_x": float(np.dot(px[0], my)),
            "prompt_tokens": len(prefix)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--tokenizer", help="tokenizer dir (for the chat template); defaults to --model")
    ap.add_argument("--family", choices=("qwen", "gemma", "muse"), required=True)
    ap.add_argument("--x-separator", default="\\n")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    x_sep = bytes(args.x_separator, "utf8").decode("unicode_escape")
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, local_files_only=True)
    variants = digit_variants(tok)
    cls = AutoModelForCausalLM if args.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", low_cpu_mem_usage=True,
                                device_map="cuda")
    model.eval()
    grid = conditions()
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["prompt"], r["cid"]))
    jobs = [(p, c) for p in ("base", "uniform") for c in grid]
    if args.limit:
        jobs = jobs[:args.limit]
    written = 0
    with args.out.open("a") as f:
        for p, c in jobs:
            if (p, c["cid"]) in done:
                continue
            r = score(model, tok, args.family, variants, body(c, p), c["n"], x_sep)
            row = {"model": args.model_name, "prompt": p, **c, **r,
                   "x_separator": x_sep, "y_pad": Y_PAD}
            f.write(json.dumps(row) + "\n")
            f.flush()
            written += 1
            if written % 50 == 0:
                print("ROWS", args.model_name, written, "/", len(jobs) - len(done), flush=True)
    print("COMPLETE", args.model_name, written, flush=True)


if __name__ == "__main__":
    main()
