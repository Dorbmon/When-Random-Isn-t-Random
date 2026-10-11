#!/usr/bin/env python3
"""F23/F24 scorer: exact next-token distribution over digit tokens after a prefilled `<var>=`.

Input: a JSONL of items {"iid", "text", "prefix", ...}. For each item the chat-formatted prompt is
followed by `prefix` (for example "roll="), and q over digits 0-9 is read from the next-token logits,
renormalized over the legal digit tokens (all tokenizer variants of 0-9), exactly as in the main
kernel. Output rows copy every input field except the text and add q (10 digits) and digit_mass.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from natural_factorization import chat, digit_variants
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    variants = digit_variants(tok)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    legal = [(d, t) for d in range(10) for t in variants[d]]
    ids_l = [t for _, t in legal]
    items = [json.loads(l) for l in a.items.read_text().splitlines() if l.strip()]
    done = set()
    if a.out.exists():
        done = {json.loads(l)["iid"] for l in a.out.read_text().splitlines() if l.strip()}
    with a.out.open("a") as f:
        for it in items:
            if it["iid"] in done:
                continue
            ids = tok.encode(chat(tok, it["text"], a.family), add_special_tokens=False) + \
                tok.encode(it["prefix"], add_special_tokens=False)
            with torch.inference_mode():
                lo = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False,
                           logits_to_keep=1).logits[0, -1].float()
            full = torch.softmax(lo, -1)
            pr = torch.softmax(lo[ids_l], -1).cpu().numpy()
            q = np.zeros(10)
            for j, (d, _) in enumerate(legal):
                q[d] += pr[j]
            row = {k: v for k, v in it.items() if k != "text"}
            row.update({"model": a.model_name, "q": q.tolist(), "digit_mass": float(full[ids_l].sum())})
            f.write(json.dumps(row) + "\n")
            f.flush()
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
