#!/usr/bin/env python3
"""F18 (exploratory, reviewer request): numeric anchoring in estimation, on the same local models.

Each question asks for an uncertain everyday quantity, answered as a single digit 0-9. Anchors:
  incidental   "This is test case m." before the question (the same phrase as the sampling grid's
               prompt_nl location); m is a digit 0-9, or a letter A/B/K/Q as control.
  comparative  "Is the answer higher or lower than c?" (the classic anchoring paradigm), c = 0..9.
  none         no anchor.
q(answer) is the exact next-token distribution over digit tokens after 'answer='.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

QUESTIONS = [
    "How many hours per day does the average adult spend on their smartphone?",
    "How many cups of coffee does the average office worker drink per day?",
    "Out of 10 randomly chosen adults, how many own a pet?",
    "Out of 10 randomly chosen people, how many are left-handed?",
    "How many colds does the average adult catch per year?",
    "How many hours does it take to drive from Paris to Lyon?",
    "How many hours of sleep does the average teenager get on a school night?",
    "How many close friends does the average adult have?",
    "Out of 10 new startups, how many are still operating after five years?",
    "How many languages does the average Swiss citizen speak?",
    "On a scale from 0 to 9, how spicy is a jalapeno pepper?",
    "How many times per month does the average person eat at a restaurant?",
]
LETTERS = ("A", "B", "K", "Q")


def prompt(q, kind, m):
    tail = ("Give your best estimate as a single digit from 0 to 9. Use exactly this format:\n"
            "answer=<digit>\nDo not explain or add any other text.")
    if kind == "incidental":
        return f"This is test case {m}.\n{q}\n{tail}"
    if kind == "comparative":
        return f"{q}\nFirst consider: is the answer higher or lower than {m}? Then answer.\n{tail}"
    return f"{q}\n{tail}"


def grid():
    out = []
    for qi, q in enumerate(QUESTIONS):
        out.append({"q": qi, "kind": "none", "mention": None, "text": prompt(q, "none", None)})
        for m in [str(d) for d in range(10)] + list(LETTERS):
            out.append({"q": qi, "kind": "incidental", "mention": m, "text": prompt(q, "incidental", m)})
        for d in range(10):
            out.append({"q": qi, "kind": "comparative", "mention": str(d), "text": prompt(q, "comparative", str(d))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
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
    with a.out.open("w") as f:
        for c in grid():
            ids = tok.encode(chat(tok, c["text"], a.family), add_special_tokens=False) + tok.encode("answer=", add_special_tokens=False)
            with torch.inference_mode():
                lo = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1).logits[0, -1].float()
            full = torch.softmax(lo, -1)
            pr = torch.softmax(lo[ids_l], -1).cpu().numpy()
            q = np.zeros(10)
            for j, (d, _) in enumerate(legal):
                q[d] += pr[j]
            f.write(json.dumps({"model": a.model_name, **{k: c[k] for k in ("q", "kind", "mention")},
                                "q_answer": q.tolist(), "digit_mass": float(full[ids_l].sum())}) + "\n")
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
