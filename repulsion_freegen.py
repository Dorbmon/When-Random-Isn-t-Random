#!/usr/bin/env python3
"""Free-generation check of the exact kernel on a subset of the repulsion grid."""
import argparse
import json
import re
import sys
from pathlib import Path

from natural_factorization import chat  # noqa: E402
from repulsion_conditions import body, conditions  # noqa: E402

X = re.compile(r"(?<![A-Za-z_])x\s*=\s*(\d+)")
Y = re.compile(r"(?<![A-Za-z_])y\s*=\s*(\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", default="qwen")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    grid = conditions(ns=(8,), exprs=("e1",), locations=("comment_down", "prompt_nl"))
    pad = tok.pad_token_id or tok.eos_token_id
    with a.out.open("w") as f:
        for j, c in enumerate(grid):
            for p in ("base", "uniform"):
                ids = tok.encode(chat(tok, body(c, p), a.family), add_special_tokens=False)
                for start in range(0, a.samples, a.batch):
                    torch.manual_seed(20260925 + 1000 * j + (0 if p == "base" else 500) + start)
                    inp = torch.tensor([ids] * a.batch, device=model.device)
                    with torch.inference_mode():
                        seq = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                                             do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                             max_new_tokens=24, pad_token_id=pad)
                    for t in seq[:, len(ids):]:
                        raw = tok.decode(t, skip_special_tokens=True)
                        starts_x = raw.lstrip().startswith("x=") or raw.lstrip().startswith("x =")
                        mx, my = X.search(raw), Y.search(raw)
                        f.write(json.dumps({"model": a.model_name, "prompt": p, "cid": c["cid"],
                                            "raw": raw, "starts_with_x": starts_x, "x": int(mx.group(1)) if mx else None,
                                            "y": int(my.group(1)) if my else None}) + "\n")
            print("DONE", j + 1, len(grid), flush=True)


if __name__ == "__main__":
    main()
