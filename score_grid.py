#!/usr/bin/env python3
"""Exact q(x), q(y|x) scoring for any grid in f45_grids (chat template, or raw completion)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from natural_factorization import digit_variants  # noqa: E402
from f45_grids import text_for  # noqa: E402
import repulsion_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--grid", required=True, help="module:function, e.g. f45_grids:f4")
    ap.add_argument("--raw", action="store_true", help="no chat template: plain completion prompt")
    ap.add_argument("--x-separator", default="\\n")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mod, fn = a.grid.split(":")
    grid = getattr(__import__(mod), fn)()
    xs = bytes(a.x_separator, "utf8").decode("unicode_escape")
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    variants = digit_variants(tok)
    if a.raw:  # plain completion: instruction + program, then the answer block
        repulsion_score.chat = lambda tk, body, family: body + "\n\nOutput:\n"
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    done = set()
    if a.out.exists():
        done = {(json.loads(l)["prompt"], json.loads(l)["cid"]) for l in a.out.read_text().splitlines()}
    with a.out.open("a") as f:
        for p in ("base", "uniform"):
            for c in grid:
                if (p, c["cid"]) in done:
                    continue
                r = repulsion_score.score(model, tok, a.family, variants, text_for(c, p), c["n"], xs)
                f.write(json.dumps({"model": a.model_name, "prompt": p, "raw_prompt": a.raw, **c, **r}) + "\n")
                f.flush()
    print("COMPLETE", a.model_name, a.grid, flush=True)


if __name__ == "__main__":
    main()
