#!/usr/bin/env python3
"""Follow-up F1 (exploratory): digit token vs number word."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from natural_factorization import digit_variants  # noqa: E402
from repulsion_conditions import EXPRS, body, located  # noqa: E402
from repulsion_score import score  # noqa: E402

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
FORMS = ([(str(d), "digit", d) for d in range(10)] + [(w, "word", d) for d, w in enumerate(WORDS)]
         + [(l, "letter", None) for l in "ABKQ"]
         + [(w, "nato", None) for w in ("alpha", "bravo", "kilo", "quebec")])


def grid(n=8, e="e1"):
    out = []
    for loc in ("comment_down", "assign_down", "prompt_nl"):
        for m, form, value in FORMS:
            code, nl = located(loc, n, EXPRS[e], m)
            out.append({"cid": f"n{n}|{e}|{loc}|{form}|{m}", "n": n, "expr": e, "location": loc,
                        "mention": m, "form": form, "value": value, "code": code, "nl": nl})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--x-separator", default="\\n")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    xs = bytes(a.x_separator, "utf8").decode("unicode_escape")
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    variants = digit_variants(tok)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    with a.out.open("w") as f:
        for p in ("base", "uniform"):
            for c in grid():
                r = score(model, tok, a.family, variants, body(c, p), c["n"], xs)
                f.write(json.dumps({"model": a.model_name, "prompt": p, **c, **r}) + "\n")
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
