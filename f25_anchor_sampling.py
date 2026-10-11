#!/usr/bin/env python3
"""F25 (exploratory, completes the anchoring 2x2): the classic anchor as prompt text in the
sampling grid, scored with the main experiment's exact two-draw kernel (repulsion_score.score).

F18 put two sentences in an estimation task: the incidental phrase "This is test case m." and the
classic anchor "First consider: is the answer higher or lower than m? Then answer.". The sampling
grid already has the incidental phrase (location prompt_nl). This script adds the classic anchor
at the same slot (the line before the instruction), so only the sentence differs:
  cmp_x       "First consider: is x higher or lower than m? Then answer."        (primary)
  cmp_answer  "First consider: is the answer higher or lower than m? Then answer." (F18 wording)
  prompt_nl   "This is test case m."  (re-scored: pipeline check against data/<model>.jsonl)
m is a digit 0-9 or a control letter A/B/K/Q; n in {6, 8, 10}; both expressions; both
instructions; plus the no-mention base context. 258 conditions x 2 instructions per model.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from natural_factorization import digit_variants  # noqa: E402
from repulsion_conditions import DIGITS, EXPRS, LETTERS, NS, body, draws  # noqa: E402
from repulsion_score import Y_PAD, score  # noqa: E402

SENTENCES = {
    "cmp_x": "First consider: is x higher or lower than {m}? Then answer.",
    "cmp_answer": "First consider: is the answer higher or lower than {m}? Then answer.",
    "prompt_nl": "This is test case {m}.",
}


def conditions():
    out = []
    for n in NS:
        for e in EXPRS:
            code = draws(n) + f"print({EXPRS[e]})"
            out.append({"n": n, "expr": e, "location": "base", "mention": None, "mention_type": "none",
                        "code": code, "nl": ""})
            for loc, tmpl in SENTENCES.items():
                for m in DIGITS + LETTERS:
                    out.append({"n": n, "expr": e, "location": loc, "mention": m,
                                "mention_type": "digit" if m.isdigit() else "letter",
                                "code": code, "nl": tmpl.format(m=m)})
    for c in out:
        c["cid"] = f"n{c['n']}|{c['expr']}|{c['location']}|{c['mention']}"
    assert len({c["cid"] for c in out}) == len(out) == 258
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--family", choices=("qwen", "gemma", "muse"), required=True)
    ap.add_argument("--x-separator", default="\\n")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    x_sep = bytes(a.x_separator, "utf8").decode("unicode_escape")
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    variants = digit_variants(tok)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", low_cpu_mem_usage=True, device_map="cuda").eval()
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["prompt"], r["cid"]))
    jobs = [(p, c) for p in ("base", "uniform") for c in conditions()]
    if a.limit:
        jobs = jobs[:a.limit]
    written = 0
    with a.out.open("a") as f:
        for p, c in jobs:
            if (p, c["cid"]) in done:
                continue
            r = score(model, tok, a.family, variants, body(c, p), c["n"], x_sep)
            f.write(json.dumps({"model": a.model_name, "prompt": p, **c, **r, "x_separator": x_sep, "y_pad": Y_PAD}) + "\n")
            f.flush()
            written += 1
            if written % 50 == 0:
                print("ROWS", a.model_name, written, flush=True)
    print("COMPLETE", a.model_name, written, flush=True)


if __name__ == "__main__":
    main()
