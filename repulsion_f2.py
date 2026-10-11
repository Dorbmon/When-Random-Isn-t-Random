#!/usr/bin/env python3
"""F2 minimal pairs: output role, live vs dead code, commented print, output order.

Also scores local models: python repulsion_f2.py --model ... --out data/f2_<model>.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repulsion_conditions import UNIFORM, cot_prompt, draws  # noqa: E402

NS = (6, 8, 10)
LETTERS = ("A", "B", "K", "Q")


def two_output_prompt(code):
    return ("Simulate exactly one execution of the Python 3 program. Pick a value for each random draw, "
            "then execute the remaining code. Use exactly three lines in this format:\n"
            "x=<digit>\ny=<digit>\nanswer=<first printed digit>,<second printed digit>\n"
            "Do not explain or add any other text.\n\nimport random\n" + code)


def code_for(cond, n, m):
    d = draws(n)
    lit = m if m.isdigit() else f"'{m}'"
    return {
        "var_print": d + f"c = {lit}\nprint(c)",
        "var_noprint": d + f"c = {lit}\nprint(max(x, y))",
        "if_true": d + f"if True:\n    print({lit})\nelse:\n    print(max(x, y))",
        "if_false": d + f"if False:\n    print({lit})\nelse:\n    print(max(x, y))",
        "commented_print": d + f"print(max(x, y))  # print({lit})",
        "print_before_x": d + f"print({lit})\nprint(x)",
        "print_after_x": d + f"print(x)\nprint({lit})",
    }[cond]


# Letter controls only where the program's printed output stays a digit.
LETTER_OK = {"var_noprint", "if_false", "commented_print"}
CONDS = ("var_print", "var_noprint", "if_true", "if_false", "commented_print",
         "print_before_x", "print_after_x")


def grid():
    out = []
    for n in NS:
        out.append({"cid": f"n{n}|base", "n": n, "cond": "base", "mention": None,
                    "mention_type": "none", "code": draws(n) + "print(max(x, y))", "two_out": False})
        for cond in CONDS:
            ms = [str(k) for k in range(10)] + (list(LETTERS) if cond in LETTER_OK else [])
            for m in ms:
                out.append({"cid": f"n{n}|{cond}|{m}", "n": n, "cond": cond, "mention": m,
                            "mention_type": "digit" if m.isdigit() else "letter",
                            "code": code_for(cond, n, m), "two_out": cond.startswith("print_")})
    assert len({c["cid"] for c in out}) == len(out) and len({c["code"] for c in out}) == len(out)
    return out


def body(c, prompt):
    text = two_output_prompt(c["code"]) if c["two_out"] else cot_prompt(c["code"])
    return (UNIFORM + "\n" + text) if prompt == "uniform" else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--x-separator", default="\\n")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", action="store_true", help="plain-completion prompt (base models)")
    a = ap.parse_args()
    from natural_factorization import digit_variants
    import repulsion_score
    from repulsion_score import score
    if a.raw:
        repulsion_score.chat = lambda tk, text, family: text + "\n\nOutput:\n"
    xs = bytes(a.x_separator, "utf8").decode("unicode_escape")
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    variants = digit_variants(tok)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    done = set()
    if a.out.exists():
        done = {(json.loads(l)["prompt"], json.loads(l)["cid"]) for l in a.out.read_text().splitlines()}
    with a.out.open("a") as f:
        for p in ("base", "uniform"):
            for c in grid():
                if (p, c["cid"]) in done:
                    continue
                r = score(model, tok, a.family, variants, body(c, p), c["n"], xs)
                f.write(json.dumps({"model": a.model_name, "prompt": p, **c, **r}) + "\n")
                f.flush()
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
