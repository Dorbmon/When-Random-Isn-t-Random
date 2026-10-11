#!/usr/bin/env python3
"""F10 (exploratory): does context repulsion extend to non-numeric categorical draws?

Program: v = random.choice([4 options]); print(v). The mention is one option word, or a
non-option control word, at assign_down / comment_down / prompt_nl. q(v) is the renormalized
sequence probability of each option string after the reply prefix 'v='.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SETS = {
    "colors": (["red", "green", "blue", "yellow"], ["purple", "pink", "black", "white"]),
    "names": (["Alice", "Bob", "Carol", "Dave"], ["Eve", "Frank", "Grace", "Heidi"]),
    "fruits": (["apple", "banana", "cherry", "grape"], ["lemon", "mango", "peach", "plum"]),
}
# F11: integer and single-letter options (logged separately; F10 grid unchanged)
SETS11 = {
    "ints": (["2", "4", "6", "8"], ["1", "3", "5", "7"]),
    "ints_b": (["1", "3", "5", "7"], ["2", "4", "6", "8"]),
    "letters": (["A", "B", "C", "D"], ["E", "F", "G", "H"]),
    "letters_b": (["P", "R", "S", "T"], ["K", "L", "M", "N"]),
}
UNIFORM = "Each random draw must be independent and uniformly sampled from its possible values."


def prompt_text(code, nl):
    t = ("Simulate exactly one execution of the Python 3 program. Pick a value for each random draw, "
         "then execute the remaining code. Use exactly two lines in this format:\n"
         "v=<value, without quotes>\nanswer=<printed output>\n"
         "Do not explain or add any other text.\n\nimport random\n" + code)
    return (nl + "\n" + t) if nl else t


def grid(sets=None):
    out = []
    for s, (opts, ctrls) in (sets or SETS).items():
        lst = "[" + ", ".join(o if o.isdigit() else f"'{o}'" for o in opts) + "]"
        draw = f"v = random.choice({lst})\n"
        out.append({"cid": f"{s}|base|None", "set": s, "location": "base", "mention": None, "kind": "none",
                    "code": draw + "print(v)", "nl": ""})
        for m in opts + ctrls:
            kind = "option" if m in opts else "control"
            for loc, code, nl in (("assign_down", draw + f"tag = '{m}'\nprint(v)", ""),
                                  ("comment_down", draw + f"print(v)  # case {m}", ""),
                                  ("prompt_nl", draw + "print(v)", f"This is test case {m}.")):
                out.append({"cid": f"{s}|{loc}|{m}", "set": s, "location": loc, "mention": m, "kind": kind,
                            "code": code, "nl": nl})
    return out


def grid11():
    return grid(SETS11)


# F12: listed vs unlisted options. Each entry: (draw code, options, controls)
SETS12 = {
    "letters_unlisted": ("import string\nv = random.choice(string.ascii_uppercase[:6])\n",
                         ["A", "B", "C", "D", "E", "F"], ["P", "Q", "R", "S"]),
    "weekdays_unlisted": ("import calendar\nv = random.choice(list(calendar.day_name))\n",
                          ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                          ["January", "March", "June", "October"]),
    "ints_listed": ("v = random.choice([0, 1, 2, 3, 4, 5, 6, 7])\n",
                    ["0", "1", "2", "3", "4", "5", "6", "7"], ["8", "9", "P", "Q"]),
    "ints_unlisted": ("v = random.randrange(8)\n",
                      ["0", "1", "2", "3", "4", "5", "6", "7"], ["8", "9", "P", "Q"]),
}


def grid12():
    out = []
    for s, (draw, opts, ctrls) in SETS12.items():
        out.append({"cid": f"{s}|base|None", "set": s, "location": "base", "mention": None, "kind": "none",
                    "code": draw + "print(v)", "nl": ""})
        for m in opts + ctrls:
            kind = "option" if m in opts else "control"
            for loc, code, nl in (("assign_down", draw + f"tag = '{m}'\nprint(v)", ""),
                                  ("comment_down", draw + f"print(v)  # case {m}", ""),
                                  ("prompt_nl", draw + "print(v)", f"This is test case {m}.")):
                out.append({"cid": f"{s}|{loc}|{m}", "set": s, "location": loc, "mention": m, "kind": kind,
                            "code": code, "nl": nl})
    return out


ALL_SETS = {**SETS, **SETS11, **{k: (v[1], v[2]) for k, v in SETS12.items()}}


def body(c, prompt):
    t = prompt_text(c["code"], c["nl"])
    return (UNIFORM + "\n" + t) if prompt == "uniform" else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--f11", action="store_true", help="integer/letter option sets")
    ap.add_argument("--f12", action="store_true", help="listed vs unlisted option sets")
    a = ap.parse_args()
    from natural_factorization import chat
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    render = (lambda text: text + "\n\nOutput:\n") if a.raw else (lambda text: chat(tok, text, a.family))
    with a.out.open("w") as f:
        for p in ("base", "uniform"):
            for c in (grid12() if a.f12 else grid11() if a.f11 else grid()):
                opts = ALL_SETS[c["set"]][0]
                text = render(body(c, p))
                prefix = tok.encode(text + "v=", add_special_tokens=False)
                seqs, spans = [], []
                for o in opts:
                    # tokenize the full string, then score everything after the common prefix
                    # (this handles merges across the '=' boundary, e.g. '=A')
                    full_o = tok.encode(text + "v=" + o + "\n", add_special_tokens=False)
                    k = 0
                    while k < min(len(prefix), len(full_o)) and prefix[k] == full_o[k]:
                        k += 1
                    seqs.append(full_o)
                    spans.append((k, len(full_o)))
                k0 = min(k for k, _ in spans)  # score all options from the earliest divergence
                spans = [(k0, e) for _, e in spans]
                L = max(len(s) for s in seqs)
                pad = tok.pad_token_id if tok.pad_token_id is not None else 0
                inp = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=model.device)
                att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=model.device)
                with torch.inference_mode():
                    lp = torch.log_softmax(model(input_ids=inp, attention_mask=att, use_cache=False).logits.float(), -1)
                scores, first_mass = [], []
                for i, (s0, s1) in enumerate(spans):
                    tot = 0.0
                    for t in range(s0, s1):
                        tot += float(lp[i, t - 1, seqs[i][t]])
                    scores.append(tot)
                first = torch.exp(lp[0, spans[0][0] - 1]).cpu()
                first_ids = [seqs[i][spans[i][0]] for i in range(len(opts))]
                sc = np.array(scores)
                q = np.exp(sc - sc.max()); q /= q.sum()
                f.write(json.dumps({"model": a.model_name, "prompt": p, **{k: c[k] for k in ("cid", "set", "location", "mention", "kind")},
                                    "options": opts, "q": q.tolist(), "logp": sc.tolist(),
                                    "first_token_mass": float(first[first_ids].sum())}) + "\n")
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
