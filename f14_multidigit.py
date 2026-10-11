#!/usr/bin/env python3
"""F14 (exploratory): multi-digit values, x = random.randrange(20). Token-level or number-level?

q(x=v) for v = 0..19 is the renormalized teacher-forced probability of the string v after
'x=' (followed by a newline). Mentions are in-range numbers 0..19, out-of-range numbers
20..29, and letters, at `tag = 'm'` or in the prompt text.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

N = 20
UNIFORM = "Each random draw must be independent and uniformly sampled from its possible values."
MENTIONS = [str(v) for v in range(30)] + ["A", "B", "K", "Q"]


def prompt_text(code, nl):
    t = ("Simulate exactly one execution of the Python 3 program. Pick a value for each random draw, "
         "then execute the remaining code. Use exactly two lines in this format:\n"
         "x=<number>\nanswer=<printed output>\n"
         "Do not explain or add any other text.\n\nimport random\n" + code)
    return (nl + "\n" + t) if nl else t


def grid():
    draw = f"x = random.randrange({N})\n"
    out = [{"cid": "base", "location": "base", "mention": None, "code": draw + "print(x)", "nl": ""}]
    for m in MENTIONS:
        out.append({"cid": f"assign_down|{m}", "location": "assign_down", "mention": m,
                    "code": draw + f"tag = '{m}'\nprint(x)", "nl": ""})
        out.append({"cid": f"prompt_nl|{m}", "location": "prompt_nl", "mention": m,
                    "code": draw + "print(x)", "nl": f"This is test case {m}."})
    return out


def body(c, p):
    t = prompt_text(c["code"], c["nl"])
    return (UNIFORM + "\n" + t) if p == "uniform" else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from natural_factorization import chat
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, local_files_only=True)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    render = (lambda text: text + "\n\nOutput:\n") if a.raw else (lambda text: chat(tok, text, a.family))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    with a.out.open("w") as f:
        for p in ("base", "uniform"):
            for c in grid():
                text = render(body(c, p))
                prefix = tok.encode(text + "x=", add_special_tokens=False)
                seqs = [tok.encode(text + "x=" + str(v) + "\n", add_special_tokens=False) for v in range(N)]
                ks = []
                for s in seqs:
                    k = 0
                    while k < min(len(prefix), len(s)) and prefix[k] == s[k]:
                        k += 1
                    ks.append(k)
                k0 = min(ks)
                L = max(len(s) for s in seqs)
                inp = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=model.device)
                att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=model.device)
                with torch.inference_mode():
                    lp = torch.log_softmax(model(input_ids=inp, attention_mask=att, use_cache=False).logits.float(), -1)
                sc = np.array([sum(float(lp[i, t - 1, s[t]]) for t in range(k0, len(s))) for i, s in enumerate(seqs)])
                q = np.exp(sc - sc.max()); q /= q.sum()
                f.write(json.dumps({"model": a.model_name, "prompt": p, **{k: c[k] for k in ("cid", "location", "mention")},
                                    "q": q.tolist(), "logp": sc.tolist()}) + "\n")
                f.flush()
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
