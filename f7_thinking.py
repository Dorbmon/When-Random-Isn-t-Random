#!/usr/bin/env python3
"""F7 (exploratory): does Qwen3 thinking mode remove context repulsion? Free generation."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repulsion_conditions import EXPRS, body, draws, located  # noqa: E402

N = 8
X = re.compile(r"(?<![A-Za-z_])x\s*=\s*(\d+)")
Y = re.compile(r"(?<![A-Za-z_])y\s*=\s*(\d+)")


def contexts():
    out = [{"cid": f"n{N}|e1|base|None", "n": N, "location": "base", "mention": None,
            "code": draws(N) + "print(max(x, y))", "nl": ""}]
    for d in range(10):
        out.append({"cid": f"n{N}|None|print_const|{d}", "n": N, "location": "print_const",
                    "mention": str(d), "code": draws(N) + f"print({d})", "nl": ""})
    for loc in ("assign_down", "comment_down"):
        for m in [str(d) for d in range(10)] + ["A", "B", "K", "Q"]:
            code, nl = located(loc, N, EXPRS["e1"], m)
            out.append({"cid": f"n{N}|e1|{loc}|{m}", "n": N, "location": loc, "mention": m,
                        "code": code, "nl": nl})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="sdpa", device_map="cuda").eval()
    pad = tok.pad_token_id or tok.eos_token_id
    done = {}
    if a.out.exists():
        for l in a.out.read_text().splitlines():
            r = json.loads(l)
            done[r["cid"]] = done.get(r["cid"], 0) + 1
    with a.out.open("a") as f:
        for j, c in enumerate(contexts()):
            have = done.get(c["cid"], 0)
            rendered = tok.apply_chat_template([{"role": "user", "content": body(c, "base")}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
            ids = tok.encode(rendered, add_special_tokens=False)
            for start in range(have, a.samples, a.batch):
                k = min(a.batch, a.samples - start)
                torch.manual_seed(20260926 + 1000 * j + start)
                inp = torch.tensor([ids] * k, device=model.device)
                with torch.inference_mode():
                    seq = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True,
                                         temperature=1.0, top_p=1.0, top_k=0,
                                         max_new_tokens=a.max_new_tokens, pad_token_id=pad)
                for t in seq[:, len(ids):]:
                    raw = tok.decode(t, skip_special_tokens=True)
                    closed = "</think>" in raw
                    ans = raw.split("</think>")[-1] if closed else ""
                    mx, my = X.search(ans), Y.search(ans)
                    f.write(json.dumps({"model": a.model_name, "prompt": "base", **{k2: c[k2] for k2 in ("cid", "n", "location", "mention")},
                                        "thinking_closed": closed, "n_tokens": int((t != pad).sum()),
                                        "answer": ans[-200:], "think_tail": raw.split("</think>")[0][-600:],
                                        "x": int(mx.group(1)) if mx else None,
                                        "y": int(my.group(1)) if my else None}) + "\n")
                f.flush()
            print("DONE", j + 1, flush=True)
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
