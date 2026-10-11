#!/usr/bin/env python3
"""F8c (exploratory): sufficiency test by activation patching.

For each location and digit c (with a letter control L at the same location), we run the
letter context and replace the candidate heads' outputs at the final position (o_proj input
slices) with their values from the digit-c context. If those heads carry the repulsion,
q(x=c) should fall in the patched letter context. Control: patch the same number of random
heads from the same layers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_knockout import contexts  # noqa: E402
from natural_factorization import chat, digit_variants  # noqa: E402
from repulsion_conditions import body  # noqa: E402

N = 8
LETTERS = ("A", "B", "K", "Q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--n-random-sets", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    variants = digit_variants(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="sdpa", device_map="cuda").eval()
    H, D = model.config.num_attention_heads, model.config.head_dim
    cand = [tuple(map(int, s.split(":"))) for s in a.heads.split(",")]
    rng = np.random.default_rng(1)
    layers_c = sorted({L for L, _ in cand})
    rand_sets = []
    for _ in range(a.n_random_sets):
        s = []
        while len(s) < len(cand):
            L = int(rng.choice(layers_c)); h = int(rng.integers(H))
            if (L, h) not in cand and (L, h) not in s:
                s.append((L, h))
        rand_sets.append(s)
    mods = model.model.layers
    st = {"mode": None, "store": {}, "heads": []}

    def o_pre(module, args):
        L = module.layer_idx
        x = args[0]
        if st["mode"] == "record":
            st["store"][L] = x[0, -1].detach().clone()
        elif st["mode"] == "patch":
            hs = [h for (LL, h) in st["heads"] if LL == L]
            if hs:
                x = x.clone()
                for h in hs:
                    x[0, -1, h * D:(h + 1) * D] = st["store"][L][h * D:(h + 1) * D]
                return (x,)
        return None

    for i, layer in enumerate(mods):
        layer.self_attn.o_proj.layer_idx = i
        if i in layers_c:
            layer.self_attn.o_proj.register_forward_pre_hook(o_pre)
    legal = [(d, t) for d in range(N) for t in variants[d]]
    ids_legal = [t for _, t in legal]

    def run(c, p):
        ids = tok.encode(chat(tok, body(c, p), "qwen"), add_special_tokens=False) + tok.encode("x=", add_special_tokens=False)
        with torch.inference_mode():
            lo = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1).logits[0, -1]
        pr = torch.softmax(lo[ids_legal].float(), -1).cpu().numpy()
        q = np.zeros(N)
        for j, (d, _) in enumerate(legal):
            q[d] += pr[j]
        return q, len(ids)

    C = {(c["location"], c["mention"]): c for c in contexts()}
    with a.out.open("w") as f:
        for p in ("base", "uniform"):
            for loc in ("assign_down", "comment_down", "prompt_nl", "varname_down"):
                for c in range(N):
                    st["mode"] = "record"; st["store"] = {}
                    q_digit, Td = run(C[(loc, str(c))], p)
                    for L in LETTERS:
                        st["mode"] = None
                        q_letter, Tl = run(C[(loc, L)], p)
                        st["mode"], st["heads"] = "patch", cand
                        q_patch, _ = run(C[(loc, L)], p)
                        rq = []
                        for s in rand_sets:
                            st["heads"] = s
                            rq.append(run(C[(loc, L)], p)[0].tolist())
                        st["mode"] = None
                        f.write(json.dumps({"model": a.model_name, "prompt": p, "location": loc, "c": c, "letter": L,
                                            "len_diff": Tl - Td, "q_digit": q_digit.tolist(), "q_letter": q_letter.tolist(),
                                            "q_patch": q_patch.tolist(), "q_patch_random": rq}) + "\n")
                f.flush()
    print("COMPLETE", a.model_name, flush=True)


if __name__ == "__main__":
    main()
