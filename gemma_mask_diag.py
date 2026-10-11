import os
import sys, numpy as np, torch
from arch import base_mask, layer_windows, text_parts
from f3_knockout import contexts, mention_positions
from natural_factorization import chat, digit_variants
from repulsion_conditions import body
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend
_ctx = sdpa_kernel([SDPBackend.MATH]); _ctx.__enter__()
tok = AutoTokenizer.from_pretrained(os.environ.get("GEMMA_TOKENIZER", os.environ["GEMMA_MODEL"]), local_files_only=True)
model = AutoModelForMultimodalLM.from_pretrained(os.environ["GEMMA_MODEL"], dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa", device_map="cuda").eval()
layers, norm, head, cfg = text_parts(model); L = len(layers); wins = layer_windows(cfg, L); dev = next(model.parameters()).device
variants = digit_variants(tok); legal = [(d, t) for d in range(8) for t in variants[d]]; ids_l = [t for _, t in legal]
seen = {}
st = {"masks": None}
def hook(m, a, k):
    if m._li not in seen:
        am = k.get("attention_mask")
        seen[m._li] = (type(am).__name__, None if am is None or not hasattr(am, "dtype") else (str(am.dtype), tuple(am.shape)))
    if st["masks"] is not None and m._li in st["masks"]: k["attention_mask"] = st["masks"][m._li]
    return a, k
for i, l in enumerate(layers): l.self_attn._li = i; l.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
c = [x for x in contexts() if x["location"] == "assign_down" and x["mention"] == "3"][0]
ids, mpos = mention_positions(tok, chat(tok, body(c, "base"), "gemma"), c); ids = ids + tok.encode("x=", add_special_tokens=False); T = len(ids)
def q(): 
    with torch.inference_mode(): lo = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1).logits[0, -1].float()
    p = torch.softmax(lo[ids_l], -1).cpu().numpy(); out = np.zeros(8)
    for j, (d, _) in enumerate(legal): out[d] += p[j]
    return out
q0 = q()
full_idx = [i for i in range(L) if wins[i] is None]
print("native mask types: sliding layer 0 ->", seen.get(0), "| full layer", full_idx[0], "->", seen.get(full_idx[0]))
print("q none", np.round(q0, 4))
def run(sel, kind):
    ms = {}
    for i in sel:
        m = base_mask(T, wins[i], dev)[None, None]
        ms[i] = m if kind == "bool" else torch.zeros(m.shape, dtype=torch.bfloat16, device=dev).masked_fill(~m, torch.finfo(torch.bfloat16).min)
    st["masks"] = ms; r = q(); st["masks"] = None; return r
for name, sel in (("all", range(L)), ("sliding only", [i for i in range(L) if wins[i]]), ("full only", [i for i in range(L) if wins[i] is None])):
    for kind in ("bool", "float"):
        r = run(sel, kind); print(f"{name:13s} {kind:5s} max|Δq|={np.abs(r - q0).max():.5f}")
# repeat native twice to see nondeterminism
print("native repeat max|Δq|", np.abs(q() - q0).max())
