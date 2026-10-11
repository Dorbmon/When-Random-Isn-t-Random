#!/usr/bin/env python3
"""F15 (exploratory): the Qwen circuit analysis (F3, F8, F8b, F8c), run on a Gemma-style model.

All phases share one model load:
  F3   route and layer knockout of attention to the mention token, masks built per layer
       (sliding-window layers keep their window).
  F8   single-head knockout at the final query over the late layers.
  F8b  joint knockout of the top-k heads (picked automatically from F8), attention readout
       (eager pass) and direct logit attribution.
  F8c  patching sufficiency: the top-k heads' final-position outputs are copied from the
       digit context into the letter context.
Sanity gate: the causal-only mask must reproduce the unmasked q(x) within 0.02, otherwise the
script aborts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch import base_mask, layer_windows, text_parts  # noqa: E402
from f3_knockout import contexts, mention_positions  # noqa: E402
from natural_factorization import chat, digit_variants  # noqa: E402
from repulsion_conditions import body  # noqa: E402

N = 8
LETTERS = ("A", "B", "K", "Q")
IRR = ("assign_down", "comment_down", "prompt_nl", "varname_down")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--family", default="gemma")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=6, help="layer window size for F3")
    ap.add_argument("--head-batch", type=int, default=16)
    ap.add_argument("--scan-from", type=float, default=0.5, help="F8 scans layers from this fraction of depth")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--phases", nargs="+", default=["f3", "f8", "f8b", "f8c"])
    ap.add_argument("--f8b-tag", default="", help="suffix for the F8b output file (reruns)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N contexts (F3 phase)")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    # Math SDPA kernel for every pass: explicit masks otherwise switch Gemma's sliding layers onto a
    # different fused kernel, and the BF16 differences shift q(x) by up to 0.06 even with no knockout.
    math_ctx = sdpa_kernel([SDPBackend.MATH])  # keep a reference: a collected context manager exits
    math_ctx.__enter__()
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    variants = digit_variants(tok)
    cls = AutoModelForCausalLM if a.family == "qwen" else AutoModelForMultimodalLM
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                attn_implementation="sdpa", device_map="cuda").eval()
    layers, final_norm, lm_head, cfg = text_parts(model)
    L = len(layers)
    H = cfg.num_attention_heads
    wins = layer_windows(cfg, L)
    dev = next(model.parameters()).device
    print("ARCH layers", L, "heads", H, "sliding layers", sum(w is not None for w in wins), "window", cfg_win(cfg), flush=True)
    st = {"masks": None, "record": None, "patch": None, "store": {}}
    cap = {}

    def pre_hook(module, args, kwargs):
        if st["masks"] is not None and module._li in st["masks"]:
            kwargs["attention_mask"] = st["masks"][module._li]
        return args, kwargs

    def attn_out_hook(module, args, kwargs, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            cap[("w", module._li)] = output[1][0, :, -1].float().cpu()
        return output

    def o_pre(module, args):
        li = module._li
        x = args[0]
        D = x.shape[-1] // H
        if st["record"]:
            st["store"][li] = x[0, -1].detach().clone()
        if st["patch"]:
            hs = [h for (l2, h) in st["patch"] if l2 == li]
            if hs:
                x = x.clone()
                for h in hs:
                    x[0, -1, h * D:(h + 1) * D] = st["store"][li][h * D:(h + 1) * D]
                return (x,)
        cap[("o", li)] = x[0, -1].float()
        return None

    for i, layer in enumerate(layers):
        layer.self_attn._li = i
        layer.self_attn.o_proj._li = i
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        layer.self_attn.register_forward_hook(attn_out_hook, with_kwargs=True)
        layer.self_attn.o_proj.register_forward_pre_hook(o_pre)
    resid = {}
    final_norm.register_forward_pre_hook(lambda m, ar: resid.__setitem__("x", ar[0][0, -1].detach()))
    legal = [(d, t) for d in range(N) for t in variants[d]]
    ids_legal = [t for _, t in legal]
    canon = [variants[d][0] for d in range(N)]

    def qx_from(logits):
        pr = torch.softmax(logits[:, ids_legal].float(), -1).cpu().numpy()
        q = np.zeros((pr.shape[0], N))
        for j, (d, _) in enumerate(legal):
            q[:, d] += pr[:, j]
        return q

    ctxs = contexts()
    C = {(c["location"], c["mention"]): c for c in ctxs}

    def prep(c, p):
        rendered = chat(tok, body(c, p), a.family)
        ids, mpos = mention_positions(tok, rendered, c)
        ids = ids + tok.encode("x=", add_special_tokens=False)
        return ids, mpos

    def masks_for(T, layer_set, blocks, batch=1):
        """blocks: list per batch element of (head or None, query slice, key positions)."""
        out = {}
        for li in layer_set:
            m0 = base_mask(T, wins[li], dev)
            if all(h is None for h, _, _ in blocks):
                m = m0[None, None].expand(batch, 1, T, T).clone()
            else:
                m = m0[None, None].expand(batch, H, T, T).clone()
            for b, (h, qsl, keys) in enumerate(blocks):
                for k in keys:
                    col = torch.zeros(T, dtype=torch.bool, device=dev)
                    col[qsl] = True
                    col[:k + 1] = False
                    if h is None:
                        m[b, :, :, k] &= ~col
                    else:
                        m[b, h, :, k] &= ~col
            out[li] = m
        return out

    def forward(ids, batch=1):
        inp = torch.tensor([ids] * batch, device=dev)
        with torch.inference_mode():
            return model(input_ids=inp, use_cache=False, logits_to_keep=1).logits[:, -1]

    # ---------------- sanity gate ----------------
    ids, mpos = prep(C[("assign_down", "3")], "base")
    T = len(ids)
    st["masks"] = None
    q0 = qx_from(forward(ids))[0]
    st["masks"] = masks_for(T, range(L), [(None, slice(0, 0), [])])
    q1 = qx_from(forward(ids))[0]
    st["masks"] = None
    gap = float(np.abs(q0 - q1).max())
    print("SANITY causal-only vs none max|Δq| =", round(gap, 5), flush=True)
    if gap > 0.02:
        raise SystemExit("sanity gate failed: custom masks change the computation")

    # ---------------- F3 ----------------
    if "f3" in a.phases:
        wlist = [list(range(s, min(L, s + a.window))) for s in range(0, L, a.window)]
        kos = {"none": (None, []), "all": ("all", range(L)), "last_only": ("last", range(L)),
               "others_only": ("others", range(L)), "early_half_last": ("last", range(L // 2)),
               "late_half_last": ("last", range(L // 2, L))}
        for w in wlist:
            kos[f"win{w[0]:02d}-{w[-1]:02d}_last"] = ("last", w)
        with (a.outdir / f"f3_{a.model_name}.jsonl").open("w") as f:
            for p in ("base", "uniform"):
                for c in (ctxs[:a.limit] if a.limit else ctxs):
                    ids, mpos = prep(c, p)
                    T = len(ids)
                    res = {}
                    for name, (who, lays) in kos.items():
                        if who is None:
                            st["masks"] = None
                        else:
                            qsl = {"all": slice(None), "last": slice(T - 1, T), "others": slice(0, T - 1)}[who]
                            st["masks"] = masks_for(T, lays, [(None, qsl, mpos)])
                        res[name] = qx_from(forward(ids))[0].tolist()
                    st["masks"] = None
                    f.write(json.dumps({"model": a.model_name, "prompt": p, "cid": c["cid"], "n": N,
                                        "location": c["location"], "mention": c["mention"],
                                        "mention_token_idx": mpos, "n_layers": L, "q_x": res}) + "\n")
                    f.flush()
        print("F3 done", flush=True)

    # ---------------- F8 ----------------
    scan = list(range(int(L * a.scan_from), L))
    f8_path = a.outdir / f"f8_{a.model_name}.jsonl"
    if "f8" in a.phases:
        with f8_path.open("w") as f:
            for c in ctxs:
                ids, mpos = prep(c, "base")
                T = len(ids)
                st["masks"] = None
                q_none = qx_from(forward(ids))[0]
                res = {}
                for li in scan:
                    rows = []
                    for h0 in range(0, H, a.head_batch):
                        hs = list(range(h0, min(H, h0 + a.head_batch)))
                        st["masks"] = masks_for(T, [li], [(h, slice(T - 1, T), mpos) for h in hs], batch=len(hs))
                        rows.append(qx_from(forward(ids, batch=len(hs))))
                    st["masks"] = None
                    res[li] = np.concatenate(rows).round(6).tolist()
                f.write(json.dumps({"model": a.model_name, "prompt": "base", "cid": c["cid"], "location": c["location"],
                                    "mention": c["mention"], "q_none": q_none.round(6).tolist(), "q_head_knock": res}) + "\n")
                f.flush()
        print("F8 done", flush=True)

    # pick top heads from the F8 scores (same rule as Qwen 32B: combined positive share)
    if not f8_path.exists() or not ({"f8b", "f8c"} & set(a.phases)):
        print("COMPLETE", a.model_name, flush=True)
        return
    rows = [json.loads(l) for l in f8_path.read_text().splitlines()]
    R = {(r["location"], r["mention"]): r for r in rows}
    si = np.zeros((len(scan), H)); sp = np.zeros((len(scan), H))
    for a_i, li in enumerate(scan):
        for h in range(H):
            v1 = [R[(loc, str(c))]["q_head_knock"][str(li)][h][c] - R[(loc, str(c))]["q_none"][c]
                  - np.mean([R[(loc, Lt)]["q_head_knock"][str(li)][h][c] - R[(loc, Lt)]["q_none"][c] for Lt in LETTERS])
                  for loc in IRR for c in range(N)]
            v2 = [R[("print_const", str(c))]["q_head_knock"][str(li)][h][c] - R[("print_const", str(c))]["q_none"][c]
                  - np.mean([R[("print_const", str(w))]["q_head_knock"][str(li)][h][c] - R[("print_const", str(w))]["q_none"][c] for w in (8, 9)])
                  for c in range(N)]
            si[a_i, h], sp[a_i, h] = np.mean(v1), np.mean(v2)
    ti = -np.mean([R[(loc, str(c))]["q_none"][c] - np.mean([R[(loc, Lt)]["q_none"][c] for Lt in LETTERS]) for loc in IRR for c in range(N)])
    tp = -np.mean([R[("print_const", str(c))]["q_none"][c] - np.mean([R[("print_const", str(w))]["q_none"][c] for w in (8, 9)]) for c in range(N)])
    comb = np.maximum(si / max(ti, 1e-9), 0) + np.maximum(sp / max(tp, 1e-9), 0)
    order = np.dstack(np.unravel_index(np.argsort(-comb, axis=None), comb.shape))[0][:a.topk]
    top = [(scan[i], int(h)) for i, h in order]
    json.dump({"scan_layers": scan, "score_irr": si.tolist(), "score_pc": sp.tolist(), "total_irr": ti, "total_pc": tp,
               "top_heads": top}, (a.outdir / f"f8_scores_{a.model_name}.json").open("w"))
    print("TOP HEADS", [(f"L{l}H{h}", round(float(si[scan.index(l), h] / max(ti, 1e-9)), 3),
                         round(float(sp[scan.index(l), h] / max(tp, 1e-9)), 3)) for l, h in top], flush=True)

    # ---------------- F8b ----------------
    if "f8b" in a.phases:
        rng = np.random.default_rng(0)
        rand = []
        lays_top = sorted({l for l, _ in top[:3]})
        while len(rand) < 6:
            li = int(rng.choice(lays_top)); h = int(rng.integers(H))
            if (li, h) not in top and (li, h) not in rand:
                rand.append((li, h))
        heads = top[:3] + rand
        W_U = lm_head.weight
        with (a.outdir / f"f8b{a.f8b_tag}_{a.model_name}.jsonl").open("w") as f:
            for c in ctxs:
                ids, mpos = prep(c, "base")
                T = len(ids)
                # attention readout (eager) + DLA
                cap.clear(); st["masks"] = None
                try:
                    model.set_attn_implementation("eager")
                    with torch.inference_mode():
                        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, output_attentions=True, logits_to_keep=1)
                finally:
                    model.set_attn_implementation("sdpa")
                q_none = qx_from(out.logits[:, -1])[0]
                x = resid["x"]
                with torch.inference_mode():
                    nx = final_norm(x[None, None])[0, 0].float()
                    xf = x.float()
                    scale = nx / torch.where(xf.abs() > 1e-6, xf, torch.full_like(xf, 1e-6))
                    rec = {}
                    all_var = [t for d in range(N) for t in variants[d]]
                    full_p = torch.softmax(out.logits[0, -1].float(), -1)
                    var_prob = {str(t): float(full_p[t]) for t in all_var}
                    for (li, h) in heads:
                        Wo = layers[li].self_attn.o_proj.weight
                        D = Wo.shape[1] // H
                        o_in = cap[("o", li)][h * D:(h + 1) * D]
                        contrib = (Wo[:, h * D:(h + 1) * D].float() @ o_in) * scale
                        dla = (W_U[canon].float() @ contrib).cpu().numpy()
                        dla_all = (W_U[all_var].float() @ contrib).cpu().numpy()
                        w = cap.get(("w", li))
                        rec[f"L{li}H{h}"] = {"attn_mention": float(w[h][mpos].sum()) if w is not None else None,
                                             "dla": dla.round(4).tolist(),
                                             "dla_by_token": {str(t): round(float(v), 4) for t, v in zip(all_var, dla_all)},
                                             "candidate": (li, h) in top[:3]}
                joint = {}
                for k in (1, 2, 3, 5, 7, 10):
                    if k > len(top):
                        break
                    sel = top[:k]
                    ms = {}
                    for li in sorted({l for l, _ in sel}):
                        hs = [h for l2, h in sel if l2 == li]
                        m = masks_for(T, [li], [(hs[0], slice(T - 1, T), mpos)])[li]
                        for h in hs[1:]:
                            for kk in mpos:
                                m[0, h, T - 1, kk] = False
                        ms[li] = m
                    st["masks"] = ms
                    joint[f"top{k}"] = qx_from(forward(ids))[0].round(6).tolist()
                st["masks"] = None
                f.write(json.dumps({"model": a.model_name, "prompt": "base", "cid": c["cid"], "location": c["location"],
                                    "mention": c["mention"], "q_none": q_none.round(6).tolist(), "joint": joint,
                                    "heads": rec, "top": top, "digit_variants": {str(d): variants[d] for d in range(N)},
                                    "variant_prob": var_prob}) + "\n")
                f.flush()
        print("F8b done", flush=True)

    # ---------------- F8c ----------------
    if "f8c" in a.phases:
        rng = np.random.default_rng(1)
        with (a.outdir / f"f8c_{a.model_name}.jsonl").open("w") as f:
            for k in (3, 10):
                sel = top[:k]
                lays = sorted({l for l, _ in sel})
                rsets = []
                for _ in range(2):
                    s = []
                    while len(s) < k:
                        li = int(rng.choice(lays)); h = int(rng.integers(H))
                        if (li, h) not in sel and (li, h) not in s:
                            s.append((li, h))
                    rsets.append(s)
                for p in ("base", "uniform"):
                    for loc in IRR:
                        for c in range(N):
                            idd, _ = prep(C[(loc, str(c))], p)
                            st["record"], st["store"] = True, {}
                            qd = qx_from(forward(idd))[0]
                            st["record"] = False
                            for Lt in LETTERS:
                                idl, _ = prep(C[(loc, Lt)], p)
                                ql = qx_from(forward(idl))[0]
                                st["patch"] = sel
                                qp = qx_from(forward(idl))[0]
                                rq = []
                                for s in rsets:
                                    st["patch"] = s
                                    rq.append(qx_from(forward(idl))[0].tolist())
                                st["patch"] = None
                                f.write(json.dumps({"model": a.model_name, "k": k, "prompt": p, "location": loc, "c": c,
                                                    "letter": Lt, "q_digit": qd.tolist(), "q_letter": ql.tolist(),
                                                    "q_patch": qp.tolist(), "q_patch_random": rq}) + "\n")
                            f.flush()
        print("F8c done", flush=True)
    print("COMPLETE", a.model_name, flush=True)


def cfg_win(cfg):
    return getattr(cfg, "sliding_window", None)


if __name__ == "__main__":
    main()
