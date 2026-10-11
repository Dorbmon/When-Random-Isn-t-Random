#!/usr/bin/env python3
"""Natural-answer-start factorization."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

PAD_TO = 0


def build_cases():
    cases = []
    for n in range(2, 6):
        threshold = max(1, n // 2)
        mod0 = 2 + (2 * n) % 3
        expressions = {
            "sum": [
                "x + y", "(x + 2*y) % 10", "(2*x + y) % 10",
                "(2*x + 2*y) % 10", "(x + y + 1) % 10",
                "(x + 2*y + 1) % 10", "(2*x + 3*y) % 10",
                "(x + 3*y + 2) % 10",
            ],
            "max": [
                "max(x, y)", "min(x, y)", "max(x, 2*y)",
                "max(2*x, y)", "(max(x, y) + 1) % 10",
                "max(x + 1, y)", "max(x, y + 1)",
                "(max(x, y) + (x == y)) % 10",
            ],
            "mod": [
                f"(x + y) % {mod0}", "(x + 2*y) % 2", "(x + 2*y) % 3",
                "(x + 2*y) % 4", "(x + 2*y) % 5", "(x + 3*y) % 3",
                "(2*x + y) % 4", "(x*y) % 5",
            ],
            "branch": [
                f"y if x < {threshold} else x",
                f"x if y < {threshold} else y",
                f"y if x < {min(n, threshold+1)} else x",
                f"y if x >= {n-1} else x",
                f"(y + 1) % 10 if x < {threshold} else x",
                f"y if x < {threshold} else (x + 1) % 10",
                "y if x % 2 == 0 else x", "x if y % 2 == 0 else y",
            ],
        }
        for op, exprs in expressions.items():
            for variant, expr in enumerate(exprs):
                code = (f"x = random.randrange({n})\n"
                        f"y = random.randrange({n})\nprint({expr})")
                if op == "branch" and variant == 0:
                    code = (f"x = random.randrange({n})\n"
                            f"y = random.randrange({n})\n"
                            f"if x < {threshold}:\n    print(y)\nelse:\n    print(x)")
                outputs = [int(eval(expr, {"x": x, "y": y}))
                           for x in range(n) for y in range(n)]
                assert all(0 <= z <= 9 for z in outputs), (op, n, variant, outputs)
                truth = np.bincount(outputs, minlength=10).astype(float) / (n*n)
                cases.append({"name": f"{op}_n{n}_v{variant}", "op": op,
                              "n": n, "variant": variant, "expression": expr,
                              "code": code, "truth": truth.tolist()})
    assert len(cases) == 128 and len({c["code"] for c in cases}) == 128
    return cases


def chat(tokenizer, body, family):
    kwargs = {"reasoning_strength": "low"} if family == "muse" else {"enable_thinking": False}
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False,
        add_generation_prompt=True, **kwargs)
    if family == "muse":
        assert rendered.endswith("<|start|>assistant")
        rendered += " to=user<|message|>"
    elif family == "gemma":
        suffix = "<|channel>thought\n<channel|>"
        assert rendered.endswith(suffix)
    return rendered


def cot_prompt(code):
    return ("Simulate exactly one execution of the Python 3 program. Pick a value for each random draw, "
            "then execute the remaining code. Use exactly three lines in this format:\n"
            "x=<digit>\ny=<digit>\nanswer=<digit>\n"
            "Do not explain or add any other text.\n\nimport random\n" + code)


def tv(p, q):
    return float(np.abs(np.asarray(p) - np.asarray(q)).sum() / 2)


def tv_with_invalid(p, truth):
    """TV on outputs 0..9 plus an invalid-token outcome."""
    p = np.asarray(p)
    return float((np.abs(p-np.asarray(truth)).sum() + abs(1-p.sum()))/2)


def digit_variants(tokenizer):
    special = set(tokenizer.all_special_ids)
    found = {i: [] for i in range(10)}
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        surface = tokenizer.decode([token_id], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        match = re.fullmatch(r"[ \t]*([0-9])[ \t]*", surface)
        if match:
            found[int(match.group(1))].append(token_id)
    assert all(found[i] for i in range(10))
    return found


def score_candidates(model, sequences, candidates, batch_size):
    """Raw probabilities and probabilities conditional on a legal one-token digit."""
    import torch
    out = []
    masses = []
    for start in range(0, len(sequences), batch_size):
        group = sequences[start:start+batch_size]
        assert len({len(s) for s in group}) == 1
        original_size = len(group)
        if PAD_TO:
            assert PAD_TO >= original_size
            group = group + [group[-1]] * (PAD_TO-original_size)
        toks = torch.tensor(group, dtype=torch.long, device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=toks, use_cache=False,
                           logits_to_keep=1).logits[:, -1, :].float()
            chosen = logits[:, candidates]
            conditional = torch.softmax(chosen, dim=-1).cpu().numpy().astype(np.float64)
            conditional /= conditional.sum(axis=1, keepdims=True)
            mass = torch.exp(torch.logsumexp(chosen, dim=-1)
                             - torch.logsumexp(logits, dim=-1)).cpu().numpy().astype(np.float64)
        out.extend(conditional[:original_size])
        masses.extend(np.clip(mass[:original_size], 0.0, 1.0))
    return np.asarray(out), np.asarray(masses)


def run_case_variant(model, tokenizer, case, prompt_kind, variants, batch_size, family,
                     x_separator="\n", y_separator="\n", trajectory_start="x="):
    n = case["n"]
    body = cot_prompt(case["code"])
    if prompt_kind == "uniform":
        body = "Each random draw must be independent and uniformly sampled from its possible values.\n" + body
    base = tokenizer.encode(chat(tokenizer, body, family), add_special_tokens=False)
    intro = tokenizer.encode(trajectory_start, add_special_tokens=False)
    sep_y = tokenizer.encode(x_separator + "y=", add_special_tokens=False)
    sep_answer = tokenizer.encode(y_separator + "answer=", add_special_tokens=False)
    prefix = base + intro
    x_choices = [(x, token) for x in range(n) for token in variants[x]]
    xy_choices = [(z, token) for z in range(10) for token in variants[z]]
    x_token_probs, x_mass = score_candidates(model, [prefix],
                                              [t for _, t in x_choices], batch_size)
    x_token_probs = x_token_probs[0]
    y_prefixes = [prefix + [token] + sep_y for _, token in x_choices]
    y_token_probs, y_mass = score_candidates(model, y_prefixes,
                                              [t for _, t in x_choices], batch_size)
    paths = []
    z_prefixes = []
    for xi, (x, xt) in enumerate(x_choices):
        for yi, (y, yt) in enumerate(x_choices):
            paths.append((xi, yi, x, y))
            z_prefixes.append(y_prefixes[xi] + [yt] + sep_answer)
    z_token_probs, z_mass = score_candidates(model, z_prefixes,
                                              [t for _, t in xy_choices], batch_size)
    qx = np.zeros(n)
    qy_joint = np.zeros((n, n))
    qz_joint = np.zeros((n, n, 10))
    raw_xy = np.zeros((n, n))
    raw_z_joint = np.zeros((n, n, 10))
    raw_full = np.zeros(10)
    variant_x_mass = 0.0
    variant_y_mass = 0.0
    variant_z_mass = 0.0
    canonical = {i: tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)}
    canonical_x_indices = [next(i for i, (value, token) in enumerate(x_choices)
                                if value == x and token == canonical[x]) for x in range(n)]
    canonical_z_indices = [next(i for i, (value, token) in enumerate(xy_choices)
                                if value == z and token == canonical[z]) for z in range(10)]
    for xi, (x, xt) in enumerate(x_choices):
        qx[x] += x_token_probs[xi]
        if xt != canonical[x]:
            variant_x_mass += x_token_probs[xi]
    for k, (xi, yi, x, y) in enumerate(paths):
        xt = x_choices[xi][1]
        yt = x_choices[yi][1]
        path_prob = x_token_probs[xi] * y_token_probs[xi, yi]
        qy_joint[x, y] += path_prob
        raw_path = x_mass[0] * x_token_probs[xi] * y_mass[xi] * y_token_probs[xi, yi]
        raw_xy[x, y] += raw_path
        if yt != canonical[y]:
            variant_y_mass += path_prob
        for zi, (z, zt) in enumerate(xy_choices):
            pz = z_token_probs[k, zi]
            qz_joint[x, y, z] += path_prob * pz
            raw_z_joint[x, y, z] += path_prob * z_mass[k] * pz
            raw_full[z] += raw_path * z_mass[k] * pz
            if zt != canonical[z]:
                variant_z_mass += path_prob * pz
    p_xy = qy_joint
    assert np.all(p_xy > 0)
    qy = p_xy / qx[:, None]
    qz = qz_joint / p_xy[:, :, None]
    raw_z = raw_z_joint / p_xy[:, :, None]
    u_xy = np.full((n, n), 1/(n*n))
    truth = np.asarray(case["truth"])
    canonical_full = np.zeros(10)
    canonical_raw_full = np.zeros(10)
    canonical_x = x_token_probs[canonical_x_indices]
    canonical_x /= canonical_x.sum()
    for x, xi in enumerate(canonical_x_indices):
        canonical_y = y_token_probs[xi, canonical_x_indices]
        canonical_y /= canonical_y.sum()
        for y, yi in enumerate(canonical_x_indices):
            k = xi*len(x_choices) + yi
            canonical_z = z_token_probs[k, canonical_z_indices]
            canonical_z /= canonical_z.sum()
            canonical_full += canonical_x[x] * canonical_y[y] * canonical_z
            canonical_raw_full += (x_mass[0] * x_token_probs[xi] *
                                   y_mass[xi] * y_token_probs[xi, yi] *
                                   z_mass[k] * z_token_probs[k, canonical_z_indices])
    oracle_exec_llm_sample = np.zeros(10)
    raw_oe = np.zeros(10)
    correct = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            z = int(eval(case["expression"], {"x": x, "y": y}))
            oracle_exec_llm_sample[z] += p_xy[x, y]
            raw_oe[z] += raw_xy[x, y]
            correct[x, y] = qz[x, y, z]
    llm_exec_oracle_sample = np.einsum("xy,xyz->z", u_xy, qz)
    full = qz_joint.sum(axis=(0, 1))
    raw_os = np.einsum("xy,xyz->z", u_xy, raw_z)
    sums = [float(p.sum()) for p in
            (p_xy, oracle_exec_llm_sample, llm_exec_oracle_sample, full)]
    assert all(abs(total-1) < 1e-7 for total in sums), sums
    return {
        "name": case["name"], "op": case["op"], "n": n,
        "trajectory_start": trajectory_start, "x_separator": x_separator,
        "y_separator": y_separator,
        "variant": case["variant"], "prompt": prompt_kind, "code": case["code"],
        "truth": case["truth"],
        "joint_sampling_tv": tv(p_xy, u_xy),
        "sampling_output_tv": tv(oracle_exec_llm_sample, truth),
        "execution_output_tv": tv(llm_exec_oracle_sample, truth),
        "full_output_tv": tv(full, truth),
        "oracle_output_tv": 0.0,
        "validity_adjusted_sampling_output_tv": tv_with_invalid(raw_oe, truth),
        "validity_adjusted_execution_output_tv": tv_with_invalid(raw_os, truth),
        "validity_adjusted_full_output_tv": tv_with_invalid(raw_full, truth),
        "canonical_only_full_output_tv": tv(canonical_full, truth),
        "canonical_only_validity_adjusted_full_output_tv": tv_with_invalid(canonical_raw_full, truth),
        "oracle_sampling_execution_error": float(1-np.sum(u_xy*correct)),
        "llm_sampling_execution_error": float(1-np.sum(p_xy*correct)),
        "digit_mass_x": float(x_mass[0]),
        "digit_mass_y_min": float(y_mass.min()),
        "digit_mass_z_min": float(z_mass.min()),
        "allowed_y_mass_given_model_x": float(np.sum(x_token_probs * y_mass)),
        "allowed_two_draw_mass": float(x_mass[0] * np.sum(x_token_probs * y_mass)),
        "noncanonical_digit_mass_x_conditional": float(variant_x_mass),
        "noncanonical_digit_mass_y_conditional": float(variant_y_mass),
        "noncanonical_digit_mass_z_conditional": float(variant_z_mass),
        "distributions": {
            "oracle": case["truth"], "full_llm": full.tolist(),
            "python_sampling_llm_execution": llm_exec_oracle_sample.tolist(),
            "llm_sampling_python_execution": oracle_exec_llm_sample.tolist(),
            "llm_xy": p_xy.tolist(), "llm_x": qx.tolist(),
            "llm_y_given_x": qy.tolist(), "llm_execution_z_given_xy": qz.tolist(),
            "validity_adjusted_full": raw_full.tolist(),
            "validity_adjusted_python_sampling_llm_execution": raw_os.tolist(),
            "validity_adjusted_llm_sampling_python_execution": raw_oe.tolist(),
        },
    }


def strict_direct_prompt(code):
    return ("Simulate exactly one execution of the Python 3 program below. Each random call is independent and uniformly samples according to Python semantics. "
            "Output exactly one digit from 0 to 9: the printed result. No explanation, punctuation, or other text.\n\n"
            "import random\n" + code)


def simple_direct_prompt(code):
    return ("Run this Python 3 program once. Reply with only the single digit it prints.\n"
            "```python\nimport random\n" + code + "\n```")


def run_invariance(model, tokenizer, variants, family, pairs_path, out_path, model_name):
    pairs = json.loads(pairs_path.read_text())
    assert len(pairs) == 300
    codes = list(dict.fromkeys(code for pair in pairs
                               for code in (pair["code_a"], pair["code_b"])))
    assert len(codes) == 365
    candidates = [(z, token) for z in range(10) for token in variants[z]]
    scores = {}
    for prompt_name, prompt_fn in (("strict", strict_direct_prompt),
                                   ("simple", simple_direct_prompt)):
        scores[prompt_name] = {}
        for index, code in enumerate(codes, 1):
            ids = tokenizer.encode(chat(tokenizer, prompt_fn(code), family),
                                   add_special_tokens=False)
            p, mass = score_candidates(model, [ids],
                                       [token for _, token in candidates], 1)
            dist = np.zeros(10)
            for (z, _), prob in zip(candidates, p[0]):
                dist[z] += prob
            scores[prompt_name][code] = {"p": dist.tolist(),
                                         "digit_mass": float(mass[0])}
            if index % 50 == 0:
                print("INVARIANCE", model_name, prompt_name, index, "/", len(codes),
                      flush=True)
    results = []
    for pair in pairs:
        row = dict(pair)
        row["model_name"] = model_name
        row["model"] = {}
        for prompt_name in scores:
            a = scores[prompt_name][pair["code_a"]]
            b = scores[prompt_name][pair["code_b"]]
            row["model"][prompt_name] = {
                "p_a": a["p"], "p_b": b["p"],
                "digit_mass_a": a["digit_mass"],
                "digit_mass_b": b["digit_mass"],
                "pair_tv": tv(a["p"], b["p"]),
                "tv_a_to_truth": tv(a["p"], pair["true_p_a"]),
                "tv_b_to_truth": tv(b["p"], pair["true_p_b"]),
            }
        row["prompt_tv_a"] = tv(scores["strict"][pair["code_a"]]["p"],
                                scores["simple"][pair["code_a"]]["p"])
        row["prompt_tv_b"] = tv(scores["strict"][pair["code_b"]]["p"],
                                scores["simple"][pair["code_b"]]["p"])
        results.append(row)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print("INVARIANCE_COMPLETE", model_name, len(results), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--model-name")
    ap.add_argument("--family", choices=("qwen", "muse", "gemma"))
    ap.add_argument("--invariance-pairs", type=Path)
    ap.add_argument("--invariance-only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--memory-fraction", type=float, default=.96)
    ap.add_argument("--gpu-max-memory-gib", type=int, default=0)
    ap.add_argument("--cases-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--case-name", default="")
    ap.add_argument("--prompts", nargs="+", choices=("base", "uniform"),
                    default=["base", "uniform"])
    ap.add_argument("--trajectory-start", default="x=",
                    help="Fixed reply text before the first digit; escaped whitespace accepted")
    ap.add_argument("--pad-to", type=int, default=0,
                    help="Pad every forward pass to this batch size for BF16 numerical consistency")
    ap.add_argument("--x-separator", default="\\n",
                    help="Text after the x digit and before y=; escaped whitespace accepted")
    ap.add_argument("--y-separator", default="\\n",
                    help="Text after the y digit and before answer=; escaped whitespace accepted")
    args = ap.parse_args()
    global PAD_TO
    PAD_TO = args.pad_to
    assert PAD_TO == 0 or PAD_TO >= args.batch_size
    args.x_separator = bytes(args.x_separator, "utf8").decode("unicode_escape")
    args.y_separator = bytes(args.y_separator, "utf8").decode("unicode_escape")
    args.trajectory_start = bytes(args.trajectory_start, "utf8").decode("unicode_escape")
    if args.invariance_only:
        assert args.invariance_pairs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    if args.cases_only:
        print("VALIDATED", len(cases), flush=True)
        return
    assert args.model and args.model_name and args.family
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                              local_files_only=True)
    variants = digit_variants(tokenizer)
    model_cls = AutoModelForCausalLM if args.family == "qwen" else AutoModelForMultimodalLM
    load_kwargs = {"device_map": "auto" if args.gpu_max_memory_gib else "cuda"}
    if args.gpu_max_memory_gib:
        load_kwargs["max_memory"] = {0: f"{args.gpu_max_memory_gib}GiB", "cpu": "300GiB"}
    model = model_cls.from_pretrained(
        args.model, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa",
        low_cpu_mem_usage=True, **load_kwargs)
    model.eval()
    rows_file = out / f"{args.model_name}.jsonl"
    done = set()
    if rows_file.exists():
        for line in rows_file.read_text().splitlines():
            r = json.loads(line)
            done.add((r["prompt"], r["name"]))
    total = 0
    if not args.invariance_only:
        with rows_file.open("a") as file:
            for prompt_kind in args.prompts:
                selected = ([c for c in cases if c["name"] == args.case_name]
                            if args.case_name else
                            cases[:args.limit] if args.limit else cases)
                for case in selected:
                    if (prompt_kind, case["name"]) in done:
                        continue
                    row = run_case_variant(model, tokenizer, case, prompt_kind,
                                           variants, args.batch_size, args.family,
                                           args.x_separator, args.y_separator,
                                           args.trajectory_start)
                    row["model"] = args.model_name
                    row["score_batch_size"] = args.batch_size
                    row["score_pad_to"] = PAD_TO
                    file.write(json.dumps(row) + "\n")
                    file.flush()
                    total += 1
                    if total % 8 == 0:
                        print("DONE_ROWS", args.model_name, total,
                              "latest", prompt_kind, case["name"], flush=True)
        print("COMPLETE", args.model_name, total, flush=True)
    if args.invariance_pairs:
        path = out / f"invariance_{args.model_name}.json"
        if not path.exists():
            run_invariance(model, tokenizer, variants, args.family,
                           args.invariance_pairs, path, args.model_name)


if __name__ == "__main__":
    main()
