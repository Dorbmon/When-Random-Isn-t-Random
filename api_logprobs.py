#!/usr/bin/env python3
"""Logprob arm: q(x) and q(y|x) read off top_logprobs at the first digit after 'x=' / 'y='.

One request per (spec, prompt, context, repeat). Resumable, with a hard USD cap.
Requests are only routed to providers that support logprobs (provider.require_parameters).
"""
import argparse
import asyncio
import json
import math
import os
import random
import re
from pathlib import Path

import aiohttp

from repulsion_conditions import body, conditions

HERE = Path(__file__).resolve().parent

SPECS = {
    "gpt-5.6-luna": dict(model="openai/gpt-5.6-luna", provider="OpenAI", reasoning={"effort": "none"}, top=20, price=(.2, 1.2)),
    "gpt-6-luna": dict(model="openai/gpt-6-luna", provider="OpenAI", reasoning={"effort": "none"}, top=20, price=(.1, .5)),
    "gpt-6-sol": dict(model="openai/gpt-6-sol", provider="OpenAI", reasoning={"effort": "none"}, top=20, price=(2, 10)),
    "deepseek-v4.1-flash": dict(model="deepseek/deepseek-v4.1-flash", provider="StreamLake", reasoning={"enabled": False}, top=20, price=(.15, .6)),
    "deepseek-v4-pro": dict(model="deepseek/deepseek-v4-pro-0813", provider="StreamLake", reasoning={"enabled": False}, top=20, price=(.35, 1.05)),
    "kimi-k3": dict(model="moonshotai/kimi-k3", provider="DigitalOcean", reasoning={"enabled": False}, top=20, price=(3, 15)),
    "qwen3.8-flash": dict(model="qwen/qwen3.8-flash", provider="Alibaba", reasoning={"enabled": False}, top=5, price=(.15, .47)),
    "qwen3.8-max": dict(model="qwen/qwen3.8-max-0902", provider="Alibaba", reasoning={"effort": "low"}, top=5, price=(2, 6)),
}
DIGIT = re.compile(r"^\s*(\d)\s*$")


def digit_dist(top):
    """Sum probability mass per digit over single-digit tokens in a top_logprobs list."""
    p = [0.0] * 10
    for t in top:
        m = DIGIT.match(t.get("token", ""))
        if m:
            p[int(m.group(1))] += math.exp(t["logprob"])
    return p


def read_positions(content_lp):
    """Find the digit token after 'x=' and after 'y='. Return {var: (sampled, dist, covered)}."""
    out = {}
    text = ""
    for tok in content_lp:
        t = tok.get("token", "")
        for var in ("x", "y"):
            if var not in out and re.search(rf"(?<![A-Za-z_]){var}\s*=\s*$", text) and DIGIT.match(t):
                d = digit_dist(tok.get("top_logprobs") or [])
                out[var] = {"sampled": int(DIGIT.match(t).group(1)), "digit_probs": d,
                            "digit_mass": sum(d)}
        text += t
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True, choices=list(SPECS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--grid", choices=("full", "api", "f2", "f4", "f5", "f10", "f11", "f12"), default="full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "api_data" / "logprobs.jsonl")
    ap.add_argument("--budget-usd", type=float, default=30.0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--rpm", type=float, default=19.0)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(s) for s in a.out.read_text().splitlines()] if a.out.exists() else []
    done = {(r["spec"], r["prompt"], r["cid"], r["repeat"]) for r in prior if r.get("ok")}
    spent = sum(float(r.get("cost_usd") or 0) for r in prior)
    # Budget shared with the free-generation ledger.
    fg = HERE / "api_data" / "raw.jsonl"
    if fg.exists():
        spent += sum(float(json.loads(s).get("cost_usd") or 0) for s in fg.read_text().splitlines())
    make_body = body
    if a.grid == "f2":
        import repulsion_f2
        grid, make_body = repulsion_f2.grid(), repulsion_f2.body
    elif a.grid in ("f10", "f11", "f12"):
        import f10_categorical
        grid = {"f10": f10_categorical.grid, "f11": f10_categorical.grid11, "f12": f10_categorical.grid12}[a.grid]()
        make_body = f10_categorical.body
        for c in grid:
            c["n"] = 4
    elif a.grid in ("f4", "f5"):
        import f45_grids
        grid, make_body = f45_grids.GRIDS[a.grid](), f45_grids.text_for
    elif a.grid == "full":
        grid = conditions()
    else:
        grid = conditions(ns=(8,), exprs=("e1",), locations=("comment_down", "comment_up", "prompt_nl"))
    state = {"spent": spent, "reserved": 0.0, "stop": False, "n": 0, "bad": 0}
    lock = asyncio.Lock()
    queues = {}
    for s in a.specs:
        q = asyncio.Queue()
        jobs = [(s, p, c, r) for r in range(a.repeats) for p in ("base", "uniform") for c in grid
                if (s, p, c["cid"], r) not in done]
        if a.limit:
            jobs = jobs[:a.limit]
        for j in jobs:
            q.put_nowait(j)
        queues[s] = q
    print("QUEUE", {s: q.qsize() for s, q in queues.items()}, "spent_all_arms", round(spent, 3),
          flush=True)
    slots = {s: 0.0 for s in a.specs}
    slot_lock = {s: asyncio.Lock() for s in a.specs}

    async def pace(s):
        loop = asyncio.get_running_loop()
        async with slot_lock[s]:
            now = loop.time()
            wait = max(0.0, slots[s] - now)
            slots[s] = max(now, slots[s]) + 60.0 / a.rpm
        await asyncio.sleep(wait)

    async def worker(session, s):
        spec = SPECS[s]
        q = queues[s]
        reserve = (300 * spec["price"][0] + (40 if spec["top"] == 20 or spec["reasoning"] != {"effort": "low"} else 800) * spec["price"][1]) / 1e6
        while not q.empty() and not state["stop"]:
            _, p, c, rep = q.get_nowait()
            async with lock:
                if state["spent"] + state["reserved"] + reserve > a.budget_usd:
                    state["stop"] = True
                    return
                state["reserved"] += reserve
            payload = {"model": spec["model"],
                       "messages": [{"role": "user", "content": make_body(c, p)}],
                       "temperature": 1.0, "logprobs": True, "top_logprobs": spec["top"],
                       "max_tokens": 24 if spec["reasoning"] != {"effort": "low"} else 2048,
                       "reasoning": spec["reasoning"],
                       "provider": {"order": [spec["provider"]], "allow_fallbacks": False},
                       "usage": {"include": True}}
            status, data, err = None, None, None
            for attempt in range(6):
                await pace(s)
                try:
                    async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                            json=payload) as r:
                        status = r.status
                        data = await r.json(content_type=None)
                    if status == 200 and not data.get("error"):
                        break
                    err = str(data.get("error"))[:300]
                except Exception as e:
                    err, status = f"{type(e).__name__}: {str(e)[:200]}", None
                await asyncio.sleep(1 if status == 429 else min(30, 2 ** attempt + random.random()))
            content, lp, usage, provider = "", [], {}, None
            if status == 200 and data and not data.get("error"):
                ch = (data.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content") or ""
                lp = ((ch.get("logprobs") or {}).get("content")) or []
                usage = data.get("usage") or {}
                provider = data.get("provider")
            pos = read_positions(lp)
            v_top = None
            if a.grid in ("f10", "f11", "f12"):
                text = ""
                for tok in lp:
                    t0 = tok.get("token", "")
                    if re.search(r"(?<![A-Za-z_])v\s*=\s*$", text):
                        v_top = {t["token"]: math.exp(t["logprob"]) for t in (tok.get("top_logprobs") or [])}
                        break
                    # merged '=A'-style token right after 'v': strip the '=' from every alternative
                    if re.search(r"(?<![A-Za-z_])v\s*$", text) and t0.startswith("=") and len(t0.strip()) > 1:
                        v_top = {}
                        for t in (tok.get("top_logprobs") or []):
                            if t["token"].startswith("="):
                                k = t["token"][1:]
                                v_top[k] = v_top.get(k, 0.0) + math.exp(t["logprob"])
                        break
                    text += t0
            cost = float(usage.get("cost") or (reserve if status == 200 else 0))
            ok = ("x" in pos and pos["x"]["digit_mass"] > 0.5) if a.grid not in ("f10", "f11", "f12") else bool(v_top)
            row = {"spec": s, "model": spec["model"], "provider": provider, "prompt": p,
                   "cid": c["cid"], "n": c["n"], "expr": c.get("expr"), "location": c.get("location"),
                   "cond": c.get("cond"), "family": c.get("family"), "level": c.get("level"),
                   "mention": c["mention"], "mention_type": c.get("mention_type"), "repeat": rep,
                   "http_status": status, "error": err, "raw": content, "ok": ok,
                   "top_logprobs": spec["top"], "reasoning_cfg": spec["reasoning"],
                   "v_top": v_top, "set": c.get("set"), "kind": c.get("kind"),
                   "x_lp": pos.get("x"), "y_lp": pos.get("y"), "n_logprob_tokens": len(lp),
                   "usage": usage, "cost_usd": cost}
            async with lock:
                with a.out.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                state["reserved"] -= reserve
                state["spent"] += cost
                state["n"] += 1
                state["bad"] += not ok
                if state["n"] % 200 == 0:
                    print("PROGRESS", state["n"], "not_ok", state["bad"], "spent_all_arms",
                          round(state["spent"], 3), {k: v.qsize() for k, v in queues.items()},
                          flush=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"}) as session:
        await asyncio.gather(*(worker(session, s) for s in a.specs for _ in range(a.concurrency)))
    print("COMPLETE written", state["n"], "not_ok", state["bad"], "spent_all_arms",
          round(state["spent"], 3), "budget_stop", state["stop"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
