#!/usr/bin/env python3
"""F19 (exploratory, reviewer request): why do DeepSeek V4 Pro and GPT-6 Sol attract in the
single-draw print(v) design (F12) but repel in the main two-draw grid?

Five cells bridge the two designs one factor at a time. Every cell has randrange(8) draws and
`tag = 'm'` placed after the draws; m is a digit 0-7, or a letter A/B/K/Q / digit 8/9 as control.
  M   main grid: x, y draws; print(max(x, y)); template x=<digit>/y=<digit>/answer=<digit>
  C1  x, y draws; print(x); main template
  C2  x draw only; print(x); template x=<digit>/answer=<digit>
  C4  x draw only; print(x); F12 template x=<value, without quotes>/answer=<printed output>
  C3  v draw only; print(v); F12 template v=<value, without quotes>/answer=<printed output> (= F12)
Read q over 0..7 from the top-20 logprobs at the first digit after '<var>=' (merged '=3' handled).
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

SPECS = {
    "deepseek-v4-pro": dict(model="deepseek/deepseek-v4-pro-0813", provider="StreamLake", reasoning={"enabled": False}),
    "gpt-6-sol": dict(model="openai/gpt-6-sol", provider="OpenAI", reasoning={"effort": "none"}),
    "deepseek-v4.1-flash": dict(model="deepseek/deepseek-v4.1-flash", provider="StreamLake", reasoning={"enabled": False}),
    "gpt-6-luna": dict(model="openai/gpt-6-luna", provider="OpenAI", reasoning={"effort": "none"}),
}
UNIFORM = "Each random draw must be independent and uniformly sampled from its possible values."
MENTIONS = [str(d) for d in range(10)] + ["A", "B", "K", "Q"]


def template(kind, var):
    if kind == "main":
        lines = "x=<digit>\ny=<digit>\nanswer=<digit>"
        n = "three"
    elif kind == "main1":
        lines = f"{var}=<digit>\nanswer=<digit>"
        n = "two"
    else:
        lines = f"{var}=<value, without quotes>\nanswer=<printed output>"
        n = "two"
    return ("Simulate exactly one execution of the Python 3 program. Pick a value for each random draw, "
            f"then execute the remaining code. Use exactly {n} lines in this format:\n{lines}\n"
            "Do not explain or add any other text.\n\nimport random\n")


CELLS = {
    "M": ("main", "x", "x = random.randrange(8)\ny = random.randrange(8)\ntag = '{m}'\nprint(max(x, y))"),
    "C1": ("main", "x", "x = random.randrange(8)\ny = random.randrange(8)\ntag = '{m}'\nprint(x)"),
    "C2": ("main1", "x", "x = random.randrange(8)\ntag = '{m}'\nprint(x)"),
    "C4": ("f12", "x", "x = random.randrange(8)\ntag = '{m}'\nprint(x)"),
    "C3": ("f12", "v", "v = random.randrange(8)\ntag = '{m}'\nprint(v)"),
}


def grid():
    out = []
    for cell, (tk, var, code) in CELLS.items():
        for m in MENTIONS:
            out.append({"cell": cell, "var": var, "mention": m, "text": template(tk, var) + code.format(m=m)})
    return out


def read_digit(lp, var):
    text = ""
    for tok in lp:
        t0 = tok.get("token", "")
        if re.search(rf"(?<![A-Za-z_]){var}\s*=\s*$", text):
            return {t["token"].strip(): math.exp(t["logprob"]) for t in (tok.get("top_logprobs") or [])}
        if re.search(rf"(?<![A-Za-z_]){var}\s*$", text) and t0.startswith("=") and len(t0.strip()) > 1:
            d = {}
            for t in (tok.get("top_logprobs") or []):
                if t["token"].startswith("="):
                    k = t["token"][1:].strip()
                    d[k] = d.get(k, 0.0) + math.exp(t["logprob"])
            return d
        text += t0
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rpm", type=float, default=19)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    jobs = [(s, p, c, r) for s in SPECS for r in range(a.repeats) for p in ("base", "uniform") for c in grid()]
    done = set()
    if a.out.exists():
        for l in a.out.read_text().splitlines():
            x = json.loads(l)
            if x.get("q"):
                done.add((x["spec"], x["prompt"], x["cell"], x["mention"], x["repeat"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2]["cell"], j[2]["mention"], j[3]) not in done]
    print("QUEUE", len(jobs), flush=True)
    slots = {s: 0.0 for s in SPECS}
    locks = {s: asyncio.Lock() for s in SPECS}
    wlock = asyncio.Lock()
    queues = {s: asyncio.Queue() for s in SPECS}
    for j in jobs:
        queues[j[0]].put_nowait(j)
    spent = [0.0]

    async def pace(s):
        loop = asyncio.get_running_loop()
        async with locks[s]:
            now = loop.time(); w = max(0.0, slots[s] - now); slots[s] = max(now, slots[s]) + 60 / a.rpm
        await asyncio.sleep(w)

    async def worker(session, s):
        spec = SPECS[s]
        while not queues[s].empty():
            _, p, c, r = queues[s].get_nowait()
            text = (UNIFORM + "\n" + c["text"]) if p == "uniform" else c["text"]
            payload = {"model": spec["model"], "messages": [{"role": "user", "content": text}], "temperature": 1.0,
                       "max_tokens": 24, "logprobs": True, "top_logprobs": 20, "reasoning": spec["reasoning"],
                       "provider": {"order": [spec["provider"]], "allow_fallbacks": False}, "usage": {"include": True}}
            data, status = None, None
            for attempt in range(6):
                await pace(s)
                try:
                    async with session.post("https://openrouter.ai/api/v1/chat/completions", json=payload) as resp:
                        status = resp.status; data = await resp.json(content_type=None)
                    if status == 200 and not data.get("error"):
                        break
                except Exception:
                    status = None
                await asyncio.sleep(1 if status == 429 else min(20, 2 ** attempt + random.random()))
            q, raw, cost = None, "", 0.0
            if status == 200 and data and not data.get("error"):
                ch = (data.get("choices") or [{}])[0]
                raw = (ch.get("message") or {}).get("content") or ""
                top = read_digit(((ch.get("logprobs") or {}).get("content")) or [], c["var"])
                cost = float((data.get("usage") or {}).get("cost") or 0)
                if top:
                    v = [sum(pv for k, pv in top.items() if k == str(d)) for d in range(8)]
                    if sum(v) > 0.3:
                        q = [x / sum(v) for x in v]
            async with wlock:
                spent[0] += cost
                with a.out.open("a") as f:
                    f.write(json.dumps({"spec": s, "prompt": p, "cell": c["cell"], "mention": c["mention"], "repeat": r,
                                        "http_status": status, "raw": raw, "q": q, "cost_usd": cost}) + "\n")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
                                     headers={"Authorization": "Bearer " + key}) as session:
        await asyncio.gather(*(worker(session, s) for s in SPECS for _ in range(4)))
    print("COMPLETE spent", round(spent[0], 4), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
