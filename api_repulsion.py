#!/usr/bin/env python3
"""Free-generation arm of the context-repulsion experiment through OpenRouter.

Resumable: successful (spec, prompt, cid, sample_index) rows are skipped. A hard USD cap
covers everything in the ledger, plus a reservation for requests still in flight.
"""
import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

import aiohttp

from repulsion_conditions import body, conditions

HERE = Path(__file__).resolve().parent

# label -> request settings. Each price is ($/M prompt, $/M completion) for the reservation.
SPECS = {
    "gpt-5.6-luna": dict(model="openai/gpt-5.6-luna", reasoning={"effort": "none"},
                         max_tokens=64, price=(.2, 1.2)),
    "gemini-3.8-flash": dict(model="google/gemini-3.8-flash", reasoning={"effort": "minimal"},
                             max_tokens=512, price=(.75, 3.75)),
    "claude-sonnet-5": dict(model="anthropic/claude-sonnet-5", reasoning={"enabled": False},
                            max_tokens=64, price=(2, 10)),
    "deepseek-v4.1-flash": dict(model="deepseek/deepseek-v4.1-flash",
                                reasoning={"enabled": False}, max_tokens=64, price=(.15, .6)),
    "gpt-6-luna": dict(model="openai/gpt-6-luna", reasoning={"effort": "none"},
                       max_tokens=64, price=(.1, .5)),
    "gpt-6-luna-reasoning": dict(model="openai/gpt-6-luna", reasoning={"effort": "medium"},
                                 max_tokens=4096, price=(.1, .5)),
    "claude-opus-5.5-reasoning": dict(model="anthropic/claude-opus-5.5",
                                      reasoning={"effort": "low"}, max_tokens=4096,
                                      price=(4, 20)),
    "gpt-6-sol-reasoning": dict(model="openai/gpt-6-sol", reasoning={"effort": "low"},
                                max_tokens=4096, price=(2, 10)),
}

STRICT = re.compile(r"\s*x\s*=\s*(\d)\s*\n\s*y\s*=\s*(\d)\s*\n\s*answer\s*=\s*(\d)\s*")
LENIENT = {k: re.compile(rf"(?<![A-Za-z_]){k}\s*=\s*(\d+)") for k in ("x", "y", "answer")}


def api_grid():
    """n=8, e1, three letter-matched locations, print_const, and base."""
    return conditions(ns=(8,), exprs=("e1",),
                      locations=("comment_down", "comment_up", "prompt_nl"))


def parse(text):
    m = STRICT.fullmatch(text or "")
    out = {"strict": m is not None}
    for k, rx in LENIENT.items():
        mm = rx.search(text or "")
        out[k] = int(mm.group(1)) if mm else None
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True, choices=list(SPECS))
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--prompts", nargs="+", default=["base", "uniform"])
    ap.add_argument("--out", type=Path, default=HERE / "api_data" / "raw.jsonl")
    ap.add_argument("--budget-usd", type=float, default=45.0)
    ap.add_argument("--concurrency", type=int, default=8, help="per spec")
    ap.add_argument("--first-sample", type=int, default=0)
    ap.add_argument("--rpm", type=float, default=19.0, help="per-spec request rate cap")
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(s) for s in a.out.read_text().splitlines()] if a.out.exists() else []
    done = {(r["spec"], r["prompt"], r["cid"], r["sample_index"]) for r in prior
            if r.get("http_status") == 200 and r.get("raw")}
    state = {"spent": sum(float(r.get("cost_usd") or 0) for r in prior), "reserved": 0.0,
             "stop": False, "n": 0, "fail": 0}
    lock = asyncio.Lock()
    grid = api_grid()
    queues = {}
    for s in a.specs:
        q = asyncio.Queue()
        for i in range(a.first_sample, a.samples):
            for p in a.prompts:
                for c in grid:
                    if (s, p, c["cid"], i) not in done:
                        q.put_nowait((s, p, c, i))
        queues[s] = q
    print("QUEUE", {s: q.qsize() for s, q in queues.items()}, "spent", round(state["spent"], 4),
          flush=True)

    rpm = {s: (None if s.startswith("deepseek") else a.rpm) for s in a.specs}
    slots = {s: 0.0 for s in a.specs}
    slot_lock = {s: asyncio.Lock() for s in a.specs}

    async def pace(s):
        if rpm[s] is None:
            return
        loop = asyncio.get_running_loop()
        async with slot_lock[s]:
            now = loop.time()
            wait = max(0.0, slots[s] - now)
            slots[s] = max(now, slots[s]) + 60.0 / rpm[s]
        await asyncio.sleep(wait)

    async def worker(session, s):
        spec = SPECS[s]
        q = queues[s]
        reserve = (300 * spec["price"][0] + spec["max_tokens"] * spec["price"][1]) / 1e6
        while not q.empty() and not state["stop"]:
            s_, p, c, i = q.get_nowait()
            async with lock:
                if state["spent"] + state["reserved"] + reserve > a.budget_usd:
                    state["stop"] = True
                    return
                state["reserved"] += reserve
            payload = {"model": spec["model"],
                       "messages": [{"role": "user", "content": body(c, p)}],
                       "temperature": 1.0, "max_tokens": spec["max_tokens"],
                       "reasoning": spec["reasoning"], "usage": {"include": True}}
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
                except Exception as e:  # network errors are retried
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    status = None
                await asyncio.sleep(1 if status == 429 else min(30, 2 ** attempt + random.random()))
            content, usage, finish, rid = "", {}, None, None
            if status == 200 and data and not data.get("error"):
                ch = (data.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content") or ""
                finish = ch.get("finish_reason")
                usage = data.get("usage") or {}
                rid = data.get("id")
            cost = usage.get("cost")
            if status == 200 and cost is None:
                cost = reserve
            cost = float(cost or 0)
            row = {"spec": s, "model": spec["model"], "reasoning": spec["reasoning"],
                   "prompt": p, "cid": c["cid"], "n": c["n"], "expr": c["expr"],
                   "location": c["location"], "mention": c["mention"],
                   "mention_type": c["mention_type"], "sample_index": i,
                   "http_status": status, "error": err, "response_id": rid,
                   "finish_reason": finish, "raw": content, "usage": usage,
                   "cost_usd": cost, **parse(content)}
            async with lock:
                with a.out.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                state["reserved"] -= reserve
                state["spent"] += cost
                state["n"] += 1
                if status != 200 or not content:
                    state["fail"] += 1
                if state["n"] % 500 == 0:
                    print("PROGRESS", state["n"], "fail", state["fail"], "spent",
                          round(state["spent"], 3),
                          {k: v.qsize() for k, v in queues.items()}, flush=True)

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"}) as session:
        await asyncio.gather(*(worker(session, s) for s in a.specs
                               for _ in range(a.concurrency)))
    print("COMPLETE written", state["n"], "fail", state["fail"], "spent",
          round(state["spent"], 4), "budget_stop", state["stop"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
