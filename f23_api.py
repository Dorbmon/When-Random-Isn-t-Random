#!/usr/bin/env python3
"""F23/F24 API arm: q over digits read from top_logprobs at the first digit after `<var>=`.

Items are the same JSONL the local scorer uses ({"iid", "text", "prefix", ...}); prefix is "<var>=".
One request per (spec, item, repeat), temperature 1.0, one provider pinned, no fallbacks, no penalty
parameters, reasoning off. Resumable. The budget cap counts every API ledger in api_data/.
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

from api_logprobs import DIGIT, SPECS, digit_dist

HERE = Path(__file__).resolve().parent


def spent_all():
    s = 0.0
    for p in (HERE / "api_data").glob("*.jsonl"):
        for l in p.read_text().splitlines():
            try:
                s += float(json.loads(l).get("cost_usd") or 0)
            except Exception:
                pass
    return s


def read_var(lp, var):
    """Digit distribution at the first digit after `var=`; handles a merged '=d' token."""
    text = ""
    for tok in lp:
        t = tok.get("token", "")
        top = tok.get("top_logprobs") or []
        if re.search(rf"(?<![A-Za-z_]){re.escape(var)}\s*=\s*$", text) and DIGIT.match(t):
            d = digit_dist(top)
            return {"sampled": int(DIGIT.match(t).group(1)), "digit_probs": d, "digit_mass": sum(d)}
        if re.search(rf"(?<![A-Za-z_]){re.escape(var)}\s*$", text) and re.fullmatch(r"\s*=\s*\d", t):
            d = [0.0] * 10
            for x in top:
                m = re.fullmatch(r"\s*=\s*(\d)", x.get("token", ""))
                if m:
                    d[int(m.group(1))] += math.exp(x["logprob"])
            return {"sampled": int(t.strip()[-1]), "digit_probs": d, "digit_mass": sum(d), "merged": True}
        text += t
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True, choices=list(SPECS))
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget-usd", type=float, default=45.0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--rpm", type=float, default=19.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    items = [json.loads(l) for l in a.items.read_text().splitlines() if l.strip()]
    prior = [json.loads(l) for l in a.out.read_text().splitlines()] if a.out.exists() else []
    done = {(r["spec"], r["iid"], r["repeat"]) for r in prior if r.get("ok")}
    state = {"spent": spent_all(), "reserved": 0.0, "stop": False, "n": 0, "bad": 0}
    lock = asyncio.Lock()
    queues = {}
    for s in a.specs:
        q = asyncio.Queue()
        jobs = [(s, it, r) for r in range(a.repeats) for it in items if (s, it["iid"], r) not in done]
        for j in jobs[: a.limit or None]:
            q.put_nowait(j)
        queues[s] = q
    print("QUEUE", {s: q.qsize() for s, q in queues.items()}, "spent_all_arms", round(state["spent"], 3), flush=True)
    slots = {s: 0.0 for s in a.specs}
    slot_lock = {s: asyncio.Lock() for s in a.specs}

    async def pace(s):
        loop = asyncio.get_running_loop()
        async with slot_lock[s]:
            now = loop.time(); wait = max(0.0, slots[s] - now); slots[s] = max(now, slots[s]) + 60.0 / a.rpm
        await asyncio.sleep(wait)

    async def worker(session, s):
        spec = SPECS[s]; q = queues[s]
        reserve = (600 * spec["price"][0] + 60 * spec["price"][1]) / 1e6
        while not q.empty() and not state["stop"]:
            _, it, rep = q.get_nowait()
            async with lock:
                if state["spent"] + state["reserved"] + reserve > a.budget_usd:
                    state["stop"] = True
                    return
                state["reserved"] += reserve
            payload = {"model": spec["model"], "messages": [{"role": "user", "content": it["text"]}],
                       "temperature": 1.0, "logprobs": True, "top_logprobs": spec["top"], "max_tokens": 24,
                       "reasoning": spec["reasoning"], "provider": {"order": [spec["provider"]], "allow_fallbacks": False},
                       "usage": {"include": True}}
            status, data, err = None, None, None
            for attempt in range(6):
                await pace(s)
                try:
                    async with session.post("https://openrouter.ai/api/v1/chat/completions", json=payload) as r:
                        status = r.status; data = await r.json(content_type=None)
                    if status == 200 and not data.get("error"):
                        break
                    err = str(data.get("error"))[:300]
                except Exception as e:
                    err, status = f"{type(e).__name__}: {str(e)[:200]}", None
                await asyncio.sleep(1 if status == 429 else min(30, 2 ** attempt + random.random()))
            content, lp, usage = "", [], {}
            if status == 200 and data and not data.get("error"):
                ch = (data.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content") or ""
                lp = ((ch.get("logprobs") or {}).get("content")) or []
                usage = data.get("usage") or {}
            var = it["prefix"].rstrip("=")
            pos = read_var(lp, var)
            ok = bool(pos and pos["digit_mass"] > 0.5)
            cost = float(usage.get("cost") or (reserve if status == 200 else 0))
            row = {k: v for k, v in it.items() if k != "text"}
            row.update({"spec": s, "repeat": rep, "http_status": status, "error": err, "raw": content, "ok": ok,
                        "lp": pos, "usage": usage, "cost_usd": cost})
            async with lock:
                with a.out.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                state["reserved"] -= reserve; state["spent"] += cost; state["n"] += 1; state["bad"] += not ok
                if state["n"] % 200 == 0:
                    print("PROGRESS", state["n"], "not_ok", state["bad"], "spent_all_arms", round(state["spent"], 3),
                          {k: v.qsize() for k, v in queues.items()}, flush=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"}) as session:
        await asyncio.gather(*(worker(session, s) for s in a.specs for _ in range(a.concurrency)))
    print("COMPLETE written", state["n"], "not_ok", state["bad"], "spent_all_arms", round(state["spent"], 3),
          "budget_stop", state["stop"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
