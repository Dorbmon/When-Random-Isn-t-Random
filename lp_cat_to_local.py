"""API categorical logprob rows → local F10/F11/F12-format files, one per spec (repeats averaged)."""
import json, sys, collections, numpy as np
from pathlib import Path
from f10_categorical import ALL_SETS
src, tag = Path(sys.argv[1]), sys.argv[2]
acc = collections.defaultdict(list)
for l in src.read_text().splitlines():
    r = json.loads(l)
    if not r.get("ok") or not r.get("v_top"): continue
    opts = ALL_SETS[r["set"]][0]; q = np.zeros(len(opts))
    for t, p in r["v_top"].items():
        ts = t.strip().lower()
        if not ts: continue
        hits = [i for i, o in enumerate(opts) if o.lower().startswith(ts)]
        if len(hits) == 1: q[hits[0]] += p
    if q.sum() > .3: acc[(r["spec"], r["prompt"], r["set"], r["location"], r["mention"])].append((q / q.sum(), q.sum()))
out = collections.defaultdict(list)
for (spec, p, s, loc, m), v in acc.items():
    out[spec].append({"model": spec, "prompt": p, "set": s, "location": loc, "mention": m,
                      "q": np.mean([a for a, _ in v], 0).tolist(), "first_token_mass": float(np.mean([b for _, b in v]))})
for spec, rows in out.items():
    Path(f"data/{tag}_lp_{spec}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print({k: len(v) for k, v in out.items()})
