"""Convert API logprob F4/F5 rows to the local grid format (q_x averaged over repeats), one file per spec."""
import json, sys, collections, numpy as np
from pathlib import Path
src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
acc = collections.defaultdict(list); meta = {}
for l in src.read_text().splitlines():
    r = json.loads(l)
    if not r.get("ok"): continue
    p = np.asarray(r["x_lp"]["digit_probs"][:r["n"]])
    if p.sum() <= 0: continue
    k = (r["spec"], r["prompt"], r["cid"]); acc[k].append(p / p.sum()); meta[k] = r
files = collections.defaultdict(list)
for k, v in acc.items():
    r = meta[k]
    files[r["spec"]].append({"model": r["spec"], "prompt": r["prompt"], "cid": r["cid"], "family": r["family"],
                             "level": r["level"], "mention": r["mention"], "n": r["n"], "q_x": np.mean(v, 0).tolist(),
                             "legal_mass_x": 1.0, "repeats": len(v)})
for spec, rows in files.items():
    (outdir / f"{src.stem.split('_')[1]}_lp_{spec}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(spec, len(rows))
