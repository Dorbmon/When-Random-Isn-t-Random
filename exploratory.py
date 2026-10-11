"""Exploratory (not pre-registered): relative change split by control probability."""
import json, sys, numpy as np, collections
cells = [json.loads(l) for l in open(sys.argv[1])]
models = list(dict.fromkeys(r["model"] for r in cells))
print(f"{'model':28s} {'favored cells':>13s} {'rel.change favored':>18s} {'neg':>5s} | {'rare cells':>10s} {'mean q ctrl→mention (rare)':>28s}")
for m in models:
    cs = [r for r in cells if r["model"] == m]
    fav = [r for r in cs if r["q_ctrl"] >= 0.05]
    rare = [r for r in cs if r["q_ctrl"] < 0.01]
    rel = sum(r["q_mention"] for r in fav) / sum(r["q_ctrl"] for r in fav) - 1 if fav else float("nan")
    negf = np.mean([r["q_mention"] < r["q_ctrl"] for r in fav]) if fav else float("nan")
    qr = (np.mean([r["q_ctrl"] for r in rare]), np.mean([r["q_mention"] for r in rare])) if rare else (np.nan, np.nan)
    print(f"{m:28s} {len(fav):13d} {rel:+18.3f} {negf:5.2f} | {len(rare):10d} {qr[0]:12.4f} → {qr[1]:.4f}")
    # by location, favored only
    bl = collections.defaultdict(list)
    for r in fav: bl[r["loc"]].append(r)
    print("    favored rel.change by location:", {l: round(sum(r['q_mention'] for r in v)/sum(r['q_ctrl'] for r in v)-1, 3) for l, v in sorted(bl.items())})

print("\nDecomposition Δ = a + b·(q_ctrl − 1/n): a = net repulsion, b = flattening toward uniform")
rng = np.random.default_rng(1)
for m in models:
    cs = [r for r in cells if r["model"] == m]
    X = np.array([r["q_ctrl"] - 1 / r["n"] for r in cs]); Y = np.array([r["delta"] for r in cs])
    dig = np.array([r["c"] for r in cs])
    def fit(idx):
        A = np.c_[np.ones(len(idx)), X[idx]]; return np.linalg.lstsq(A, Y[idx], rcond=None)[0]
    a, b = fit(np.arange(len(cs)))
    groups = {d: np.where(dig == d)[0] for d in np.unique(dig)}
    bs = np.array([fit(np.concatenate([groups[d] for d in rng.choice(list(groups), len(groups))])) for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"  {m:28s} a={a:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]   b={b:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]")
