import json, sys, numpy as np, collections
rng = np.random.default_rng(5)
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    by = collections.defaultdict(list)
    for r in rows:
        c = r["c"]
        full = r["q_digit"][c] - r["q_letter"][c]
        pat = r["q_patch"][c] - r["q_letter"][c]
        rnd = np.mean([q[c] for q in r["q_patch_random"]]) - r["q_letter"][c]
        by[c].append((full, pat, rnd, r["q_letter"][c], r["q_patch"][c], r["location"]))
    allr = [x for v in by.values() for x in v]
    def boot(i):
        ds = sorted(by)
        bs = [np.mean([x[i] for d in rng.choice(ds, len(ds)) for x in by[d]]) for _ in range(3000)]
        return np.mean([x[i] for x in allr]), np.percentile(bs, [2.5, 97.5])
    f, fc = boot(0); p, pc = boot(1); r_, rc = boot(2)
    fav = [x for x in allr if x[3] >= .05]
    print(f"{path.split('/')[-1]}: rows={len(rows)}")
    print(f"  full effect  (digit − letter)       {f:+.4f} [{fc[0]:+.4f},{fc[1]:+.4f}]")
    print(f"  patched heads (patch − letter)      {p:+.4f} [{pc[0]:+.4f},{pc[1]:+.4f}]  → {p/f:.0%} of full")
    print(f"  random heads  (patch − letter)      {r_:+.4f} [{rc[0]:+.4f},{rc[1]:+.4f}]  → {r_/f:.0%} of full")
    if fav:
        print(f"  favored cells (q_letter≥.05, n={len(fav)}): q_letter {np.mean([x[3] for x in fav]):.3f} → patched {np.mean([x[4] for x in fav]):.3f} (digit ctx {np.mean([x[3]+x[0] for x in fav]):.3f})")
    for loc in sorted({x[5] for x in allr}):
        xs = [x for x in allr if x[5] == loc]
        print(f"    {loc:13s} full {np.mean([x[0] for x in xs]):+.4f}  patch {np.mean([x[1] for x in xs]):+.4f}  random {np.mean([x[2] for x in xs]):+.4f}")
