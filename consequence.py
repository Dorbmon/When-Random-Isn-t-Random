"""Exploratory: does a digit mention change the fidelity of the simulated OUTPUT distribution?
Output under Python execution of the model's draws: z = expr(x, y). Truth: uniform draws."""
import json, sys, collections, numpy as np
from repulsion_conditions import EXPRS
rng = np.random.default_rng(3)
def out_dist(q, e, n):
    z = np.zeros(10)
    for x in range(n):
        for y in range(n):
            z[int(eval(EXPRS[e], {"x": x, "y": y}))] += q[x][y]
    return z
for path in sys.argv[1:]:
    D = {}
    for l in open(path):
        r = json.loads(l)
        if r["location"] in ("base", "print_const"): continue
        D[(r["prompt"], r["n"], r["expr"], r["location"], r["mention"])] = np.array(r["q_xy"])
    cells = []
    for (p, n, e, loc, m), q in D.items():
        if not m.isdigit() or int(m) >= n: continue
        truth = out_dist(np.full((n, n), 1 / n**2), e, n)
        ctrl = [D[(p, n, e, loc, L)] for L in "ABKQ"]
        tv_m = 0.5 * np.abs(out_dist(q, e, n) - truth).sum()
        tv_c = np.mean([0.5 * np.abs(out_dist(c, e, n) - truth).sum() for c in ctrl])
        # input fidelity too
        u = np.full((n, n), 1 / n**2)
        itv_m = 0.5 * np.abs(q - u).sum(); itv_c = np.mean([0.5 * np.abs(c - u).sum() for c in ctrl])
        cells.append({"c": int(m), "d_out_tv": tv_m - tv_c, "d_in_tv": itv_m - itv_c, "tv_c": tv_c})
    by = collections.defaultdict(list)
    for c in cells: by[c["c"]].append(c)
    def boot(k):
        ds = sorted(by); bs = [np.mean([x[k] for d in rng.choice(ds, len(ds)) for x in by[d]]) for _ in range(3000)]
        return np.mean([c[k] for c in cells]), np.percentile(bs, [2.5, 97.5])
    mo, co = boot("d_out_tv"); mi, ci = boot("d_in_tv")
    print(f"{path.split('/')[-1]:28s} ctrl output TV={np.mean([c['tv_c'] for c in cells]):.3f}  Δ output TV={mo:+.4f} [{co[0]:+.4f},{co[1]:+.4f}]  Δ input TV={mi:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  cells={len(cells)}")
