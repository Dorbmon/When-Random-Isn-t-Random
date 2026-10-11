"""F6 paired test: Δx(base) − Δx(instruct) on matched cells, digit-cluster bootstrap. Pairs given as base:instruct."""
import json, sys, collections, numpy as np
from analyze_repulsion import load_local, cells_from_dists, print_const_cells
rng = np.random.default_rng(11)
def cb(cells, key):
    by = collections.defaultdict(list)
    for c in cells: by[c["c"]].append(c[key])
    ds = sorted(by); s = np.array([sum(by[d]) for d in ds]); n = np.array([len(by[d]) for d in ds])
    idx = rng.integers(0, len(ds), (5000, len(ds))); bs = s[idx].sum(1) / n[idx].sum(1)
    return np.mean([c[key] for c in cells]), np.percentile(bs, [2.5, 97.5])
for pair in sys.argv[1:]:
    b, i = pair.split(":")
    Db, Di = load_local(b)[0], load_local(i)[0]
    cb_ = {(r["prompt"], r["n"], r["expr"], r["loc"], r["c"]): r for r in cells_from_dists(Db, "qx")}
    ci_ = {(r["prompt"], r["n"], r["expr"], r["loc"], r["c"]): r for r in cells_from_dists(Di, "qx")}
    cells = [{"c": k[-1], "d": cb_[k]["delta"] - ci_[k]["delta"]} for k in cb_ if k in ci_]
    pb = {(r["prompt"], r["n"], r["c"]): r for r in print_const_cells(Db) if "delta_oor" in r}
    pi = {(r["prompt"], r["n"], r["c"]): r for r in print_const_cells(Di) if "delta_oor" in r}
    pcs = [{"c": k[-1], "d": pb[k]["delta_oor"] - pi[k]["delta_oor"]} for k in pb if k in pi]
    m, ci = cb(cells, "d"); mp, cp = cb(pcs, "d")
    print(f"{b.split('/')[-1]:28s} − instruct: irrelevant Δx diff {m:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] (cells {len(cells)}) | print_const diff {mp:+.4f} [{cp[0]:+.4f},{cp[1]:+.4f}]")
