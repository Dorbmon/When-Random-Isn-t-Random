import json, sys, collections, numpy as np
rng = np.random.default_rng(2)
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    D = {(r["prompt"], r["location"], r["form"], r["mention"]): np.array(r["q_x"]) for r in rows}
    print("==", rows[0]["model"])
    for form, ctrl_form, ctrl in (("digit", "letter", "ABKQ"), ("word", "nato", ("alpha", "bravo", "kilo", "quebec"))):
        from itertools import product
        cells = []
        words = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        for p, loc in product(("base", "uniform"), ("comment_down", "assign_down", "prompt_nl")):
            base = np.mean([D[(p, loc, ctrl_form, m)] for m in ctrl], axis=0)
            for c in range(8):
                m = str(c) if form == "digit" else words[c]
                q = D[(p, loc, form, m)]
                cells.append((loc, c, q[c] - base[c], base[c], q[c]))
        d = np.array([x[2] for x in cells]); dig = np.array([x[1] for x in cells])
        bs = [np.mean(np.concatenate([d[dig == k] for k in rng.integers(0, 8, 8)])) for _ in range(5000)]
        fav = [x for x in cells if x[3] >= .05]
        rel = sum(x[4] for x in fav) / sum(x[3] for x in fav) - 1
        byloc = {l: round(np.mean([x[2] for x in cells if x[0] == l]), 4) for l in ("comment_down", "assign_down", "prompt_nl")}
        print(f"  {form:5s} vs {ctrl_form:6s}: Δx={d.mean():+.4f} CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] neg={np.mean(d<0):.2f}  favored rel.change={rel:+.3f} (n={len(fav)})  by loc {byloc}")
    # letter vs NATO controls themselves
    diff = [0.5*np.abs(np.mean([D[(p,l,'letter',m)] for m in 'ABKQ'],0)-np.mean([D[(p,l,'nato',m)] for m in ('alpha','bravo','kilo','quebec')],0)).sum() for p in ('base','uniform') for l in ('comment_down','assign_down','prompt_nl')]
    print("  TV(letter ctrl mean, NATO ctrl mean) avg", round(np.mean(diff),4))
