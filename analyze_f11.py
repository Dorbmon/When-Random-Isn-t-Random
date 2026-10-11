import json, sys, numpy as np
from f10_categorical import ALL_SETS
for fn in sys.argv[1:]:
    rows=[json.loads(l) for l in open(fn)]
    D={(r["prompt"],r["set"],r["location"],r["mention"]):np.array(r["q"]) for r in rows}
    sets=sorted({r["set"] for r in rows})
    print("==", fn.split("/")[-1], "first-token mass", round(np.mean([r["first_token_mass"] for r in rows]),3))
    for s in sets:
        opts,ctrls=ALL_SETS[s]
        out=[]
        for loc in ("assign_down","comment_down","prompt_nl"):
            cells=[]
            for p in ("base","uniform"):
                if any((p,s,loc,c) not in D for c in ctrls): continue
                base=np.mean([D[(p,s,loc,c)] for c in ctrls],0)
                for i,o in enumerate(opts):
                    if (p,s,loc,o) in D: cells.append((D[(p,s,loc,o)][i]-base[i], base[i], D[(p,s,loc,o)][i]))
            fav=[c for c in cells if c[1]>=.05]
            rel=sum(c[2] for c in fav)/sum(c[1] for c in fav)-1 if fav else float('nan')
            out.append(f"{loc[:6]} Δ={np.mean([c[0] for c in cells]):+.3f} fav={rel:+.2f} n={len(cells)}" if cells else f"{loc[:6]} n=0")
        print(f"   {s:18s}", " | ".join(out))
