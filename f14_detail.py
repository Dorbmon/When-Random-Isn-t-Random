import json, sys, numpy as np
for fn in sys.argv[1:]:
    rows=[json.loads(l) for l in open(fn)]
    D={(r['prompt'],r['location'],r['mention']):np.array(r['q']) for r in rows}
    base=D[('base','base',None)]; top=[int(i) for i in np.argsort(-base)[:3]]
    print("==", fn.split('/')[-1], "top values", {t: round(float(base[t]),3) for t in top})
    for t in top[:2]:
        a, b = t % 10, t // 10
        ms = [str(t), str(a), str(t+10) if t < 10 else str(20 + a), str(b) if t >= 10 else str(10 + a), "13" if t != 13 else "12"]
        for loc in ('assign_down','prompt_nl'):
            ctrl=np.mean([D[('base',loc,L)] for L in 'ABKQ'],0)
            print(f"   x={t:2d} {loc:12s} ctrl {ctrl[t]:.3f} | " + " ".join(f"'{m}':{D[('base',loc,m)][t]:.3f}" for m in ms))
