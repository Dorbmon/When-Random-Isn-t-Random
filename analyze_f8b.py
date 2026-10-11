import json, sys, numpy as np
rows=[json.loads(l) for l in open(sys.argv[1])]
LET="ABKQ"
R={(r["location"],r["mention"]):r for r in rows}
def delta(get):
    irr=[];pc=[]
    for loc in ("assign_down","comment_down","prompt_nl","varname_down"):
        ctrl=np.mean([get(R[(loc,L)]) for L in LET],0)
        for c in range(8): irr.append(get(R[(loc,str(c))])[c]-ctrl[c])
    oor=np.mean([get(R[("print_const",str(w))]) for w in (8,9)],0)
    for c in range(8): pc.append(get(R[("print_const",str(c))])[c]-oor[c])
    return np.array([np.mean(irr), np.mean(pc)])
b=delta(lambda r: np.array(r["q_none"])); print(sys.argv[1], "none Δ irr/print", np.round(b,4))
for k in rows[0]["joint"]:
    d=delta(lambda r: np.array(r["joint"][k])); print(" ",k, "Δ", np.round(d,4), "recovery", np.round(1-d/b,3))
print("  head  cand | attn→mention digit/letter | DLA(c) − mean others: digit ctx | letter ctx")
for h in rows[0]["heads"]:
    ad=[];al=[];dd=[];dl=[]
    for r in rows:
        H=r["heads"][h]; v=np.array(H["dla"])
        if r["mention"] and r["mention"].isdigit() and int(r["mention"])<8:
            c=int(r["mention"]); ad.append(H["attn_mention"]); dd.append(v[c]-np.mean(np.delete(v,c)))
        elif r["mention"] in LET:
            al.append(H["attn_mention"]); dl.append(np.mean([v[c]-np.mean(np.delete(v,c)) for c in range(8)]))
    print(f"  {h:8s} {str(rows[0]['heads'][h]['candidate']):5s} | {np.mean(ad):.3f} / {np.mean(al):.3f} | {np.mean(dd):+.3f} | {np.mean(dl):+.3f}")
