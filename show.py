import json,sys
R=json.load(open(sys.argv[1]))
for name,r in R["local"].items():
    print("=====",name, "excluded",r["excluded_low_legal_mass"])
    for k in ['primary_delta_x','secondary_logratio_x','delta_y','delta_either','diagonal_contrast_x','print_const_oor','print_const_oor_logratio','print_const_vs_base']:
        b=r[k]; print(f"  {k:26s} {b['mean']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] neg={b['frac_negative']:.2f}")
    print("  letter_noise mean|.|", r["letter_noise"]["mean_abs"], " |Δ|/noise", round(r.get("abs_delta_over_letter_noise",0),1))
    for k in ['by_location','by_location_logratio','by_n','by_prompt','by_expr']:
        print(" ",k, "  ".join(f"{a}:{b['mean']:+.3f}[{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]" for a,b in r[k].items()))
    import numpy as np
    cb=r["collapse_base"]; print("  collapse: normH", round(np.mean([c["norm_entropy_x"] for c in cb]),3), "maxmass", round(np.mean([c["max_mass_x"] for c in cb]),3), "argmax", sorted({c["argmax_x"] for c in cb}), "P(x=y)", round(np.mean([c["p_x_eq_y"] for c in cb]),4))
