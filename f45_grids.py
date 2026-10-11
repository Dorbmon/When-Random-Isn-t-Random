"""F4 (distance, dose) and F5 (realistic constants) grids. Each context gives code, nl and n."""
from repulsion_conditions import UNIFORM, cot_prompt, draws

N = 8
LETTERS = ("A", "B", "K", "Q")
MENTIONS = tuple(str(d) for d in range(10)) + LETTERS
FILLER = ["label = 'demo'", "mode = 'fast'", "verbose = False", "name = 'trial'",
          "debug = True", "unit = 'cm'", "color = 'red'", "shape = 'box'",
          "owner = 'lab'", "kind = 'test'"]


def lit(m):
    return f"'{m}'"


def f4():
    out = []
    # Distance: the tag line sits above the draws, with k filler lines in between.
    for k in (0, 2, 5, 10):
        for m in MENTIONS:
            code = f"tag = {lit(m)}\n" + "".join(l + "\n" for l in FILLER[:k]) + draws(N) + "print(max(x, y))"
            out.append({"cid": f"dist{k:02d}|{m}", "family": "distance", "level": k, "mention": m,
                        "n": N, "code": code, "nl": ""})
    # Filler below the draws: the tag line is immediately after the draws, followed by k filler lines.
    for k in (0, 5, 10):
        for m in MENTIONS:
            code = draws(N) + f"tag = {lit(m)}\n" + "".join(l + "\n" for l in FILLER[:k]) + "print(max(x, y))"
            out.append({"cid": f"below{k:02d}|{m}", "family": "below", "level": k, "mention": m,
                        "n": N, "code": code, "nl": ""})
    # Dose: 1, 2 or 3 mentions of the same symbol at different spots.
    for k in (1, 2, 3):
        for m in MENTIONS:
            code = draws(N) + f"tag = {lit(m)}\n" + "print(max(x, y))" + (f"  # case {m}" if k >= 2 else "")
            nl = f"This is test case {m}." if k >= 3 else ""
            out.append({"cid": f"dose{k}|{m}", "family": "dose", "level": k, "mention": m,
                        "n": N, "code": code, "nl": nl})
    return out


REALISTIC = {
    "retries": ("MAX_RETRIES = {d}\n\n", "def attempt():\n    return max(x, y)\n\nprint(attempt())"),
    "timeout": ("TIMEOUT = {d}  # seconds\n", "print(max(x, y))"),
    "default_arg": ("", "def score(a, b, bonus={d}):\n    return max(a, b)\n\nprint(score(x, y))"),
    "config_dict": ("config = {{'workers': {d}, 'mode': 'fast'}}\n", "print(max(x, y))"),
    "loop_bound": ("", "for _ in range({d}):\n    pass\nprint(max(x, y))"),
}


def f5():
    """Realistic integer constants. Controls: out-of-range digits (8, 9) and the diagonal contrast."""
    out = []
    for name, (pre, post) in REALISTIC.items():
        for d in range(10):
            code = pre.format(d=d) + draws(N) + post.format(d=d)
            out.append({"cid": f"real_{name}|{d}", "family": "realistic", "level": name,
                        "mention": str(d), "n": N, "code": code, "nl": ""})
    return out


def main_grid():
    """The full pre-registered 912-context grid (456 per prompt)."""
    from repulsion_conditions import conditions
    return conditions()


GRIDS = {"f4": f4, "f5": f5, "main": main_grid}


def text_for(c, prompt):
    t = cot_prompt(c["code"])
    if c.get("nl"):
        t = c["nl"] + "\n" + t
    return (UNIFORM + "\n" + t) if prompt == "uniform" else t

if __name__ == "__main__":
    for g in GRIDS.values():
        cs = g()
        assert len({c["cid"] for c in cs}) == len(cs)
        print(len(cs))
        print(cs[len(cs) // 2]["code"])
