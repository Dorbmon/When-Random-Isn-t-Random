"""Condition grid for the pre-registered context-repulsion experiment."""
import json

NS = (6, 8, 10)
EXPRS = {"e1": "max(x, y)", "e2": "abs(x - y)"}
LETTERS = ("A", "B", "K", "Q")
DIGITS = tuple(str(d) for d in range(10))
LOCATIONS = ("comment_down", "assign_down", "varname_down", "comment_up", "prompt_nl")
UNIFORM = "Each random draw must be independent and uniformly sampled from its possible values."


def draws(n):
    return f"x = random.randrange({n})\ny = random.randrange({n})\n"


def located(location, n, expr, m):
    """Return (code, nl_prefix) for mention m at a location."""
    d = draws(n)
    if location == "comment_down":
        return d + f"print({expr})  # case {m}", ""
    if location == "assign_down":
        return d + f"tag = '{m}'\nprint({expr})", ""
    if location == "varname_down":
        return d + f"z{m} = {expr}\nprint(z{m})", ""
    if location == "comment_up":
        return f"# case {m}\n" + d + f"print({expr})", ""
    if location == "prompt_nl":
        return d + f"print({expr})", f"This is test case {m}."
    raise ValueError(location)


def conditions(ns=NS, exprs=tuple(EXPRS), locations=LOCATIONS, include_print=True,
               include_base=True, mentions=DIGITS + LETTERS):
    out = []
    for n in ns:
        if include_print:
            for c in range(10):
                out.append({"n": n, "expr": None, "location": "print_const",
                            "mention": str(c), "mention_type": "digit",
                            "code": draws(n) + f"print({c})", "nl": ""})
        for e in exprs:
            if include_base:
                out.append({"n": n, "expr": e, "location": "base", "mention": None,
                            "mention_type": "none",
                            "code": draws(n) + f"print({EXPRS[e]})", "nl": ""})
            for loc in locations:
                for m in mentions:
                    code, nl = located(loc, n, EXPRS[e], m)
                    out.append({"n": n, "expr": e, "location": loc, "mention": m,
                                "mention_type": "digit" if m.isdigit() else "letter",
                                "code": code, "nl": nl})
    for c in out:
        c["cid"] = f"n{c['n']}|{c['expr']}|{c['location']}|{c['mention']}"
    assert len({c["cid"] for c in out}) == len(out)
    assert len({(c["code"], c["nl"]) for c in out}) == len(out)
    return out


def cot_prompt(code):
    # Identical wording to natural_factorization.cot_prompt.
    return ("Simulate exactly one execution of the Python 3 program. Pick a value for each random draw, "
            "then execute the remaining code. Use exactly three lines in this format:\n"
            "x=<digit>\ny=<digit>\nanswer=<digit>\n"
            "Do not explain or add any other text.\n\nimport random\n" + code)


def body(cond, prompt):
    text = cot_prompt(cond["code"])
    if cond["nl"]:
        text = cond["nl"] + "\n" + text
    if prompt == "uniform":
        text = UNIFORM + "\n" + text
    return text


def truth_answer(cond, x, y):
    if cond["location"] == "print_const":
        return int(cond["mention"])
    return int(eval(EXPRS[cond["expr"]], {"x": x, "y": y}))


if __name__ == "__main__":
    cs = conditions()
    print(len(cs), "conditions per prompt;", 2 * len(cs), "contexts per model")
    for c in cs[:3] + [c for c in cs if c["location"] == "prompt_nl"][:1]:
        print("----", c["cid"])
        print(body(c, "uniform"))
    json.dump(cs, open("conditions.json", "w"), indent=1)
