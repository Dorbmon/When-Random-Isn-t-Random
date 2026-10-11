#!/usr/bin/env python3
"""F23 step 4: build scoring items from the hand-reviewed candidates (review.json: {id: "keep" | reason})."""
import argparse
import json
from pathlib import Path

from candidates import substitute

HERE = Path(__file__).parent


def text(code, var):
    return ("Simulate exactly one execution of the Python 3 program. Pick a value for each random draw, "
            "then execute the remaining code. On the first line, write the value that the first random draw "
            f"assigns to {var}, in this format:\n{var}=<digit>\n"
            "Then write what the program prints. Do not explain.\n\n" + code)


def main():
    ap = argparse.ArgumentParser(description="Build private F23 prompts from independently authorized source snippets.")
    ap.add_argument("--candidates", type=Path, required=True, help="Private candidate JSON containing snippet bodies")
    ap.add_argument("--review", type=Path, default=HERE / "review.json")
    ap.add_argument("--out", type=Path, required=True, help="Private output JSONL; source prompts are not distributed")
    args = ap.parse_args()
    cands = {c["id"]: c for c in json.loads(args.candidates.read_text(encoding="utf-8"))}
    review = json.loads(args.review.read_text(encoding="utf-8"))
    keep = [cands[i] for i, v in review.items() if v == "keep"]
    items = []
    for c in keep:
        for v in sorted(set(c["legal"]) | set(c["oor"])):
            code = substitute(c["snippet"], c["constant"], v)
            items.append({"iid": f"f23|{c['id']}|{v}", "sid": c["id"], "c": v, "c0": c["constant"]["value"],
                          "legal": c["legal"], "oor": c["oor"], "func": c["func"], "var": c["var"],
                          "prefix": f"{c['var']}=", "text": text(code.rstrip("\n"), c["var"])})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    print(len(keep), "snippets,", len(items), "items")


if __name__ == "__main__":
    main()
