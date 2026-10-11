#!/usr/bin/env python3
"""F23 step 1: collect candidate Python files from GitHub code search (small-range draws)."""
import argparse
import json, subprocess, time, urllib.request, re
from pathlib import Path
Q = ["roll = random.randint(1, 6)", "dice = random.randint(1, 6)", "die = random.randint(1, 6)", "x = random.randint(1, 6)",
     "n = random.randint(1, 6)", "randint(1,6)", "randint(0,5)", "randint(1,3)", "randint(1,4)", "randint(1,5)", "randint(0,3)",
     "randint(0,2)", "randint(1,2)", "randint(0,1)", "randint(1,8)", "randrange(1,7)", "random.choice([1,2,3", "random.choice((1, 2, 3",
     "computer = random.randint(1, 3)", "num = random.randint(1, 5)", "number = random.randint(1, 5)", "guess = random.randint(1, 5)",
     "coin = random.randint(0, 1)", "flip = random.randint(0, 1)", "move = random.randint(1, 4)", "direction = random.randint(0, 3)",
     "damage = random.randint(1, 6)", "attack = random.randint(1, 6)", "r = random.randint(1, 6)", "d = random.randint(1, 6)"]


def main():
    ap = argparse.ArgumentParser(description="Collect source files only where you have permission to do so.")
    ap.add_argument("--out", type=Path, required=True, help="Private source directory; not part of the public dataset")
    args = ap.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {}
    idx = OUT / "index.json"
    if idx.exists():
        seen = json.load(open(idx))
    for q in Q:
        for attempt in range(3):
            r = subprocess.run(["gh", "search", "code", q, "--language", "python", "--limit", "100", "--json", "path,repository,url"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            time.sleep(30)
        try:
            hits = json.loads(r.stdout)
        except Exception:
            print("fail", q, r.stderr[:200]); continue
        new = 0
        for h in hits:
            m = re.match(r"https://github.com/([^/]+/[^/]+)/blob/([0-9a-f]{40})/(.+)", h["url"])
            if not m or h["url"] in seen:
                continue
            seen[h["url"]] = {"repo": m.group(1), "sha": m.group(2), "path": m.group(3), "query": q}; new += 1
        print(f"{q!r}: {len(hits)} hits, {new} new, total {len(seen)}", flush=True)
        json.dump(seen, open(idx, "w"), indent=0)
        time.sleep(7)
    # fetch contents
    for i, (u, v) in enumerate(seen.items()):
        fn = OUT / (re.sub(r"[^A-Za-z0-9]+", "_", v["repo"] + "_" + v["path"])[:150] + ".py")
        if fn.exists():
            continue
        raw = f"https://raw.githubusercontent.com/{v['repo']}/{v['sha']}/{v['path']}"
        try:
            data = urllib.request.urlopen(raw, timeout=20).read()
            if len(data) < 60000:
                fn.write_bytes(data)
                v["file"] = fn.name
        except Exception as e:
            print("fetch fail", raw, e)
    json.dump(seen, open(idx, "w"), indent=0)
    print("files", len(list(OUT.glob("*.py"))))


if __name__ == "__main__":
    main()
