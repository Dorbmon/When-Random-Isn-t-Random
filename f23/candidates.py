#!/usr/bin/env python3
"""F23 step 2: find candidate (snippet, draw, constant) triples in the collected files.

A candidate needs:
  - a draw `var = random.randint/randrange/choice(...)` with literal arguments whose legal values
    are 2-8 single digits (so at least 2 out-of-range digits remain as controls);
  - a snippet of at most 30 lines that runs on its own: the whole file, or the enclosing top-level
    function (no required parameters) plus `import random` and a call;
  - no input(), file, network or process access, and only random / math / time imports;
  - an integer literal 0-9 outside the draw call, outside random.seed, not in the test or iterator of
    any statement enclosing the draw, and not assigned to a name used by the draw or those tests.
Each candidate is then checked by execution (check_exec): for every variant of the constant, the first
random call executed must be the target draw with the same arguments, for seeds 0-19. Candidates
that pass are printed for manual review."""
import argparse
import ast
import io
import json
import re
import subprocess
import sys
from pathlib import Path

from pyflakes.api import check as flakes_check
from pyflakes.reporter import Reporter

HERE = Path(__file__).parent
FUNCS = {"randint", "randrange", "choice"}
BANNED = re.compile(r"\binput\s*\(|\bopen\s*\(|\bos\.|\bsys\.|\bsubprocess|\bsocket|\brequests|\burllib|\bexec\s*\(|\beval\s*\(|"
                    r"\bthreading|\btkinter|\bpygame|\bturtle|\bnumpy|\bseed\s*\(")
ALLOWED_IMPORTS = {"random", "math", "time"}


def lit_int(n):
    return isinstance(n, ast.Constant) and type(n.value) is int


def legal_values(call, fname):
    a = call.args
    if call.keywords or not all(lit_int(x) for x in a if not isinstance(x, (ast.List, ast.Tuple, ast.Call))):
        return None
    if fname == "randint" and len(a) == 2 and all(lit_int(x) for x in a):
        vals = list(range(a[0].value, a[1].value + 1))
    elif fname == "randrange" and 1 <= len(a) <= 3 and all(lit_int(x) for x in a):
        vals = list(range(*[x.value for x in a]))
    elif fname == "choice" and len(a) == 1:
        s = a[0]
        if isinstance(s, (ast.List, ast.Tuple)) and s.elts and all(lit_int(e) for e in s.elts):
            vals = [e.value for e in s.elts]
            if len(set(vals)) != len(vals):
                return None
        elif (isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id == "range"
              and 1 <= len(s.args) <= 3 and all(lit_int(x) for x in s.args)):
            vals = list(range(*[x.value for x in s.args]))
        else:
            return None
    else:
        return None
    if not vals or not all(0 <= v <= 9 for v in vals) or not (2 <= len(vals) <= 8):
        return None
    return vals


def draw_name(call):
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "random" and f.attr in FUNCS:
        return f.attr
    if isinstance(f, ast.Name) and f.id in FUNCS:
        return f.id
    return None


def parents(tree):
    P = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            P[ch] = node
    return P


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def undefined_names(src):
    buf = io.StringIO()
    flakes_check(src, "snippet", Reporter(buf, buf))
    return [l for l in buf.getvalue().splitlines() if "undefined name" in l or "syntax" in l.lower()]


def make_snippets(src, tree):
    """Yield (snippet_source, offset_lines, prefix) candidates: whole file or enclosing function."""
    lines = src.splitlines()
    if len([l for l in lines if l.strip()]) <= 30:
        yield src, 0, ""
    # top-level prefix: the file's first statements up to at most 30 lines, cut at a statement boundary,
    # containing the first random call (the harness stops there, so the rest need not be executed)
    ends = [n.end_lineno for n in tree.body]
    first_draw = min((n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and draw_name(n)), default=None)
    if first_draw is not None and len(lines) > 30:
        ok = [e for e in ends if e >= first_draw and e <= 30]
        if ok:
            cut = max(ok)
            yield "\n".join(lines[:cut]) + "\n", 0, ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            a = node.args
            if len(a.args) - len(a.defaults) > 0 or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
                continue
            body = "\n".join(lines[node.lineno - 1: node.end_lineno])
            if len([l for l in body.splitlines() if l.strip()]) > 28:
                continue
            prefix = "import random\n\n"
            yield prefix + body + f"\n\n{node.name}()\n", node.lineno - 1 - prefix.count("\n"), prefix


def analyse(snippet):
    """Return candidate dicts for one snippet source."""
    if BANNED.search(snippet):
        return []
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(al.name.split(".")[0] not in ALLOWED_IMPORTS for al in n.names):
            return []
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            return []
    if "import random" not in snippet and "from random import" not in snippet:
        return []
    if undefined_names(snippet):
        return []
    P = parents(tree)
    calls = sorted([n for n in ast.walk(tree) if isinstance(n, ast.Call) and draw_name(n)],
                   key=lambda n: (n.lineno, n.col_offset))
    out = []
    for call in calls[:1]:  # the first random call in source order is the target
        fn = draw_name(call)
        vals = legal_values(call, fn)
        st = P.get(call)
        if vals is None or not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
            continue
        var = st.targets[0].id
        # statements enclosing the draw and the names their tests use
        anc, node = [], st
        while node in P:
            node = P[node]; anc.append(node)
        control = []
        for a_ in anc:
            if isinstance(a_, (ast.If, ast.While)):
                control.append(a_.test)
            elif isinstance(a_, ast.For):
                control.append(a_.iter)
        banned_names = names_in(call) | set().union(*[names_in(c) for c in control]) if control else names_in(call)
        excluded = set()
        for c in [call] + control:
            excluded |= {id(n) for n in ast.walk(c)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "seed":
                excluded |= {id(m) for m in ast.walk(n)}
            # arguments of any other random call: changing them can make that call invalid
            if isinstance(n, ast.Call) and (draw_name(n) or (isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                                                              and n.func.value.id == "random")):
                excluded |= {id(m) for m in ast.walk(n)}
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in banned_names for t in n.targets):
                excluded |= {id(m) for m in ast.walk(n.value)}
            if isinstance(n, ast.arguments):  # default args that feed the draw
                for d in n.defaults:
                    excluded |= {id(m) for m in ast.walk(d)}
        consts = [n for n in ast.walk(tree) if lit_int(n) and 0 <= n.value <= 9 and id(n) not in excluded
                  and n.lineno == n.end_lineno]
        # keep the constant with a distinct token span; prefer NAME = d assignments, then nearest to the draw
        def rank(n):
            par = P.get(n)
            is_assign = isinstance(par, ast.Assign) and par.value is n
            return (0 if is_assign else 1, abs(n.lineno - call.lineno))
        consts.sort(key=rank)
        if not consts:
            continue
        out.append({"func": fn, "legal": vals, "var": var, "draw_line": call.lineno,
                    "draw_src": ast.get_source_segment(snippet, st),
                    "constants": [{"value": n.value, "line": n.lineno, "col": n.col_offset, "end_col": n.end_col_offset,
                                   "context": snippet.splitlines()[n.lineno - 1].strip()} for n in consts[:4]]})
    return out


def substitute(snippet, const, v):
    lines = snippet.splitlines()
    b = lines[const["line"] - 1].encode()  # ast column offsets are UTF-8 byte offsets
    assert b[const["col"]:const["end_col"]] == str(const["value"]).encode(), (lines[const["line"] - 1], const)
    lines[const["line"] - 1] = (b[:const["col"]] + str(v).encode() + b[const["end_col"]:]).decode()
    return "\n".join(lines) + "\n"


HARNESS = r'''
import random, sys, io, contextlib, json
_orig = {n: getattr(random, n) for n in ("randint", "randrange", "choice", "random", "uniform", "shuffle", "sample")}
class _Stop(BaseException): pass
_rec = []
def _wrap(n):
    def f(*a, **k):
        r = _orig[n](*a, **k)
        _rec.append([n, repr(a), sys._getframe(1).f_lineno, repr(r)])
        raise _Stop()
    return f
for _n in _orig: setattr(random, _n, _wrap(_n))
src = sys.stdin.read()
out = []
for seed in range(20):
    _orig_seed = random.seed; _orig_seed(seed); _rec.clear()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, "snippet", "exec"), {"__name__": "__main__"})
        out.append(None)
    except _Stop:
        out.append(_rec[0])
    except BaseException as e:
        out.append(["error", type(e).__name__])
print(json.dumps(out))
'''


def check_exec(snippet, cand, const):
    """All variants: first random call = target draw with same args, same results across variants."""
    ref = None
    vals = sorted(set(cand["legal"]) | set(oor_values(cand["legal"])))
    for v in vals:
        s = substitute(snippet, const, v)
        try:
            r = subprocess.run([sys.executable, "-c", HARNESS], input=s, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode:
            return False, r.stderr[-200:]
        rec = json.loads(r.stdout)
        if any(x is None or x[0] == "error" or x[2] != cand["draw_line"] or x[0] != cand["func"] for x in rec):
            return False, f"v={v}: {rec[0]}"
        if ref is None:
            ref = rec
        elif rec != ref:
            return False, f"v={v}: draw differs"
    return True, ""


def oor_values(legal):
    """Up to 3 out-of-range digits, taken from 0-9 closest above the range, then below."""
    lo, hi = min(legal), max(legal)
    cand = [d for d in range(hi + 1, 10)] + [d for d in range(lo - 1, -1, -1)]
    cand = [d for d in cand if d not in legal]
    return cand[:3]


def main():
    ap = argparse.ArgumentParser(description="Filter independently authorized source files in an isolated environment.")
    ap.add_argument("--raw", type=Path, required=True, help="Private source directory containing index.json")
    ap.add_argument("--out", type=Path, required=True, help="Private candidate JSON output")
    args = ap.parse_args()
    raw = args.raw
    idx = json.loads((raw / "index.json").read_text(encoding="utf-8"))
    meta = {v.get("file"): (u, v) for u, v in idx.items() if v.get("file")}
    from multiprocessing import Pool
    with Pool(12) as pool:
        per_file = pool.map(file_candidates, sorted(raw.glob("*.py")))
    found, seen_src = [], set()
    for fn, lst in zip(sorted(raw.glob("*.py")), per_file):
        for snip, cand, const in lst:
            key = re.sub(r"\s+", "", snip)
            if key in seen_src:
                continue
            u, v = meta.get(fn.name, (None, {}))
            if sum(f["repo"] == v.get("repo") for f in found) >= 2:  # at most 2 snippets per repository
                continue
            found.append({"id": f"s{len(found):03d}", "file": fn.name, "url": u, "repo": v.get("repo"),
                          "snippet": snip, **{k: cand[k] for k in ("func", "legal", "var", "draw_line", "draw_src")},
                          "constant": const, "oor": oor_values(cand["legal"])})
            seen_src.add(key)
            break  # one snippet per file
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    print(len(found), "candidates pass the static and execution checks")
    from collections import Counter
    print(Counter(len(c["legal"]) for c in found), Counter(c["func"] for c in found))


def file_candidates(fn):
    res = []
    try:
        src = fn.read_text(errors="strict")
        tree = ast.parse(src)
    except Exception:
        return res
    for snip, off, prefix in make_snippets(src, tree):
        for cand in analyse(snip):
            for const in cand["constants"]:
                try:
                    ok, why = check_exec(snip, cand, const)
                except Exception:
                    ok = False
                if ok:
                    res.append((snip, cand, const))
                    return res
    return res



if __name__ == "__main__":
    main()
