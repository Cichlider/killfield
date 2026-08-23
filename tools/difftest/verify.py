"""Criteria 2 and 3: first-divergence frame, event-count equality, AI equality."""
import collections
import sys

FAIL = False


def first_div(a_path, b_path):
    a = open(a_path).read().split("\n")
    b = open(b_path).read().split("\n")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, len(a)
    return None, len(a)


def events(path):
    per = collections.defaultdict(collections.Counter)
    seed = None
    for line in open(path):
        if line.startswith("== seed"):
            seed = line.split()[2]
            continue
        for tok in line.split():
            if tok.startswith("E"):
                per[seed][tok[1:].split(",")[0]] += 1
    return per


def ai_divergence(a_path, b_path):
    """Count frames where the AI's chosen goal type differs."""
    a = open(a_path).read().split("\n")
    b = open(b_path).read().split("\n")
    n = 0
    for x, y in zip(a, b):
        ja = [t for t in x.split() if t.startswith("A:")]
        jb = [t for t in y.split() if t.startswith("A:")]
        if ja and jb and ja[0].split(",")[0] != jb[0].split(",")[0]:
            n += 1
    return n


for label, js, rs, min_frames in [
    ("engine", "js_game.txt", "rs_game.txt", 500),
    ("laika", "js_ai.txt", "rs_ai.txt", 500),
]:
    idx, total = first_div(js, rs)
    where = "no divergence" if idx is None else f"line {idx}"
    ok = idx is None or idx >= min_frames
    print(f"   {label}: first divergence {where} / {total}"
          f"  {'OK' if ok else 'FAIL (too early — likely a logic bug)'}")
    FAIL |= not ok

    ej, er = events(js), events(rs)
    same = ej == er
    print(f"   {label}: event counts {'match  OK' if same else 'DIFFER  FAIL'}")
    FAIL |= not same

n = ai_divergence("js_ai.txt", "rs_ai.txt")
print(f"   laika: AI goal-type mismatches {n}  {'OK' if n == 0 else 'FAIL'}")
FAIL |= n != 0

print("\n" + ("VERIFICATION FAILED" if FAIL else "ALL CRITERIA PASS"))
sys.exit(1 if FAIL else 0)
