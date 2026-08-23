"""MPC criterion: the chosen action and decision kind must never differ.

Score vectors carry the engine's 1-ULP cos/sin noise, so they are compared as
numbers with a tolerance rather than as bits. What must hold exactly is that
the noise never flips an argmax.
"""
import collections
import struct
import sys


def tok(line, pre):
    for t in line.split():
        if t.startswith(pre):
            return t
    return None


a = open("js_mpc.txt").read().split("\n")
b = open("rs_mpc.txt").read().split("\n")

act_diff = kind_diff = frames = 0
max_rel = 0.0
for x, y in zip(a, b):
    if x.startswith("==") or not x.strip():
        continue
    frames += 1
    if tok(x, "act") != tok(y, "act"):
        act_diff += 1
    if tok(x, "k:") != tok(y, "k:"):
        kind_diff += 1
    vx, vy = tok(x, "V"), tok(y, "V")
    if vx and vy and vx != vy and vx != "V-" and vy != "V-":
        for u, v in zip(vx[1:].split(","), vy[1:].split(",")):
            if u == v or u == "NaN" or v == "NaN":
                continue
            fu = struct.unpack(">d", bytes.fromhex(u))[0]
            fv = struct.unpack(">d", bytes.fromhex(v))[0]
            max_rel = max(max_rel, abs(fu - fv) / max(abs(fu), abs(fv), 1e-300))


def events(path):
    per = collections.defaultdict(collections.Counter)
    seed = None
    for line in open(path):
        if line.startswith("== seed"):
            seed = line.split()[2]
            continue
        for t in line.split():
            if t.startswith("E"):
                per[seed][t[1:].split(",")[0]] += 1
    return per


ev_same = events("js_mpc.txt") == events("rs_mpc.txt")
tol_ok = max_rel < 1e-9

print(f"   mpc: chosen action differs on {act_diff}/{frames} frames"
      f"  {'OK' if act_diff == 0 else 'FAIL'}")
print(f"   mpc: decision kind differs on {kind_diff}/{frames} frames"
      f"  {'OK' if kind_diff == 0 else 'FAIL'}")
print(f"   mpc: max relative score error {max_rel:.2e}"
      f"  {'OK' if tol_ok else 'FAIL (beyond trig noise)'}")
print(f"   mpc: event counts {'match  OK' if ev_same else 'DIFFER  FAIL'}")

bad = act_diff or kind_diff or not tol_ok or not ev_same
print("\n" + ("MPC VERIFICATION FAILED" if bad else "MPC CRITERIA PASS"))
sys.exit(1 if bad else 0)
