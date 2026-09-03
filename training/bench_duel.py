"""What the duel environment costs, before any policy is attached.

Two numbers decide the shape of a training run and neither can be guessed:

* **The planner opponent.** `docs/DEVLOG.md` records 0.607 ms, but that is the
  *field rebuild*, not a decision. A decision is one rebuild plus eighteen
  sandboxed rollouts of up to 36 frames each, amortised over a commitment
  window. The ratio to a bare Laika frame is what sets `--mpc-fraction`.
* **The reset.** A duel draws a new maze every episode, and `setup_battle`
  builds one BFS distance grid per reachable cell — 120 of them on a 12x10.
  The range curriculum never paid this because its arena is fixed.

Run it before a long training session; the numbers move with the machine.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from duel_env import DuelVec


def throughput(count: int, mpc_fraction: float, steps: int, seed: int = 4242):
    env = DuelVec(count, seed, (1.0 - mpc_fraction, mpc_fraction, 0.0))
    rng = np.random.default_rng(0)
    actions = rng.integers(0, env.actions, size=(steps + 20, count)).astype(np.uint16)
    try:
        for t in range(20):  # warm up: page in the buffers, build the first fields
            env.step(actions[t])
            env.reset_done()

        resets = 0.0
        started = time.perf_counter()
        for t in range(steps):
            env.step(actions[t])
            mark = time.perf_counter()
            env.reset_done()
            resets += time.perf_counter() - mark
        elapsed = time.perf_counter() - started
    finally:
        env.close()

    return {
        "steps_per_second": count * steps / elapsed,
        "us_per_env_step": 1e6 * elapsed / (steps * count),
        "reset_share": resets / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    print(f"{args.envs} envs x {args.steps} steps\n")
    print(f"{'opponent mix':<22}{'steps/s':>12}{'us/env-step':>14}{'reset share':>14}")
    print("-" * 62)
    rows = {}
    for label, fraction in (("100% Laika", 0.0), ("50/50", 0.5), ("100% MPC", 1.0)):
        result = throughput(args.envs, fraction, args.steps)
        rows[label] = result
        print(f"{label:<22}{result['steps_per_second']:>12,.0f}"
              f"{result['us_per_env_step']:>14.1f}{result['reset_share']:>13.0%}")

    laika = rows["100% Laika"]["steps_per_second"]
    mixed = rows["50/50"]["steps_per_second"]
    print(f"\nthe planner costs {laika / max(mixed, 1e-9):.1f}x throughput at half strength")
    hours = 8
    print(f"overnight ({hours} h) at the 50/50 rate: "
          f"{mixed * 3600 * hours / 1e6:,.0f}M steps")


if __name__ == "__main__":
    main()
