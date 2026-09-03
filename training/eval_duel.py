"""Deterministic evaluation of a published duel checkpoint.

Training reports the win rate of the *sampled* policy, which is the thing being
optimised but not the thing you watch — the viewer runs argmax. This runs argmax
and reports the two numbers separately when asked.

It also measures dithering, because "it twitches" is a claim that should be a
number before it becomes a design change. Three of them:

* **switch rate** — how often the chosen action differs from the previous one.
* **turn reversals** — left immediately after right, or the reverse. This is the
  one that costs: the hull turns 10 degrees a frame, so a reversal throws away
  the previous frame's rotation.
* **throttle reversals** — forward immediately after backup, same idea for
  position.

A reversal is not automatically a fault. Between two headings on the turn
lattice there is no "hold" action that splits the difference, so alternating is
the only way to sit between them, and a policy that has learned to do that on
purpose will look identical to one that is merely unstable. What separates them
is whether the reversals cluster while aiming or spray uniformly, so the report
splits reversals by whether the barrel was already on target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duel_env import OPPONENT_NAMES, OUTCOME_NAMES, DuelVec
from duel_ppo import ActorCritic

# duel_obs.rs layout: the aim-assist block starts here and its first channel is
# "this shot reaches the enemy".
AIM_SELF_OFFSET = 890


def load(run: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    manifest = json.loads((run / "live.json").read_text())
    weights = run / ("live.pt" if (run / "live.pt").exists() else "final.pt")
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    model = ActorCritic().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, manifest


@torch.inference_mode()
def evaluate(model, device, episodes: int, envs: int, weights,
             seed: int, sample: bool, frozen=None):
    env = DuelVec(envs, seed, weights)
    results = {name: {k: 0 for k in ("win", "loss", "double", "draw")}
               for name in OPPONENT_NAMES.values()}
    frames_seen = []

    previous = np.full(envs, -1, np.int64)
    switches = np.zeros(2, np.int64)          # [changed, total]
    turn_reversals = np.zeros(2, np.int64)    # [on target, off target]
    turn_total = 0
    throttle_reversals = 0
    steps = 0

    try:
        while sum(sum(r.values()) for r in results.values()) < episodes:
            obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
            mask = torch.as_tensor(env.masks.astype(bool), dtype=torch.bool, device=device)
            logits, _ = model(obs, mask)
            if sample:
                action = torch.distributions.Categorical(logits=logits).sample()
            else:
                action = logits.argmax(-1)
            action = action.cpu().numpy().astype(np.int64)

            # CANDIDATES index = throttle * 6 + turn * 2 + fire.
            throttle, turn = action // 6, (action // 2) % 3
            live = previous >= 0
            if live.any():
                prev_throttle = previous[live] // 6
                prev_turn = (previous[live] // 2) % 3
                switches += [int((action[live] != previous[live]).sum()), int(live.sum())]

                reversed_turn = (
                    ((turn[live] == 0) & (prev_turn == 2))
                    | ((turn[live] == 2) & (prev_turn == 0))
                )
                # Was the barrel already lined up when it reversed?
                on_target = env.obs[live, AIM_SELF_OFFSET] > 0.5
                turn_reversals[0] += int((reversed_turn & on_target).sum())
                turn_reversals[1] += int((reversed_turn & ~on_target).sum())
                turn_total += int(live.sum())

                throttle_reversals += int(
                    (((throttle[live] == 0) & (prev_throttle == 2))
                     | ((throttle[live] == 2) & (prev_throttle == 0))).sum()
                )

            # A frozen-checkpoint opponent is driven from here too, from its
            # own seat's observation. Sampled rather than argmax, so the pool
            # opponent is not a single deterministic script to memorise.
            opponent_action = None
            if frozen is not None and env.needs_action.any():
                theirs = torch.as_tensor(env.obs_opponent.copy(),
                                         dtype=torch.float32, device=device)
                their_mask = torch.as_tensor(env.masks_opponent.astype(bool),
                                             dtype=torch.bool, device=device)
                logits_o, _ = frozen(theirs, their_mask)
                opponent_action = (
                    torch.distributions.Categorical(logits=logits_o)
                    .sample().cpu().numpy().astype(np.uint16)
                )
            env.step(action.astype(np.uint16), opponent_action)
            steps += envs
            previous = action.copy()

            for i in np.flatnonzero(env.terminals):
                name = OPPONENT_NAMES[int(env.opponents[i])]
                results[name][OUTCOME_NAMES[int(env.outcomes[i])]] += 1
                frames_seen.append(int(env.frames[i]))
            previous[env.dones.astype(bool)] = -1
            env.reset_done()
    finally:
        env.close()

    return {
        "results": results,
        "frames_mean": float(np.mean(frames_seen)) if frames_seen else None,
        "switch_rate": switches[0] / max(switches[1], 1),
        "turn_reversal_rate": turn_reversals.sum() / max(turn_total, 1),
        "turn_reversal_on_target": int(turn_reversals[0]),
        "turn_reversal_off_target": int(turn_reversals[1]),
        "throttle_reversal_rate": throttle_reversals / max(turn_total, 1),
        "steps": steps,
    }


def report(label: str, r: dict):
    print(f"\n{label}")
    for name, row in r["results"].items():
        n = sum(row.values())
        if not n:
            continue
        print(f"  vs {name:<6} {n:>4} 局  "
              f"胜 {row['win'] / n:>6.1%}  负 {row['loss'] / n:>6.1%}  "
              f"双亡 {row['double'] / n:>6.1%}  平 {row['draw'] / n:>6.1%}")
    print(f"  平均局长 {r['frames_mean']:.0f} 帧")
    print(f"  动作切换率      {r['switch_rate']:>6.1%}  （每帧换一个动作的比例）")
    print(f"  转向反向率      {r['turn_reversal_rate']:>6.1%}  "
          f"（瞄准中 {r['turn_reversal_on_target']} / 未瞄准 {r['turn_reversal_off_target']}）")
    print(f"  油门反向率      {r['throttle_reversal_rate']:>6.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/ppo_duel_v2/s11"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--mix", type=float, nargs=3, default=(0.4, 0.4, 0.2),
                        metavar=("LAIKA", "MPC", "FROZEN"))
    parser.add_argument("--frozen", type=Path, default=None,
                        help="checkpoint driving the frozen slots of the pool")
    parser.add_argument("--seed", type=int, default=9_000)
    parser.add_argument("--both", action="store_true",
                        help="also evaluate the sampled policy for comparison")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, manifest = load(args.run, device)
    print(f"checkpoint: {manifest['steps']:,} 步 · update {manifest['update']} · "
          f"schema {manifest['schema_version']}")

    frozen = None
    mix = tuple(args.mix)
    if args.frozen is not None:
        payload = torch.load(args.frozen, map_location=device, weights_only=False)
        frozen = ActorCritic().to(device)
        frozen.load_state_dict(payload["model"])
        frozen.eval()
        print(f"对手池冻结档: {args.frozen}")
    elif mix[2] > 0:
        # A frozen slot with nobody driving it is a stationary target, which
        # would quietly inflate the win rate. Refuse rather than mislead.
        print("警告: --mix 里有 frozen 权重但没给 --frozen，改为 laika/mpc 对半")
        mix = (0.5, 0.5, 0.0)

    report("argmax（网页看到的就是这个）",
           evaluate(model, device, args.episodes, args.envs,
                    mix, args.seed, sample=False, frozen=frozen))
    if args.both:
        report("采样（训练时优化的那个）",
               evaluate(model, device, args.episodes, args.envs,
                        mix, args.seed, sample=True, frozen=frozen))


if __name__ == "__main__":
    main()
