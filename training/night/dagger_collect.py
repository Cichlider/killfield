from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel_env import ACTIONS, DuelVec, OPPONENT_FROZEN
from teacher_search import choose_search, load, logits


def run_game(seed, policy, opponent, device, horizon, gamma):
    env = DuelVec(1, seed, OPPONENT_FROZEN, 1)
    obs, masks, labels, student_actions, outcomes = [], [], [], [], []
    frames = 0
    while True:
        search = env.make_search(0, 0, ACTIONS, 16)
        teacher_action, opp_action, score, _src, _scores = choose_search(
            env, search, policy, opponent, device, horizon, gamma,
        )
        search.close()
        logits_out, _ = logits(
            policy, env.obs[0][None], env.masks[0][None].astype(bool), device,
        )
        student_action = int(logits_out.argmax(-1)[0])
        obs.append(env.obs[0].copy())
        masks.append(env.masks[0].astype(bool, copy=True))
        labels.append(teacher_action)
        student_actions.append(student_action)
        env.step(
            np.array([student_action], np.uint16),
            np.array([opp_action], np.uint16),
        )
        frames += 1
        if env.terminals[0]:
            outcomes.append(int(env.outcomes[0]))
            break
        env.reset_done()
        if frames > 900:
            outcomes.append(4)
            break
    env.close()
    return obs, masks, labels, student_actions, outcomes, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/night_run1/checkpoints/hybrid_night_final.pt"))
    parser.add_argument("--opponent", type=Path, default=Path(
        "outputs/night_run1/checkpoints/initial.pt"))
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    policy = load(args.checkpoint, device)
    opponent = load(args.opponent, device)
    all_obs, all_masks, all_labels, all_student, all_outcomes = [], [], [], [], []
    wins = 0
    for i in range(args.games):
        obs, masks, labels, student, outcomes, frames = run_game(
            args.seed + i, policy, opponent, device, args.horizon, args.gamma,
        )
        wins += outcomes[0] == 1
        all_obs.extend(obs)
        all_masks.extend(masks)
        all_labels.extend(labels)
        all_student.extend(student)
        all_outcomes.extend(outcomes * len(obs))
        print(f"game {i + 1}/{args.games} outcome={outcomes[0]} frames={frames} "
              f"teacher_labels={len(labels)}", flush=True)
    np.savez_compressed(
        args.save,
        obs=np.asarray(all_obs, np.float32),
        masks=np.asarray(all_masks, bool),
        action=np.asarray(all_labels, np.int64),
        student=np.asarray(all_student, np.int64),
        outcome=np.asarray(all_outcomes, np.uint8),
    )
    print(f"student winrate {wins}/{args.games}; wrote {args.save} "
          f"with {len(all_obs)} frames", flush=True)


if __name__ == "__main__":
    main()
