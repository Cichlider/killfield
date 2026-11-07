from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel_env import ACTIONS, DuelVec, OPPONENT_FROZEN
from hybrid_model import HybridActorCritic


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def logits(model, obs, masks, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
    return model(obs_t, mask_t)


@torch.no_grad()
def choose_search(env, search, policy, opponent, device, horizon, gamma, seat=0,
                  depth=1, reward_mode=False):
    source_obs = (env.obs[0] if seat == 0 else env.obs_opponent[0]).copy()
    source_mask = (env.masks[0] if seat == 0 else env.masks_opponent[0]).astype(bool, copy=True)
    source_opp_obs = (env.obs_opponent[0] if seat == 0 else env.obs[0]).copy()
    source_opp_mask = (env.masks_opponent[0] if seat == 0 else env.masks[0]).astype(bool, copy=True)
    _, source_value = logits(
        policy, source_obs[None], source_mask[None], device,
    )
    opp_logits, _ = logits(opponent, source_opp_obs[None], source_opp_mask[None], device)
    opp_action = int(opp_logits.argmax(-1)[0])
    branches = search.branches
    scores = np.zeros(branches, np.float32)
    discount = np.zeros(branches, np.float32)
    actions = np.arange(branches, dtype=np.uint16)
    learner_root = (actions // (ACTIONS ** (depth - 1))) % ACTIONS
    learner_second = actions % ACTIONS
    opp_hist = np.zeros(branches, np.uint16)
    opp_hist[:] = opp_action
    for step in range(horizon):
        alive_before = search.dones == 0
        s_obs = torch.as_tensor(search.obs, dtype=torch.float32, device=device)
        s_mask = torch.as_tensor(search.masks.astype(bool, copy=False), device=device)
        s_logits, s_value = policy(s_obs, s_mask)
        s_opp_obs = torch.as_tensor(search.obs_opponent, dtype=torch.float32, device=device)
        s_opp_mask = torch.as_tensor(
            search.masks_opponent.astype(bool, copy=False), device=device,
        )
        s_opp_logits, _ = opponent(s_opp_obs, s_opp_mask)
        if step == 0:
            learner_actions = learner_root
        elif step == 1 and depth == 2:
            learner_actions = learner_second
        else:
            learner_actions = s_logits.argmax(-1).cpu().numpy().astype(np.uint16)
        opponent_actions = s_opp_logits.argmax(-1).cpu().numpy().astype(np.uint16)
        if step < depth:
            opponent_actions[:] = opp_action
        discount += (gamma ** step) * alive_before
        search.step(learner_actions, opponent_actions)
        if reward_mode:
            scores += (gamma ** step) * search.rewards * alive_before.astype(np.float32)
        done = search.dones.astype(bool)
        if done.any():
            terminal = search.outcomes
            for i in np.flatnonzero(done):
                if discount[i] < 0.5:  # branch finished on an earlier step
                    continue
                if reward_mode:
                    # engine rewards already price outcome, timing win-decay,
                    # trade and draw; adding a bonus would double count
                    pass
                else:
                    scores[i] = 1.0 if terminal[i] == 1 else -1.0
            break
    live = ~search.dones.astype(bool)
    if live.any():
        live_obs = torch.as_tensor(search.obs[live], dtype=torch.float32, device=device)
        live_mask = torch.as_tensor(search.masks[live].astype(bool, copy=False), device=device)
        _, live_value = policy(live_obs, live_mask)
        leaf = live_value.cpu().numpy() * (gamma ** horizon)
        if reward_mode:
            scores[live] += leaf
        else:
            scores[live] = leaf
    # Already-terminal branches keep their terminal score.
    return int(np.argmax(scores)), opp_action, float(scores.max()), float(source_value[0]), scores.copy()


def run_game(seed, policy, opponent, device, horizon, gamma, seat=0, trajectory=None,
             depth=1, threads=18, reward_mode=False):
    env = DuelVec(1, seed, OPPONENT_FROZEN, 1)
    branches = ACTIONS ** max(1, depth)
    frames = 0
    teacher_win = False
    while True:
        search = env.make_search(0, seat, branches, threads)
        action, opp_action, score, _, scores = choose_search(
            env, search, policy, opponent, device, horizon, gamma, seat,
            depth, reward_mode,
        )
        search.close()
        if trajectory is not None:
            trajectory["obs"].append(
                (env.obs[0] if seat == 0 else env.obs_opponent[0]).copy()
            )
            trajectory["masks"].append(
                (env.masks[0] if seat == 0 else env.masks_opponent[0]).astype(bool, copy=True)
            )
            trajectory["action"].append(action)
            trajectory["q"].append(scores)
            trajectory["seat"].append(seat)
        if seat == 0:
            env.step(np.array([action], np.uint16), np.array([opp_action], np.uint16))
        else:
            env.step(np.array([opp_action], np.uint16), np.array([action], np.uint16))
        frames += 1
        if env.terminals[0]:
            teacher_win = int(env.outcomes[0]) == (1 if seat == 0 else 2)
            break
        env.reset_done()
        if frames > 900:
            break
    outcome = int(env.outcomes[0])
    env.close()
    if trajectory is not None:
        trajectory["outcome"].append(outcome)
    return teacher_win, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/night_run1/checkpoints/hybrid_night_final.pt"))
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--opponent", type=Path, default=Path(
        "outputs/night_run1/checkpoints/initial.pt"))
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--depth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--reward-mode", action="store_true",
                        help="score branches with discounted engine rewards plus value leaf")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    policy = load(args.checkpoint, device)
    opponent = load(args.opponent, device)
    wins = 0
    all_obs, all_masks, all_actions, all_q, all_outcomes, all_seeds, all_seats = (
        [], [], [], [], [], [], []
    )
    for i in range(args.games):
        trajectory = {
            "obs": [], "masks": [], "action": [], "q": [], "outcome": [], "seat": [],
        }
        win, frames = run_game(
            args.seed + i, policy, opponent, device, args.horizon, args.gamma,
            args.seat, trajectory, args.depth, args.threads, args.reward_mode,
        )
        wins += win
        if args.save is not None:
            all_obs.extend(trajectory["obs"])
            all_masks.extend(trajectory["masks"])
            all_actions.extend(trajectory["action"])
            all_q.extend(trajectory["q"])
            all_outcomes.extend(trajectory["outcome"] * len(trajectory["obs"]))
            all_seeds.extend([args.seed + i] * len(trajectory["obs"]))
            all_seats.extend(trajectory["seat"])
        print(f"game {i + 1}/{args.games} win={win} frames={frames}", flush=True)
    print(f"teacher winrate {wins}/{args.games} = {wins / max(args.games, 1):.3f}")
    if args.save is not None:
        np.savez_compressed(
            args.save,
            obs=np.asarray(all_obs, np.float32),
            masks=np.asarray(all_masks, bool),
            action=np.asarray(all_actions, np.int64),
            q=np.asarray(all_q, np.float32),
            outcome=np.asarray(all_outcomes, np.uint8),
            seed=np.asarray(all_seeds, np.uint32),
            seat=np.asarray(all_seats, np.uint8),
        )
        print(f"wrote {args.save} with {len(all_obs)} frames")


if __name__ == "__main__":
    main()
