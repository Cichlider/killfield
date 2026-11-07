from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel_env import DuelVec, OPPONENT_FROZEN
from hybrid_model import HybridActorCritic


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def act(model, obs, mask, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    logits, _ = model(obs_t, mask_t)
    return logits.argmax(-1).cpu().numpy().astype(np.uint16)


def seat_outcome(env, seat):
    outcome = int(env.outcomes[0])
    if seat == 0:
        return outcome
    if outcome == 1:
        return 2
    if outcome == 2:
        return 1
    return outcome


def run_seat(seed, games, envs, policy, opponent, device, seat):
    env = DuelVec(envs, seed, OPPONENT_FROZEN, min(20, envs))
    rows = []
    frames = []
    last = [
        {"safety": [], "current": [], "best": [], "fire": [], "idle": [], "bullets": []}
        for _ in range(envs)
    ]
    changes = np.zeros(envs, np.int64)
    previous = np.full(envs, -1, np.int64)
    episodes = 0
    while episodes < games:
        if seat == 0:
            obs = env.obs.copy()
            masks = env.masks.astype(bool, copy=True)
            opp_obs = env.obs_opponent.copy()
            opp_masks = env.masks_opponent.astype(bool, copy=True)
        else:
            obs = env.obs_opponent.copy()
            masks = env.masks_opponent.astype(bool, copy=True)
            opp_obs = env.obs.copy()
            opp_masks = env.masks.astype(bool, copy=True)
        actions = act(policy, obs, masks, device)
        opp_actions = act(opponent, opp_obs, opp_masks, device)
        for i in range(envs):
            safety = obs[i, 1018:1027]
            movement = int(actions[i]) // 2
            current = float(safety[movement])
            best = float(safety.max())
            item = {
                "safety": float(safety.min()),
                "current": current,
                "best": best,
                "fire": int(actions[i]) & 1,
                "idle": int(actions[i]) == 8,
                "bullets": int(masks[i].sum()),
            }
            last[i]["safety"].append(item["safety"])
            last[i]["current"].append(current)
            last[i]["best"].append(best)
            last[i]["fire"].append(item["fire"])
            last[i]["idle"].append(item["idle"])
            last[i]["bullets"].append(item["bullets"])
            frames.append(item)
            if previous[i] >= 0 and previous[i] != actions[i]:
                changes[i] += 1
            previous[i] = actions[i]
        if seat == 0:
            env.step(actions, opp_actions)
        else:
            env.step(opp_actions, actions)
        for i in range(envs):
            if env.dones[i] and env.terminals[i]:
                outcome = seat_outcome(env, seat)
                recent = last[i]
                row = {
                    "outcome": outcome,
                    "frames": int(env.episode_frames[i]),
                    "changes": int(changes[i]),
                    "final_safety": recent["safety"][-1] if recent["safety"] else 1.0,
                    "final_current": recent["current"][-1] if recent["current"] else 1.0,
                    "final_best": recent["best"][-1] if recent["best"] else 1.0,
                    "final_bullets": recent["bullets"][-1] if recent["bullets"] else 0,
                    "last5_fire": sum(recent["fire"][-5:]),
                    "last5_idle": sum(recent["idle"][-5:]),
                    "mean_safety": float(np.mean(recent["safety"])) if recent["safety"] else 1.0,
                    "choice_best": float(np.mean(np.array(recent["current"]) >= np.array(recent["best"]) - 1e-6)) if recent["current"] else 1.0,
                    "mean_regret": float(np.mean(np.array(recent["best"]) - np.array(recent["current"]))) if recent["current"] else 0.0,
                }
                rows.append(row)
                episodes += 1
                if episodes >= games:
                    break
                changes[i] = 0
                previous[i] = -1
                last[i] = {"safety": [], "current": [], "best": [], "fire": [], "idle": [], "bullets": []}
        env.reset_done()
    env.close()
    return rows, frames


def summary(rows, frames):
    def mean(values):
        return float(np.mean(values)) if values else 0.0

    out = {"games": len(rows), "frames": len(frames)}
    labels = {1: "win", 2: "loss", 3: "double", 4: "draw"}
    for code, name in labels.items():
        subset = [r for r in rows if r["outcome"] == code]
        if not subset:
            out[name] = {"games": 0}
            continue
        out[name] = {
            "games": len(subset),
            "rate": len(subset) / max(len(rows), 1),
            "mean_frames": mean([r["frames"] for r in subset]),
            "mean_changes": mean([r["changes"] for r in subset]),
            "final_safety": mean([r["final_safety"] for r in subset]),
            "final_bullets": mean([r["final_bullets"] for r in subset]),
            "last5_fire": mean([r["last5_fire"] for r in subset]),
            "last5_idle": mean([r["last5_idle"] for r in subset]),
            "mean_safety": mean([r["mean_safety"] for r in subset]),
            "choice_best": mean([r["choice_best"] for r in subset]),
            "mean_regret": mean([r["mean_regret"] for r in subset]),
        }
    out["all"] = {
        "fire_rate": mean([f["fire"] for f in frames]),
        "idle_rate": mean([f["idle"] for f in frames]),
        "mean_safety": mean([f["safety"] for f in frames]),
        "mean_bullets": mean([f["bullets"] for f in frames]),
        "choice_best": mean([
            f["current"] >= f["best"] - 1e-6 for f in frames
        ]),
        "mean_regret": mean([f["best"] - f["current"] for f in frames]),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/night_run1/checkpoints/hybrid_night_final.pt"))
    parser.add_argument("--opponent", type=Path, default=Path(
        "outputs/night_run1/checkpoints/initial.pt"))
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261301)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load(args.checkpoint, device)
    opponent = load(args.opponent, device)
    result = {}
    for seat in (0, 1):
        rows, frames = run_seat(
            args.seed + seat * 10000, args.games, args.envs,
            policy, opponent, device, seat,
        )
        result[f"seat{seat}"] = {"summary": summary(rows, frames), "games": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        seat: result[f"seat{seat}"]["summary"] for seat in (0, 1)
    }, indent=2))


if __name__ == "__main__":
    main()
