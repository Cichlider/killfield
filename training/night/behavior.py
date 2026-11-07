from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duel_env import OPPONENT_LAIKA, OPPONENT_MPC, DuelVec
from hybrid_model import HybridActorCritic


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def run(model, opponent, games, envs, seed, device):
    env = DuelVec(envs, seed, opponent, min(20, envs))
    rows = []
    episode_shots = np.zeros(envs, np.int64)
    episode_hits = np.zeros(envs, np.int64)
    while len(rows) < games:
        obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
        mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
        logits, _ = model(obs, mask)
        actions = logits.argmax(-1).cpu().numpy().astype(np.uint16)
        env.step(actions, None)
        episode_shots += env.shots
        episode_hits += env.hits
        for i in range(envs):
            if env.dones[i] and env.terminals[i]:
                frames = int(env.episode_frames[i])
                changes = int(env.action_changes[i])
                rows.append({
                    "outcome": int(env.outcomes[i]),
                    "frames": frames,
                    "changes": changes,
                    "change_rate": changes / max(frames - 1, 1),
                    "shots": int(episode_shots[i]),
                    "hits": int(episode_hits[i]),
                })
                if len(rows) >= games:
                    break
        done = env.dones.astype(bool)
        episode_shots[done] = 0
        episode_hits[done] = 0
        env.reset_done()
    env.close()
    frame = np.asarray([r["frames"] for r in rows], np.float64)
    change = np.asarray([r["change_rate"] for r in rows], np.float64)
    shots = np.asarray([r["shots"] for r in rows], np.float64)
    hits = np.asarray([r["hits"] for r in rows], np.float64)
    outcomes = np.asarray([r["outcome"] for r in rows], np.int64)
    return {
        "games": len(rows),
        "wins": int((outcomes == 1).sum()),
        "losses": int((outcomes == 2).sum()),
        "doubles": int((outcomes == 3).sum()),
        "draws": int((outcomes == 4).sum()),
        "win_rate": float((outcomes == 1).mean()),
        "double_rate": float((outcomes == 3).mean()),
        "frames_mean": float(frame.mean()),
        "frames_median": float(np.median(frame)),
        "change_rate_mean": float(change.mean()),
        "shots_mean": float(shots.mean()),
        "hits_mean": float(hits.mean()),
        "hit_per_shot": float(hits.sum() / max(shots.sum(), 1.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load(args.checkpoint, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "laika": run(model, OPPONENT_LAIKA, args.games, args.envs, args.seed + 1, device),
        "mpc": run(model, OPPONENT_MPC, args.games, min(args.envs, 8), args.seed + 2, device),
    }
    output = args.output or args.checkpoint.with_suffix(".behavior.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
