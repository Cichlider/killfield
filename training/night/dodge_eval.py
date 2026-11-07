from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duel_env import DuelVec, OPPONENT_FROZEN
from hybrid_model import HybridActorCritic


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def run(model, games, envs, seed, device, prob, cooldown, count, spread,
        shield, horizon, predict_fire, threshold, margin):
    env = DuelVec(envs, seed, OPPONENT_FROZEN, min(20, envs))
    env.set_barrage(prob, cooldown, count, spread)
    rows = []
    while len(rows) < games:
        obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
        mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
        logits, _ = model(obs, mask)
        movement = logits[:, 0::2].argmax(-1).cpu().numpy()
        if shield:
            for i in range(envs):
                safety = env.predictive_safety(i, horizon, predict_fire)
                current = safety[movement[i]]
                best = int(np.argmax(safety))
                if current < threshold and safety[best] > current + margin:
                    movement[i] = best
        actions = (movement * 2).astype(np.uint16)
        env.step(actions, np.zeros(envs, dtype=np.uint16))
        for i in range(envs):
            if env.dones[i] and env.terminals[i]:
                rows.append({
                    "outcome": int(env.outcomes[i]),
                    "frames": int(env.episode_frames[i]),
                    "safety": float(env.safety[i]),
                })
                if len(rows) >= games:
                    break
        env.reset_done()
    env.close()
    frames = np.asarray([r["frames"] for r in rows], np.float64)
    return {
        "games": len(rows),
        "deaths": int(sum(r["outcome"] == 2 for r in rows)),
        "double_deaths": int(sum(r["outcome"] == 3 for r in rows)),
        "draws": int(sum(r["outcome"] == 4 for r in rows)),
        "survival_rate": float(sum(r["outcome"] == 4 for r in rows) / max(len(rows), 1)),
        "frames_mean": float(frames.mean()),
        "frames_median": float(np.median(frames)),
        "frames_min": float(frames.min()),
        "frames_max": float(frames.max()),
        "barrage": {"prob": prob, "cooldown": cooldown, "count": count, "spread": spread},
        "shield": {
            "enabled": shield, "horizon": horizon, "predict_fire": predict_fire,
            "threshold": threshold, "margin": margin,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--prob", type=float, default=0.08)
    parser.add_argument("--cooldown", type=int, default=15)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--spread", type=float, default=50.0)
    parser.add_argument("--shield", action="store_true")
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--predict-fire", type=int, default=1)
    parser.add_argument("--shield-threshold", type=float, default=0.35)
    parser.add_argument("--shield-margin", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load(args.checkpoint, device)
    result = run(
        model, args.games, args.envs, args.seed, device,
        args.prob, args.cooldown, args.count, args.spread,
        args.shield, args.horizon, bool(args.predict_fire), args.shield_threshold,
        args.shield_margin,
    )
    output = args.output or args.checkpoint.with_suffix(".dodge.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
