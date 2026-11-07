from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duel_env import (
    DuelVec,
    OPPONENT_FROZEN,
    OPPONENT_LAIKA,
    OPPONENT_MPC,
)
from hybrid_model import HybridActorCritic


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


def choose(model, obs, mask, env, shield, horizon, predict_fire, threshold, margin):
    logits, _ = model(obs, mask)
    actions = logits.argmax(-1).cpu().numpy().astype(np.uint16)
    if shield:
        for i in range(len(actions)):
            safety = env.predictive_safety(i, horizon, predict_fire)
            movement = int(actions[i]) // 2
            current = safety[movement]
            best = int(np.argmax(safety))
            if current < threshold and safety[best] > current + margin:
                actions[i] = best * 2 + (int(actions[i]) & 1)
    return actions


@torch.no_grad()
def run(model, opponent, games, envs, seed, device, shield, horizon, predict_fire,
        threshold, margin, opponent_model=None):
    env = DuelVec(envs, seed, opponent, min(20, envs))
    wins = losses = doubles = draws = episodes = 0
    while episodes < games:
        obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
        mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
        actions = choose(model, obs, mask, env, shield, horizon, predict_fire, threshold, margin)
        opp = None
        if opponent == OPPONENT_FROZEN and opponent_model is not None:
            oo = torch.as_tensor(env.obs_opponent.copy(), dtype=torch.float32, device=device)
            om = torch.as_tensor(env.masks_opponent.astype(bool, copy=True), device=device)
            opp = opponent_model(oo, om)[0].argmax(-1).cpu().numpy().astype(np.uint16)
        env.step(actions, opp)
        for i in range(envs):
            if env.dones[i] and env.terminals[i]:
                o = int(env.outcomes[i])
                wins += o == 1
                losses += o == 2
                doubles += o == 3
                draws += o == 4
                episodes += 1
                if episodes >= games:
                    break
        env.reset_done()
    env.close()
    return {
        "games": episodes, "wins": wins, "losses": losses, "doubles": doubles,
        "draws": draws, "win_rate": wins / max(episodes, 1),
        "shield": {
            "enabled": shield, "horizon": horizon, "predict_fire": predict_fire,
            "threshold": threshold, "margin": margin,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent", choices=("laika", "mpc", "frozen"), default="laika")
    parser.add_argument("--baseline", type=Path, default=Path("outputs/night_run1/checkpoints/initial.pt"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261025)
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
    baseline = load(args.baseline, device) if args.opponent == "frozen" else None
    kind = {"laika": OPPONENT_LAIKA, "mpc": OPPONENT_MPC, "frozen": OPPONENT_FROZEN}[args.opponent]
    result = run(
        model, kind, args.games, args.envs, args.seed, device, args.shield,
        args.horizon, bool(args.predict_fire), args.shield_threshold, args.shield_margin,
        baseline,
    )
    output = args.output or args.checkpoint.with_suffix(f".{args.opponent}.shield.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
