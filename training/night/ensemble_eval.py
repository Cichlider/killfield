from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duel_env import OPPONENT_FROZEN, OPPONENT_LAIKA, OPPONENT_MPC, DuelVec
from hybrid_model import HybridActorCritic, load_default


def load_model(path: Path | None, device):
    if path is None:
        return load_default(device).eval()
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def ensemble_logits(models, obs, mask):
    total = None
    for model in models:
        logits, _ = model(obs, mask)
        total = logits if total is None else total + logits
    return total / len(models)


@torch.no_grad()
def run(models, opponent, games, envs, seed, device, opponent_model=None):
    env = DuelVec(envs, seed, opponent, min(20, envs))
    wins = losses = doubles = draws = episodes = 0
    while episodes < games:
        obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
        mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
        logits = ensemble_logits(models, obs, mask)
        actions = logits.argmax(-1).cpu().numpy().astype(np.uint16)
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
    return {"wins": wins, "losses": losses, "doubles": doubles, "draws": draws,
            "games": episodes, "win_rate": wins / max(episodes, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--games-laika", type=int, default=64)
    parser.add_argument("--games-mpc", type=int, default=24)
    parser.add_argument("--games-h2h", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    models = [load_model(path, device) for path in args.checkpoints]
    baseline = load_default(device).eval()
    result = {
        "members": [str(p) for p in args.checkpoints],
        "laika": run(models, OPPONENT_LAIKA, args.games_laika, 32, args.seed + 101, device),
        "mpc": run(models, OPPONENT_MPC, args.games_mpc, 8, args.seed + 202, device),
        "h2h": run(models, OPPONENT_FROZEN, args.games_h2h, 32, args.seed + 303, device, baseline),
    }
    output = args.output or Path("outputs/night_run1/ensemble_eval.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
