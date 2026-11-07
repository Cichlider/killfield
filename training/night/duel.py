from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_model import HybridActorCritic
from train import evaluate_h2h


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    a = load(args.a, device)
    b = load(args.b, device)
    ab = evaluate_h2h(a, b, args.games, args.envs, args.seed + 1, device)
    ba = evaluate_h2h(b, a, args.games, args.envs, args.seed + 2, device)
    result = {
        "a": str(args.a), "b": str(args.b),
        "a_as_tank0": ab, "b_as_tank0": ba,
        "a_wins": ab["wins"] + ba["losses"],
        "b_wins": ab["losses"] + ba["wins"],
        "doubles": ab["doubles"] + ba["doubles"],
        "draws": ab["draws"] + ba["draws"],
        "games": ab["games"] + ba["games"],
    }
    result["a_win_rate"] = result["a_wins"] / max(result["games"], 1)
    output = args.output or Path("outputs/night_run1/duel.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
