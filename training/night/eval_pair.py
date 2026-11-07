from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_model import HybridActorCritic
from train import evaluate_h2h


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tank0", type=Path, required=True)
    parser.add_argument("--tank1", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/night_run1/checkpoints/initial.pt"))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = load(args.tank0, device)
    b = load(args.tank1, device)
    base = load(args.baseline, device)
    t0 = evaluate_h2h(a, base, args.games, args.envs, args.seed, device)
    t1 = evaluate_h2h(base, b, args.games, args.envs, args.seed, device)
    t1_policy = {
        "games": t1["games"],
        "wins": t1["losses"],
        "losses": t1["wins"],
        "doubles": t1["doubles"],
        "draws": t1["draws"],
        "win_rate": t1["losses"] / max(t1["games"], 1),
    }
    combined = {
        "games": t0["games"] + t1_policy["games"],
        "wins": t0["wins"] + t1_policy["wins"],
        "losses": t0["losses"] + t1_policy["losses"],
        "doubles": t0["doubles"] + t1_policy["doubles"],
        "draws": t0["draws"] + t1_policy["draws"],
    }
    combined["win_rate"] = combined["wins"] / max(combined["games"], 1)
    result = {"tank0": t0, "tank1": t1_policy, "combined": combined,
              "seed": args.seed, "games_per_seat": args.games}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
