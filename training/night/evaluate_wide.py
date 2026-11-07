from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import evaluate, evaluate_h2h
from wide_model import WideActorCritic
from duel_env import OPPONENT_LAIKA, OPPONENT_MPC


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = WideActorCritic()
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(
        "outputs/night_run1/checkpoints/initial.pt"))
    parser.add_argument("--games-laika", type=int, default=300)
    parser.add_argument("--games-mpc", type=int, default=100)
    parser.add_argument("--games-h2h", type=int, default=1000)
    parser.add_argument("--envs-laika", type=int, default=64)
    parser.add_argument("--envs-mpc", type=int, default=16)
    parser.add_argument("--envs-h2h", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=20261101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load(args.checkpoint, device)
    baseline = load(args.baseline, device)
    laika = evaluate(model, OPPONENT_LAIKA, args.games_laika, args.envs_laika,
                     args.seed_base + 101, device)
    mpc = evaluate(model, OPPONENT_MPC, args.games_mpc, args.envs_mpc,
                   args.seed_base + 202, device)
    a = evaluate_h2h(model, baseline, args.games_h2h, args.envs_h2h,
                     args.seed_base + 303, device)
    b = evaluate_h2h(baseline, model, args.games_h2h, args.envs_h2h,
                     args.seed_base + 404, device)
    tank1 = {
        "games": b["games"], "wins": b["losses"], "losses": b["wins"],
        "doubles": b["doubles"], "draws": b["draws"],
        "win_rate": b["losses"] / max(b["games"], 1),
    }
    combined = {
        "games": a["games"] + tank1["games"],
        "wins": a["wins"] + tank1["wins"],
        "losses": a["losses"] + tank1["losses"],
        "doubles": a["doubles"] + tank1["doubles"],
        "draws": a["draws"] + tank1["draws"],
    }
    combined["win_rate"] = combined["wins"] / max(combined["games"], 1)
    result = {"checkpoint": str(args.checkpoint), "laika": laika, "mpc": mpc,
              "as_tank0": a, "as_tank1": tank1, "combined": combined}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"laika": laika["win_rate"], "mpc": mpc["win_rate"],
                      "tank0": a["win_rate"], "tank1": tank1["win_rate"],
                      "combined": combined["win_rate"],
                      "double_rate": combined["doubles"] / combined["games"]},
                     indent=2))


if __name__ == "__main__":
    main()
