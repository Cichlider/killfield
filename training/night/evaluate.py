from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from duel_env import OPPONENT_LAIKA, OPPONENT_MPC
from hybrid_model import HybridActorCritic, load_default
from train import evaluate, evaluate_h2h


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--games-laika", type=int, default=100)
    parser.add_argument("--games-mpc", type=int, default=100)
    parser.add_argument("--games-h2h", type=int, default=200)
    parser.add_argument("--envs-laika", type=int, default=32)
    parser.add_argument("--envs-mpc", type=int, default=8)
    parser.add_argument("--envs-h2h", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=20261018)
    parser.add_argument("--two-bases", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    baseline = load_default(device).eval()
    bases = [args.seed_base] + ([args.seed_base + 1_000_000] if args.two_bases else [])
    result = {
        "checkpoint": str(args.checkpoint),
        "update": payload.get("update"),
        "total_steps": payload.get("total_steps"),
        "seeds": bases,
        "laika": [],
        "mpc": [],
        "h2h": [],
    }
    for seed in bases:
        result["laika"].append(evaluate(
            model, OPPONENT_LAIKA, args.games_laika, args.envs_laika, seed + 101, device
        ))
        result["mpc"].append(evaluate(
            model, OPPONENT_MPC, args.games_mpc, args.envs_mpc, seed + 202, device
        ))
        a = evaluate_h2h(model, baseline, args.games_h2h, args.envs_h2h, seed + 303, device)
        b = evaluate_h2h(baseline, model, args.games_h2h, args.envs_h2h, seed + 5_000_404, device)
        tank1 = {
            "games": b["games"],
            "wins": b["losses"],
            "losses": b["wins"],
            "doubles": b["doubles"],
            "draws": b["draws"],
            "win_rate": b["losses"] / max(b["games"], 1),
        }
        combined = {
            "games": a["games"] + b["games"],
            "wins": a["wins"] + b["losses"],
            "losses": a["losses"] + b["wins"],
            "doubles": a["doubles"] + b["doubles"],
            "draws": a["draws"] + b["draws"],
        }
        combined["win_rate"] = combined["wins"] / max(combined["games"], 1)
        result["h2h"].append({"as_tank0": a, "as_tank1": tank1, "combined": combined})
    def total(rows):
        wins = sum(r["wins"] for r in rows)
        games = sum(r["games"] for r in rows)
        return {"wins": wins, "games": games, "win_rate": wins / max(games, 1)}
    result["summary"] = {
        "laika": total(result["laika"]),
        "mpc": total(result["mpc"]),
        "h2h": {
            "wins": sum(r["combined"]["wins"] for r in result["h2h"]),
            "games": sum(r["combined"]["games"] for r in result["h2h"]),
            "win_rate": sum(r["combined"]["wins"] for r in result["h2h"]) / max(sum(r["combined"]["games"] for r in result["h2h"]), 1),
        },
    }
    output = args.output or args.checkpoint.with_suffix(".eval.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
