from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hybrid_model import HybridActorCritic


def load_state(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(state, strict=False)
    return model, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.alpha is not None:
        if len(args.checkpoints) != 2:
            raise SystemExit("--alpha needs exactly two checkpoints")
        a, pa = load_state(args.checkpoints[0], device)
        b, pb = load_state(args.checkpoints[1], device)
        sa = a.state_dict()
        sb = b.state_dict()
        mixed = {}
        for key in sa:
            mixed[key] = sa[key] * args.alpha + sb[key] * (1.0 - args.alpha)
        a.load_state_dict(mixed)
        torch.save({
            "model": a.state_dict(),
            "update": pa.get("update", pb.get("update")),
            "total_steps": pa.get("total_steps", pb.get("total_steps")),
            "alpha": args.alpha,
            "members": [str(p) for p in args.checkpoints],
        }, args.output)
        print(f"wrote {args.output} alpha={args.alpha}")
        return
    models = [load_state(path, device)[0] for path in args.checkpoints]
    states = [m.state_dict() for m in models]
    keys = states[0].keys()
    averaged = {}
    for key in keys:
        values = [s[key].float() for s in states]
        averaged[key] = torch.stack(values).mean(0).to(states[0][key].dtype)
    models[0].load_state_dict(averaged)
    torch.save({
        "model": models[0].state_dict(),
        "members": [str(p) for p in args.checkpoints],
        "count": len(models),
    }, args.output)
    print(f"wrote {args.output} from {len(models)} members")


if __name__ == "__main__":
    main()
