from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hybrid_model import ACTIONS, BULLET_SLOTS, OBS_DIM, HybridActorCritic

FIXED = (
    "map.0.weight", "map.0.bias",
    "map.2.weight", "map.2.bias",
    "map.5.weight", "map.5.bias",
    "bullets.0.weight", "bullets.0.bias",
    "bullets.2.weight", "bullets.2.bias",
    "scalars.0.weight", "scalars.0.bias",
    "trunk.0.weight", "trunk.0.bias",
    "actor.weight", "actor.bias",
    "idle_logit_penalty",
)
DODGE = (
    "dodge_alpha_old",
    "dodge_delta.0.weight", "dodge_delta.0.bias",
    "dodge_delta.2.weight", "dodge_delta.2.bias",
)
AMMO = (
    "ammo_alpha_old",
    "ammo_delta.0.weight", "ammo_delta.0.bias",
    "ammo_delta.2.weight", "ammo_delta.2.bias",
)
DERIVED = ("derived_up.weight", "derived_up.bias")
STATE_NAMES = {
    "map.0.weight": "map0.weight", "map.0.bias": "map0.bias",
    "map.2.weight": "map2.weight", "map.2.bias": "map2.bias",
    "map.5.weight": "map5.weight", "map.5.bias": "map5.bias",
    "bullets.0.weight": "bullets0.weight", "bullets.0.bias": "bullets0.bias",
    "bullets.2.weight": "bullets2.weight", "bullets.2.bias": "bullets2.bias",
    "scalars.0.weight": "scalars0.weight", "scalars.0.bias": "scalars0.bias",
    "trunk.0.weight": "trunk0.weight", "trunk.0.bias": "trunk0.bias",
    "dodge_delta.0.weight": "dodge0.weight", "dodge_delta.0.bias": "dodge0.bias",
    "dodge_delta.2.weight": "dodge2.weight", "dodge_delta.2.bias": "dodge2.bias",
    "ammo_delta.0.weight": "ammo0.weight", "ammo_delta.0.bias": "ammo0.bias",
    "ammo_delta.2.weight": "ammo2.weight", "ammo_delta.2.bias": "ammo2.bias",
    "derived_up.weight": "derived.weight", "derived_up.bias": "derived.bias",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("viewer/assets/hybrid_night"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-derived", action="store_true",
                        help="export the derived_up zero-init projection tensors")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = payload.get("model", payload)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    keys = list(FIXED)
    if model.gated_dodge:
        keys = list(DODGE) + keys
    if model.gated_ammo:
        keys = list(AMMO) + keys
    if args.with_derived and all(name in model.state_dict() for name in DERIVED):
        keys = keys + list(DERIVED)
    arrays = []
    tensors = {}
    offset = 0
    state = model.state_dict()
    for key in keys:
        state_name = STATE_NAMES.get(key, key)
        lookup = state_name if state_name in state else key
        value = state[lookup].detach().cpu().numpy().astype("<f4", copy=False)
        flat = value.reshape(-1)
        tensors[key] = {
            "shape": list(value.shape),
            "offset": offset,
            "length": int(flat.size),
        }
        arrays.append(flat)
        offset += int(flat.size)
    total = offset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = args.output.with_suffix(".bin")
    manifest_path = args.output.with_suffix(".json")
    np.concatenate(arrays).tofile(weights_path)
    manifest = {
        "format": "killfield-hybrid-f32-v1",
        "checkpoint": args.checkpoint.name,
        "schema": 24,
        "observation": OBS_DIM,
        "bullet_slots": BULLET_SLOTS,
        "actions": ACTIONS,
        "floats": total,
        "gated": {"dodge": bool(model.gated_dodge), "ammo": bool(model.gated_ammo)},
        "tensors": tensors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {weights_path} ({weights_path.stat().st_size} bytes)")
    print(f"wrote {manifest_path} floats={total}")


if __name__ == "__main__":
    main()
