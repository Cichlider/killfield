#!/usr/bin/env python3
"""Export the deployed Hybrid policy and a deterministic browser parity case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


FIXED_KEYS = (
    "map.0.weight", "map.0.bias",
    "map.2.weight", "map.2.bias", "map.5.weight", "map.5.bias",
    "bullets.0.weight", "bullets.0.bias", "bullets.2.weight", "bullets.2.bias",
    "scalars.0.weight", "scalars.0.bias", "trunk.0.weight", "trunk.0.bias",
    "actor.weight", "actor.bias", "idle_logit_penalty",
)
# `dodge_gate`/`ammo_gate` swap a single flat scalar for a warm-started
# residual gate `alpha_old + delta(features)`; the two shapes are mutually
# exclusive per checkpoint, so the exported key set depends on which one the
# checkpoint was actually trained with -- read from the checkpoint's own
# manifest fields, not assumed. Constructing `ActorCritic()` with defaults
# here (like `serve_live.py`/`eval_duel.py` used to) silently builds the
# wrong architecture for a gated checkpoint and either KeyErrors on load or,
# worse, loads a partial state_dict without complaint.
UNGATED_DODGE_KEYS = ("dodge_scale",)
GATED_DODGE_KEYS = (
    "dodge_alpha_old", "dodge_delta.0.weight", "dodge_delta.0.bias",
    "dodge_delta.2.weight", "dodge_delta.2.bias",
)
UNGATED_AMMO_KEYS = ("ammo_scale", "shot_quality_scale", "ammo_lock_scale", "suicide_scale")
GATED_AMMO_KEYS = (
    "ammo_alpha_old", "ammo_delta.0.weight", "ammo_delta.0.bias",
    "ammo_delta.2.weight", "ammo_delta.2.bias",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", type=Path, default=Path("../killfield/training"))
    parser.add_argument("--output", type=Path, default=Path("viewer/assets/hybrid"))
    parser.add_argument("--name", help="Public checkpoint label stored in the web manifest")
    args = parser.parse_args()

    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from duel_env import BULLET_SLOTS, OBS_DIM  # noqa: PLC0415
    from duel_ppo import ActorCritic  # noqa: PLC0415

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Gate config (dodge_gate/ammo_gate/*_alpha_old) is architecture metadata,
    # not weights, so it lives in the checkpoint's companion manifest
    # (live.json/complete.json's shape), not inside the .pt payload itself --
    # same place serve_live.py/eval_duel.py read it from.
    manifest_sidecar = args.checkpoint.with_suffix(".json")
    if not manifest_sidecar.exists():
        for sibling in (args.checkpoint.with_name("live.json"),
                        args.checkpoint.with_name("complete.json")):
            if sibling.exists():
                manifest_sidecar = sibling
                break
    side = json.loads(manifest_sidecar.read_text()) if manifest_sidecar.exists() else {}
    dodge_gate = bool(side.get("dodge_gate", False))
    ammo_gate = bool(side.get("ammo_gate", False))
    model = ActorCritic(
        dodge_gate=dodge_gate, ammo_gate=ammo_gate,
        dodge_alpha_old=side.get("dodge_alpha_old", 0.0),
        ammo_alpha_old=side.get("ammo_alpha_old", (0.0, 0.0, 0.0, 0.0)),
    )
    model.load_state_dict(payload["model"])
    model.eval()

    actor_keys = (
        (GATED_DODGE_KEYS if dodge_gate else UNGATED_DODGE_KEYS)
        + (GATED_AMMO_KEYS if ammo_gate else UNGATED_AMMO_KEYS)
        + FIXED_KEYS
    )
    print(f"dodge_gate={dodge_gate} ammo_gate={ammo_gate}")

    arrays: list[np.ndarray] = []
    tensors: dict[str, dict[str, object]] = {}
    offset = 0
    for key in actor_keys:
        value = payload["model"][key].detach().cpu().numpy().astype("<f4", copy=False)
        flat = value.reshape(-1)
        tensors[key] = {"shape": list(value.shape), "offset": offset, "length": flat.size}
        arrays.append(flat)
        offset += flat.size

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = args.output.with_suffix(".bin")
    manifest_path = args.output.with_suffix(".json")
    parity_path = args.output.with_name(args.output.name + "-parity").with_suffix(".json")
    np.concatenate(arrays).tofile(weights_path)

    result = payload.get("result", {})
    manifest = {
        "format": "killfield-hybrid-f32-v1",
        "checkpoint": args.name or args.checkpoint.name,
        "schema": int(side.get("schema_version", result.get("obs_schema", 24))),
        "observation": OBS_DIM,
        "bullet_slots": BULLET_SLOTS,
        "actions": 18,
        "floats": offset,
        "gated": {"dodge": dodge_gate, "ammo": ammo_gate},
        "tensors": tensors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rng = np.random.default_rng(240918)
    obs = rng.uniform(-1, 1, size=(1, OBS_DIM)).astype(np.float32)
    mask = np.array([[True, False, True, True, False, False, True, False, False, True]])
    dodge = rng.uniform(-1, 1, size=(1, 9)).astype(np.float32)
    with torch.inference_mode():
        logits, _ = model(torch.from_numpy(obs), torch.from_numpy(mask), torch.from_numpy(dodge))
    parity = {
        "obs": obs[0].tolist(), "mask": mask[0].astype(int).tolist(),
        "dodge": dodge[0].tolist(), "logits": logits[0].tolist(),
        "action": int(logits.argmax(1).item()),
    }
    parity_path.write_text(json.dumps(parity, separators=(",", ":")) + "\n")
    print(f"wrote {weights_path} ({weights_path.stat().st_size} bytes)")
    print(f"wrote {manifest_path} and {parity_path}")


if __name__ == "__main__":
    main()
