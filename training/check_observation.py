"""Range and normalization audit for the frozen semantic observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ppo_models import (
    ACTION_COUNT, ACTION_OFFSET, BULLET_DIM, BULLET_OFFSET, BULLET_SLOTS, MAP_DIM, OBS_DIM,
    OBS_SCHEMA_VERSION, OPPONENT_OFFSET, PHASE_OFFSET, SELF_OFFSET, TIME_OFFSET,
)


def stats(values):
    return {
        "min": float(values.min(initial=0.0)),
        "max": float(values.max(initial=0.0)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "nonzero_rate": float(np.count_nonzero(values) / values.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("/tmp/kf-probe-oar"))
    args = parser.parse_args()
    meta = json.loads((args.root / "rollout.json").read_text())
    if meta["obs_schema_version"] != OBS_SCHEMA_VERSION or meta["obs_dim"] != OBS_DIM:
        raise RuntimeError(f"schema mismatch: {meta}")
    obs = np.fromfile(args.root / "obs.f32", np.float32).reshape(-1, OBS_DIM)
    masks = np.fromfile(args.root / "bullet_mask.u8", np.uint8).reshape(-1, BULLET_SLOTS).astype(bool)
    failures = []
    warnings = []

    def require(name, condition, detail):
        if not bool(condition):
            failures.append({"check": name, "detail": detail})

    require("finite", np.isfinite(obs).all(), "observation contains NaN or Inf")
    binary_columns = np.r_[
        np.arange(MAP_DIM),
        np.arange(SELF_OFFSET + 5, SELF_OFFSET + 9),
        [OPPONENT_OFFSET + 5],
        np.arange(PHASE_OFFSET, PHASE_OFFSET + 3),
        np.arange(ACTION_OFFSET, ACTION_OFFSET + ACTION_COUNT),
    ]
    binary = obs[:, binary_columns]
    require("binary_channels", np.isin(binary, (0.0, 1.0)).all(), "binary value outside {0,1}")
    require(
        "self_xy", ((obs[:, SELF_OFFSET:SELF_OFFSET + 2] >= 0.0)
                    & (obs[:, SELF_OFFSET:SELF_OFFSET + 2] <= 1.0)).all(),
        "self position outside [0,1]",
    )
    path_upper_bound = (12 * 10 - 1) / (12 + 10)
    require(
        "path_length_bound",
        ((obs[:, MAP_DIM] >= 0.0) & (obs[:, MAP_DIM] <= path_upper_bound + 1e-6)).all(),
        f"path exceeds analytic bound {path_upper_bound:.6g}",
    )
    if obs[:, MAP_DIM].max() > 1.0:
        warnings.append({
            "check": "path_not_unit_normalized",
            "detail": (
                f"empirical max {obs[:, MAP_DIM].max():.6g}; frozen schema divides by 22, "
                f"analytic bound is {path_upper_bound:.6g}"
            ),
        })
    require(
        "opponent_relative", np.abs(obs[:, OPPONENT_OFFSET:OPPONENT_OFFSET + 2]).max() <= 2**0.5 + 1e-5,
        "opponent local position exceeds geometric sqrt(2) bound",
    )
    require(
        "phase_one_hot", np.allclose(obs[:, PHASE_OFFSET:PHASE_OFFSET + 3].sum(1), 1.0),
        "phase is not exactly one-hot",
    )
    require(
        "time_range", ((obs[:, TIME_OFFSET] >= 0.0) & (obs[:, TIME_OFFSET] <= 1.0)).all(),
        "elapsed time is outside [0,1]",
    )
    action_sum = obs[:, ACTION_OFFSET:].sum(1)
    require(
        "previous_action",
        np.isin(action_sum, (0.0, 1.0)).all(),
        "previous Discrete(130) action is not empty/one-hot",
    )

    bullets = obs[:, BULLET_OFFSET:PHASE_OFFSET].reshape(-1, BULLET_SLOTS, BULLET_DIM)
    require("mask_prefix", np.all(np.diff(masks.astype(np.int8), axis=1) <= 0), "mask has holes")
    require("padding_zero", np.all(bullets[~masks] == 0.0), "padded bullet row is nonzero")
    if masks.any():
        active = bullets[masks]
        require("bullet_position", np.abs(active[:, :2]).max() <= 2**0.5 + 0.1,
                "bullet local position exceeds geometric bound")
        require("bullet_velocity", np.abs(active[:, 2:4]).max() <= 1.0 + 1e-4,
                "normalized bullet velocity component exceeds 1")
        require("bullet_flags", np.isin(active[:, 4:6], (0.0, 1.0)).all(),
                "bullet owner/bounce value outside {0,1}")

    groups = {
        "map": obs[:, :MAP_DIM],
        "path_length": obs[:, MAP_DIM:SELF_OFFSET],
        "self": obs[:, SELF_OFFSET:OPPONENT_OFFSET],
        "opponent": obs[:, OPPONENT_OFFSET:BULLET_OFFSET],
        "bullets_active": bullets[masks] if masks.any() else np.zeros((1, BULLET_DIM), np.float32),
        "phase": obs[:, PHASE_OFFSET:TIME_OFFSET],
        "time": obs[:, TIME_OFFSET:ACTION_OFFSET],
        "previous_action": obs[:, ACTION_OFFSET:],
    }
    report = {
        "schema_version": OBS_SCHEMA_VERSION,
        "obs_dim": OBS_DIM,
        "samples": len(obs),
        "finite": bool(np.isfinite(obs).all()),
        "overall": stats(obs),
        "groups": {name: stats(value) for name, value in groups.items()},
        "bullet_count": {
            "mean": float(masks.sum(1).mean()),
            "max": int(masks.sum(1).max()),
            "full_slot_rate": float((masks.sum(1) == BULLET_SLOTS).mean()),
        },
        "failures": failures,
        "warnings": warnings,
        "passed": not failures,
    }
    output = args.root / "observation_health.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
