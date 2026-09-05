"""Record a browser-portable replay from the real engine and a PPO checkpoint.

The output contains render states, not inferred illustrations: each frame is
read from the same native engine buffer used by the live viewer.  This keeps
the paper deployable as a static GitHub Pages site without shipping PyTorch.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from duel_env import BULLET_SLOTS, DODGE_DIM, OBS_DIM  # noqa: E402
from duel_ppo import ActorCritic, load_weights  # noqa: E402

HEADER = 14


class Duel:
    def __init__(self, library: Path, seed: int):
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.lib.kf_new_duel.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.lib.kf_new_duel.restype = ctypes.c_void_p
        self.lib.kf_free.argtypes = [ctypes.c_void_p]
        self.lib.kf_observation.argtypes = [ctypes.c_void_p]
        self.lib.kf_observation.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.kf_dodge_safety.argtypes = [ctypes.c_void_p]
        self.lib.kf_dodge_safety.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.kf_render_ptr.argtypes = [ctypes.c_void_p]
        self.lib.kf_render_ptr.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.kf_render_len.argtypes = [ctypes.c_void_p]
        self.lib.kf_render_len.restype = ctypes.c_uint32
        self.lib.kf_step.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.kf_step.restype = ctypes.c_uint32
        self.handle = self.lib.kf_new_duel(seed, 0)  # fixed Laika

    def observe(self):
        raw = np.ctypeslib.as_array(
            self.lib.kf_observation(self.handle), shape=(OBS_DIM + BULLET_SLOTS,)
        ).copy()
        dodge = np.ctypeslib.as_array(
            self.lib.kf_dodge_safety(self.handle), shape=(DODGE_DIM,)
        ).copy()
        return raw[:OBS_DIM], raw[OBS_DIM:].astype(bool), dodge

    def render(self):
        length = int(self.lib.kf_render_len(self.handle))
        return np.ctypeslib.as_array(
            self.lib.kf_render_ptr(self.handle), shape=(length,)
        ).copy()

    def step(self, action: int):
        return int(self.lib.kf_step(self.handle, action))

    def close(self):
        if self.handle:
            self.lib.kf_free(self.handle)
            self.handle = None


def unpack(buf: np.ndarray) -> dict:
    walls_n, tanks_n, bullets_n = map(int, buf[4:7])
    at = HEADER
    walls = buf[at : at + walls_n * 4].reshape(-1, 4).round(3).tolist()
    at += walls_n * 4
    tanks = buf[at : at + tanks_n * 4].reshape(-1, 4).round(3).tolist()
    at += tanks_n * 4
    bullets = buf[at : at + bullets_n * 3].reshape(-1, 3).round(3).tolist()
    return {
        "maze": [int(buf[0]), int(buf[1])],
        "scale": round(float(buf[2]), 3),
        "wall_half": round(float(buf[3]), 3),
        "walls": walls,
        "frame": int(buf[7]),
        "outcome": int(buf[8]),
        "shots": int(buf[9]),
        "reward": round(float(buf[10]), 4),
        "tanks": tanks,
        "bullets": bullets,
    }


@torch.inference_mode()
def play(model: ActorCritic, library: Path, seed: int) -> dict:
    duel = Duel(library, seed)
    frames = []
    actions = []
    try:
        initial = unpack(duel.render())
        walls = initial.pop("walls")
        frames.append(initial)
        for _ in range(15_000):
            obs, mask, dodge = duel.observe()
            logits, _ = model(
                torch.from_numpy(obs[None]).float(),
                torch.from_numpy(mask[None]),
                torch.from_numpy(dodge[None]).float(),
            )
            action = int(logits.argmax(-1).item())
            actions.append(action)
            ended = duel.step(action) & 1
            state = unpack(duel.render())
            state.pop("walls")
            frames.append(state)
            if ended:
                break
        return {"seed": seed, "walls": walls, "frames": frames, "actions": actions}
    finally:
        duel.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/ppo_duel_v16/s11")
    parser.add_argument("--library", type=Path, default=ROOT / "engine/target/release/libkf_engine.dylib")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/data/replay.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260904, 20260920)))
    args = parser.parse_args()

    manifest = json.loads((args.run / "live.json").read_text())
    payload = torch.load(args.run / "live.pt", map_location="cpu", weights_only=False)
    model = ActorCritic()
    load_weights(model, payload["model"], str(args.run / "live.pt"))
    model.eval()

    candidates = [play(model, args.library, seed) for seed in args.seeds]
    wins = [r for r in candidates if r["frames"][-1]["outcome"] == 1]
    useful = [r for r in wins if 120 <= r["frames"][-1]["frame"] <= 450]
    pool = useful or wins or candidates
    # Prefer a legible, eventful round: some duration and several live bullets.
    replay = max(
        pool,
        key=lambda r: sum(len(f["bullets"]) for f in r["frames"])
        + 8 * min(r["frames"][-1]["frame"], 300),
    )
    replay["checkpoint"] = {
        "run": f"{args.run.parent.name}/{args.run.name}",
        "steps": manifest["steps"],
        "trained_steps": manifest["trained_steps"],
        "schema": manifest["schema_version"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(replay, separators=(",", ":")))
    final = replay["frames"][-1]
    print(f"seed={replay['seed']} outcome={final['outcome']} frames={final['frame']} shots={final['shots']}")


if __name__ == "__main__":
    main()
