"""ctypes wrapper over the Rust vectorised duel environment.

Shared by the trainer and the benchmark so there is exactly one place that
knows the ABI. Every buffer is a zero-copy `numpy` view into Rust memory and
stays valid until the next call that mutates the environment, so anything kept
across a `step` has to be copied.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

# Mirrored from engine/src/duel_obs.rs. Checked against the engine's own
# reported values at construction rather than trusted.
OBS_SCHEMA_VERSION = 20
OBS_DIM = 1010
BULLET_SLOTS = 10
ACTIONS = 18

MAP_W, MAP_H, MAP_CHANNELS = 12, 10, 7
MAP_DIM = MAP_W * MAP_H * MAP_CHANNELS          # 840
BULLET_OFFSET = 900
BULLET_DIM = 10
BULLET_BLOCK = BULLET_SLOTS * BULLET_DIM        # 100
# Everything that is neither the map grid nor the bullet rows.
SCALAR_DIM = OBS_DIM - MAP_DIM - BULLET_BLOCK   # 70

OUTCOME_NAMES = {0: "running", 1: "win", 2: "loss", 3: "double", 4: "draw"}
OPPONENT_NAMES = {0: "laika", 1: "mpc", 2: "frozen"}

DEFAULT_LIBRARY = Path("engine/target/release/libkf_engine.dylib")


class DuelVec:
    def __init__(self, count: int, seed: int, weights=(1.0, 0.0, 0.0),
                 threads: int = 0, library: Path = DEFAULT_LIBRARY):
        """`weights` is (laika, mpc, frozen); it is normalised, not required to
        sum to one. A frozen slot publishes tank 1's observation and expects an
        action back — see `obs_opponent` and `needs_action`."""
        self.count = count
        self.lib = ctypes.CDLL(str(Path(library).resolve()))

        for name in ("kf_duel_obs_dim", "kf_duel_bullet_slots", "kf_duel_action_count",
                     "kf_duel_frames", "kf_duel_grace_frames",
                     "kf_duel_obs_schema_version"):
            getattr(self.lib, name).restype = ctypes.c_uint32

        native = (
            int(self.lib.kf_duel_obs_dim()),
            int(self.lib.kf_duel_bullet_slots()),
            int(self.lib.kf_duel_action_count()),
            int(self.lib.kf_duel_obs_schema_version()),
        )
        expected = (OBS_DIM, BULLET_SLOTS, ACTIONS, OBS_SCHEMA_VERSION)
        if native != expected:
            raise RuntimeError(
                f"engine/python schema mismatch: {native} != {expected}. "
                "Rebuild the engine (cargo build --release) or fix duel_env.py."
            )
        self.actions = ACTIONS
        self.episode_frames = int(self.lib.kf_duel_frames())
        self.grace_frames = int(self.lib.kf_duel_grace_frames())

        self.lib.kf_duel_new.argtypes = [ctypes.c_uint32] * 6
        self.lib.kf_duel_new.restype = ctypes.c_void_p
        self.lib.kf_duel_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.lib.kf_duel_reset_done.argtypes = [ctypes.c_void_p]
        self.lib.kf_duel_free.argtypes = [ctypes.c_void_p]

        total = sum(max(0.0, w) for w in weights) or 1.0
        permille = [int(round(1000 * max(0.0, w) / total)) for w in weights]
        self.weights = tuple(w / total for w in weights)
        # threads=0 lets the engine ask the OS how many cores it has.
        self.handle = self.lib.kf_duel_new(count, seed, *permille, threads)

        self.obs = self._view("kf_duel_obs", ctypes.c_float, (count, OBS_DIM))
        self.masks = self._view("kf_duel_masks", ctypes.c_uint8, (count, BULLET_SLOTS))
        self.obs_opponent = self._view("kf_duel_opponent_obs", ctypes.c_float,
                                       (count, OBS_DIM))
        self.masks_opponent = self._view("kf_duel_opponent_masks", ctypes.c_uint8,
                                         (count, BULLET_SLOTS))
        self.needs_action = self._view("kf_duel_needs_action", ctypes.c_uint8, (count,))
        self.rewards = self._view("kf_duel_rewards", ctypes.c_float, (count,))
        self.dones = self._view("kf_duel_dones", ctypes.c_uint8, (count,))
        self.terminals = self._view("kf_duel_terminals", ctypes.c_uint8, (count,))
        self.outcomes = self._view("kf_duel_outcomes", ctypes.c_uint8, (count,))
        self.opponents = self._view("kf_duel_opponents", ctypes.c_uint8, (count,))
        self.frames = self._view("kf_duel_episode_frames", ctypes.c_uint32, (count,))
        self.action_changes = self._view("kf_duel_action_changes", ctypes.c_uint32, (count,))
        self.shots = self._view("kf_duel_shots", ctypes.c_uint32, (count,))
        self.hits = self._view("kf_duel_hits", ctypes.c_uint32, (count,))

    def _view(self, name, ctype, shape):
        function = getattr(self.lib, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.POINTER(ctype)
        return np.ctypeslib.as_array(function(self.handle), shape=shape)

    def step(self, actions, opponent_actions=None):
        actions = np.ascontiguousarray(actions, np.uint16)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        if opponent_actions is None:
            theirs = ctypes.cast(None, u16)
        else:
            buf = np.ascontiguousarray(opponent_actions, np.uint16)
            theirs = buf.ctypes.data_as(u16)
            self._opponent_buffer = buf  # keep alive across the call
        self.lib.kf_duel_step(self.handle, actions.ctypes.data_as(u16), theirs)

    def reset_done(self):
        self.lib.kf_duel_reset_done(self.handle)

    def close(self):
        if getattr(self, "handle", None):
            self.lib.kf_duel_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
