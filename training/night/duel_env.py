from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

OBS_DIM = 1028
BULLET_SLOTS = 10
ACTIONS = 18
OPPONENT_LAIKA = 0
OPPONENT_MPC = 1
OPPONENT_FROZEN = 2


class DuelVec:
    def __init__(
        self,
        count: int,
        seed: int,
        opponent: int = OPPONENT_LAIKA,
        threads: int = 0,
        library: Path | None = None,
    ):
        root = Path(__file__).resolve().parents[2]
        lib_path = Path(library) if library is not None else root / "engine" / "target" / "release" / "kf_engine.dll"
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.kf_duel24_new.argtypes = [ctypes.c_uint32] * 4
        self.lib.kf_duel24_new.restype = ctypes.c_void_p
        self.lib.kf_duel24_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.lib.kf_duel24_reset_done.argtypes = [ctypes.c_void_p]
        self.lib.kf_duel24_set_barrage.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float,
        ]
        self.lib.kf_duel24_predictive_safety.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.kf_duel24_free.argtypes = [ctypes.c_void_p]
        self.lib.kf_duel24_search_snapshot.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        self.lib.kf_duel24_search_snapshot.restype = ctypes.c_void_p
        self.lib.kf_duel24_search_free.argtypes = [ctypes.c_void_p]
        self.lib.kf_duel24_search_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        for name in (
            "kf_duel24_search_obs",
            "kf_duel24_search_opponent_obs",
            "kf_duel24_search_rewards",
        ):
            getattr(self.lib, name).restype = ctypes.POINTER(ctypes.c_float)
        for name in (
            "kf_duel24_search_masks",
            "kf_duel24_search_opponent_masks",
            "kf_duel24_search_dones",
            "kf_duel24_search_outcomes",
        ):
            getattr(self.lib, name).restype = ctypes.POINTER(ctypes.c_uint8)
        self.lib.kf_duel24_search_frames.restype = ctypes.POINTER(ctypes.c_uint32)
        for name in (
            "kf_duel24_obs_dim",
            "kf_duel24_bullet_slots",
            "kf_duel24_action_count",
            "kf_duel24_frames",
            "kf_duel24_grace_frames",
            "kf_duel24_obs_schema_version",
        ):
            getattr(self.lib, name).restype = ctypes.c_uint32
        native = (
            int(self.lib.kf_duel24_obs_dim()),
            int(self.lib.kf_duel24_bullet_slots()),
            int(self.lib.kf_duel24_action_count()),
            int(self.lib.kf_duel24_obs_schema_version()),
        )
        if native != (OBS_DIM, BULLET_SLOTS, ACTIONS, 24):
            raise RuntimeError(f"native/Python schema mismatch: {native}")
        self.count = count
        self.opponent = opponent
        self.handle = self.lib.kf_duel24_new(count, seed, opponent, threads)
        if not self.handle:
            raise RuntimeError("kf_duel24_new failed")
        self.episode_frames_limit = int(self.lib.kf_duel24_frames())
        self.grace_frames = int(self.lib.kf_duel24_grace_frames())
        self.obs = self._view("kf_duel24_obs", ctypes.c_float, (count, OBS_DIM))
        self.masks = self._view("kf_duel24_masks", ctypes.c_uint8, (count, BULLET_SLOTS))
        self.obs_opponent = self._view("kf_duel24_opponent_obs", ctypes.c_float, (count, OBS_DIM))
        self.masks_opponent = self._view("kf_duel24_opponent_masks", ctypes.c_uint8, (count, BULLET_SLOTS))
        self.needs_action = self._view("kf_duel24_needs_action", ctypes.c_uint8, (count,))
        self.rewards = self._view("kf_duel24_rewards", ctypes.c_float, (count,))
        self.dones = self._view("kf_duel24_dones", ctypes.c_uint8, (count,))
        self.terminals = self._view("kf_duel24_terminals", ctypes.c_uint8, (count,))
        self.outcomes = self._view("kf_duel24_outcomes", ctypes.c_uint8, (count,))
        self.episode_frames = self._view("kf_duel24_episode_frames", ctypes.c_uint32, (count,))
        self.action_changes = self._view("kf_duel24_action_changes", ctypes.c_uint32, (count,))
        self.shots = self._view("kf_duel24_shots", ctypes.c_uint32, (count,))
        self.hits = self._view("kf_duel24_hits", ctypes.c_uint32, (count,))
        self.safety = self._view("kf_duel24_safety", ctypes.c_float, (count,))
        self._opponent_buffer = None

    def _view(self, name, ctype, shape):
        function = getattr(self.lib, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.POINTER(ctype)
        return np.ctypeslib.as_array(function(self.handle), shape=shape)

    def step(self, actions, opponent_actions=None):
        actions = np.ascontiguousarray(actions, np.uint16)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        a_ptr = actions.ctypes.data_as(u16)
        if opponent_actions is None:
            o_ptr = ctypes.cast(None, u16)
        else:
            buf = np.ascontiguousarray(opponent_actions, np.uint16)
            self._opponent_buffer = buf
            o_ptr = buf.ctypes.data_as(u16)
        self.lib.kf_duel24_step(self.handle, a_ptr, o_ptr)

    def reset_done(self):
        self.lib.kf_duel24_reset_done(self.handle)

    def set_barrage(self, prob, cooldown=1, count=1, spread=30.0):
        self.lib.kf_duel24_set_barrage(self.handle, prob, cooldown, count, spread)

    def predictive_safety(self, slot, horizon=75, predict_fire=True):
        out = (ctypes.c_float * 9)()
        self.lib.kf_duel24_predictive_safety(
            self.handle, slot, horizon, 1 if predict_fire else 0, out,
        )
        return np.ctypeslib.as_array(out).copy()

    def make_search(self, slot, seat=0, branches=ACTIONS, threads=0):
        handle = self.lib.kf_duel24_search_snapshot(
            self.handle, slot, seat, branches, threads,
        )
        if not handle:
            raise RuntimeError("search snapshot failed (slot not frozen or invalid)")
        return DuelSearch(self.lib, handle, branches)

    def close(self):
        if getattr(self, "handle", None):
            self.lib.kf_duel24_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


class DuelSearch:
    def __init__(self, lib, handle, branches):
        self.lib = lib
        self.handle = handle
        self.branches = branches
        self.obs = self._view("kf_duel24_search_obs", ctypes.c_float, (branches, OBS_DIM))
        self.masks = self._view("kf_duel24_search_masks", ctypes.c_uint8, (branches, BULLET_SLOTS))
        self.obs_opponent = self._view(
            "kf_duel24_search_opponent_obs", ctypes.c_float, (branches, OBS_DIM),
        )
        self.masks_opponent = self._view(
            "kf_duel24_search_opponent_masks", ctypes.c_uint8, (branches, BULLET_SLOTS),
        )
        self.rewards = self._view("kf_duel24_search_rewards", ctypes.c_float, (branches,))
        self.dones = self._view("kf_duel24_search_dones", ctypes.c_uint8, (branches,))
        self.outcomes = self._view("kf_duel24_search_outcomes", ctypes.c_uint8, (branches,))
        self.frames = self._view("kf_duel24_search_frames", ctypes.c_uint32, (branches,))
        self._buffers = (None, None)

    def _view(self, name, ctype, shape):
        function = getattr(self.lib, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.POINTER(ctype)
        return np.ctypeslib.as_array(function(self.handle), shape=shape)

    def step(self, actions, opponent_actions):
        actions = np.ascontiguousarray(actions, np.uint16)
        opponent_actions = np.ascontiguousarray(opponent_actions, np.uint16)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        self._buffers = (actions, opponent_actions)
        self.lib.kf_duel24_search_step(
            self.handle, actions.ctypes.data_as(u16), opponent_actions.ctypes.data_as(u16),
        )

    def close(self):
        if getattr(self, "handle", None):
            self.lib.kf_duel24_search_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
