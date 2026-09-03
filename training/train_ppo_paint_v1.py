"""PPO curricula for fixed-map locomotion, pursuit, and static-target combat."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ppo_models import (
    ACTION_COUNT, BULLET_SLOTS, FIRE_ACTIONS, MAP_DIM, OBS_DIM,
    NAV_OFFSET, OBS_SCHEMA_VERSION, STOP_ACTIONS, make_actor_critic,
)


CHANNEL_NAMES = (
    "initiative", "navigation", "terminal", "kill_quality", "shot_attempt",
    "precision", "knife", "dodge", "repeated_wall",
)
TRAIN_SEEDS = (11, 22, 33)
TRAIN_ENV_BASE = 1_100_000
EVAL_ENV_BASE = 2_100_000


def atomic_torch_save(value, path):
    """Write checkpoints without leaving a half-written model on interruption."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class Config:
    stage: str = "static-target-fixed-v1-joystick130"
    observation_schema: int = OBS_SCHEMA_VERSION
    action_count: int = ACTION_COUNT
    opponent: str = "Laika"
    training_opponent: str = "inert-fixed-seed-20260824"
    episode_frames: int = 750
    navigation_total: float = 0.5
    success_base: float = 10.0
    speed_bonus_max: float = 2.0
    failure_reward: float = -10.0
    shot_attempt_reward: float = 0.10
    shot_attempt_cap: float = 0.50
    map_name: str = "static-target-fixed-seed-20260824"
    step_cost: float = 0.0
    failure_rules: str = "self-death,double-death,timeout=-10"
    initial_checkpoint: str = ""
    actor_logit_scale_on_init: float = 1.0
    direction_pretrain_epochs: int = 0
    direction_pretrain_learning_rate: float = 1e-4
    envs: int = 64
    rollout_steps: int = 256
    total_steps: int = 5_000_000
    epochs: int = 4
    minibatch_steps: int = 1024
    sequence_length: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.9975
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    eval_every_updates: int = 0
    eval_episodes: int = 100
    eval_envs: int = 32


class PpoVec:
    def __init__(self, count: int, seed: int, static_target=False, walking=False,
                 walking_training=False, walking_map=1, pursuit=False,
                 hunt=False, hunt_map="mixed", eval_laika=False,
                 library=Path("engine/target/release/libkf_engine.dylib")):
        self.count = count
        self.lib = ctypes.CDLL(str(library.resolve()))
        constructor_name = (
            {"real": "kf_vec_new_hunt_v1", "room": "kf_vec_new_hunt_room_v1",
             "mixed": "kf_vec_new_hunt_mixed_v1"}[hunt_map] if hunt else
            "kf_vec_new_pursuit_v1" if pursuit else
            "kf_vec_new_walking_train_v3" if walking_training else
            f"kf_vec_new_walking_v{walking_map}" if walking else
            "kf_vec_new_static_target_v1" if static_target else
            "kf_vec_new_ppo_eval" if eval_laika else
            "kf_vec_new_ppo_paint_v1"
        )
        constructor = getattr(self.lib, constructor_name)
        fixed_map = (static_target or walking or pursuit or hunt) and not walking_training
        constructor.argtypes = [ctypes.c_uint32] if fixed_map else [ctypes.c_uint32, ctypes.c_uint32]
        constructor.restype = ctypes.c_void_p
        self.lib.kf_vec_obs_dim.restype = ctypes.c_uint32
        self.lib.kf_vec_bullet_slots.restype = ctypes.c_uint32
        self.lib.kf_vec_reward_channel_count.restype = ctypes.c_uint32
        native = (
            int(self.lib.kf_vec_obs_dim()), int(self.lib.kf_vec_bullet_slots()),
            int(self.lib.kf_vec_reward_channel_count()),
        )
        expected = (OBS_DIM, BULLET_SLOTS, len(CHANNEL_NAMES))
        if native != expected:
            raise RuntimeError(f"native/Python schema mismatch: {native} != {expected}")
        self.handle = constructor(count) if fixed_map else constructor(count, seed)
        if not self.handle:
            raise RuntimeError("kf_vec_new_ppo_paint_v1 failed")
        self.lib.kf_vec_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)]
        self.lib.kf_vec_reset_done.argtypes = [ctypes.c_void_p]
        self.lib.kf_vec_free.argtypes = [ctypes.c_void_p]
        self.obs = self._view("kf_vec_obs", ctypes.c_float, (count, OBS_DIM))
        self.masks = self._view("kf_vec_masks", ctypes.c_uint8, (count, BULLET_SLOTS))
        self.rewards = self._view("kf_vec_rewards", ctypes.c_float, (count,))
        self.channels = self._view(
            "kf_vec_reward_channels", ctypes.c_float, (count, len(CHANNEL_NAMES))
        )
        self.diagnostics = self._view("kf_vec_reward_diagnostics", ctypes.c_float, (count, 3))
        self.dones = self._view("kf_vec_dones", ctypes.c_uint8, (count,))
        self.terminals = self._view("kf_vec_terminals", ctypes.c_uint8, (count,))
        self.winners = self._view("kf_vec_winners", ctypes.c_int8, (count,))
        self.walking_failure_reasons = self._view(
            "kf_vec_walking_failure_reasons", ctypes.c_int8, (count,)
        )

    def _view(self, name, ctype, shape):
        function = getattr(self.lib, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.POINTER(ctype)
        return np.ctypeslib.as_array(function(self.handle), shape=shape)

    def step(self, actions):
        actions = np.asarray(actions, np.uint16)
        self.lib.kf_vec_step(
            self.handle, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        )

    def reset_done(self):
        self.lib.kf_vec_reset_done(self.handle)

    def close(self):
        if self.handle:
            self.lib.kf_vec_free(self.handle)
            self.handle = None


def tensors(obs, masks, device):
    return (
        torch.as_tensor(obs, dtype=torch.float32, device=device),
        torch.as_tensor(masks, dtype=torch.bool, device=device),
    )


@torch.inference_mode()
def collect_rollout(env, model, kind, config, device, starts, hidden):
    tmax, count = config.rollout_steps, config.envs
    shape = (tmax, count)
    obs = np.empty(shape + (OBS_DIM,), np.float32)
    masks = np.empty(shape + (BULLET_SLOTS,), bool)
    actions = np.empty(shape, np.int64)
    logprob = np.empty(shape, np.float32)
    values = np.empty(shape, np.float32)
    next_values = np.empty(shape, np.float32)
    rewards = np.empty(shape, np.float32)
    channels = np.empty(shape + (len(CHANNEL_NAMES),), np.float32)
    dones = np.empty(shape, bool)
    terminals = np.empty(shape, bool)
    episode_starts = np.empty(shape, bool)
    diagnostics = np.empty(shape + (3,), np.float32)
    hidden_in = np.empty(shape + (256,), np.float32) if kind == "gru" else None
    outcome_counts = {"win": 0, "loss": 0, "double": 0, "timeout": 0}

    for t in range(tmax):
        current_obs = env.obs.copy()
        current_masks = env.masks.astype(bool, copy=True)
        obs[t], masks[t], episode_starts[t] = current_obs, current_masks, starts
        if kind == "gru":
            hidden = hidden * torch.as_tensor(
                ~starts, device=device, dtype=torch.float32
            ).view(1, count, 1)
            hidden_in[t] = hidden[0].cpu().numpy()
        obs_t, mask_t = tensors(current_obs, current_masks, device)
        logits, value, next_hidden = model.step(obs_t, mask_t, hidden)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        actions[t] = action.cpu().numpy()
        logprob[t] = distribution.log_prob(action).cpu().numpy()
        values[t] = value.cpu().numpy()

        env.step(actions[t])
        rewards[t] = env.rewards
        channels[t] = env.channels
        diagnostics[t] = env.diagnostics
        dones[t] = env.dones.astype(bool)
        terminals[t] = env.terminals.astype(bool)
        for winner in env.winners[dones[t]]:
            key = "win" if winner == 0 else "loss" if winner == 1 else "double" if winner == -1 else "timeout"
            outcome_counts[key] += 1

        next_obs_t, next_mask_t = tensors(env.obs.copy(), env.masks.astype(bool, copy=True), device)
        bootstrap_hidden = next_hidden.clone() if next_hidden is not None else None
        _, bootstrap, _ = model.step(next_obs_t, next_mask_t, bootstrap_hidden)
        next_values[t] = bootstrap.cpu().numpy()
        env.reset_done()
        starts = dones[t].copy()
        hidden = next_hidden

    advantages = np.zeros_like(rewards)
    carry = np.zeros(count, np.float32)
    for t in range(tmax - 1, -1, -1):
        delta = rewards[t] + config.gamma * next_values[t] * (~terminals[t]) - values[t]
        carry = delta + config.gamma * config.gae_lambda * (~dones[t]) * carry
        advantages[t] = carry
    returns = advantages + values
    batch = {
        "obs": obs, "masks": masks, "actions": actions, "logprob": logprob,
        "values": values, "advantages": advantages, "returns": returns,
        "episode_starts": episode_starts, "hidden_in": hidden_in,
    }
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "fire_rate": float((actions % 2 == 1).mean()),
        "stop_rate": float(np.isin(actions, STOP_ACTIONS).mean()),
        "done_count": int(dones.sum()),
        "outcomes": outcome_counts,
        "phi_self_mean": float(diagnostics[..., 0].mean()),
        "phi_enemy_mean": float(diagnostics[..., 1].mean()),
        "phi_difference_mean": float(diagnostics[..., 2].mean()),
        "channel_mean": {
            name: float(channels[..., i].mean()) for i, name in enumerate(CHANNEL_NAMES)
        },
        "channel_sum": {
            name: float(channels[..., i].sum()) for i, name in enumerate(CHANNEL_NAMES)
        },
    }
    return batch, metrics, starts, hidden


def ppo_loss(model, batch, indices, config, device, recurrent):
    if not recurrent:
        flat_obs = batch["obs"].reshape(-1, OBS_DIM)[indices]
        flat_masks = batch["masks"].reshape(-1, BULLET_SLOTS)[indices]
        logits, value, _ = model.step(*tensors(flat_obs, flat_masks, device))
        action = torch.as_tensor(
            batch["actions"].reshape(-1)[indices], device=device
        ).long()
        pick = lambda name: torch.as_tensor(batch[name].reshape(-1)[indices], device=device)
    else:
        env_indices, starts = indices
        length = config.sequence_length
        obs = np.stack([batch["obs"][start:start + length, env] for env, start in zip(env_indices, starts)])
        masks = np.stack([batch["masks"][start:start + length, env] for env, start in zip(env_indices, starts)])
        episode_starts = np.stack([
            batch["episode_starts"][start:start + length, env]
            for env, start in zip(env_indices, starts)
        ])
        initial_hidden = np.stack([
            batch["hidden_in"][start, env] for env, start in zip(env_indices, starts)
        ])
        obs_t, masks_t = tensors(obs, masks, device)
        hidden_t = torch.as_tensor(initial_hidden, device=device).unsqueeze(0)
        starts_t = torch.as_tensor(episode_starts, device=device, dtype=torch.bool)
        logits, value, _ = model.sequence(obs_t, masks_t, hidden_t, starts_t)
        logits = logits.flatten(0, 1)
        value = value.flatten()
        action = torch.as_tensor(np.stack([
            batch["actions"][start:start + length, env]
            for env, start in zip(env_indices, starts)
        ]).reshape(-1), device=device).long()
        pick = lambda name: torch.as_tensor(np.stack([
            batch[name][start:start + length, env] for env, start in zip(env_indices, starts)
        ]).reshape(-1), device=device)

    old_logprob = pick("logprob")
    old_value = pick("values")
    advantage = pick("advantages")
    returns = pick("returns")
    advantage = (advantage - advantage.mean()) / advantage.std().clamp(min=1e-8)
    distribution = Categorical(logits=logits)
    new_logprob = distribution.log_prob(action)
    ratio = (new_logprob - old_logprob).exp()
    policy_loss = -torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * advantage,
    ).mean()
    clipped_value = old_value + (value - old_value).clamp(-config.value_clip, config.value_clip)
    value_loss = 0.5 * torch.maximum(
        (value - returns).square(), (clipped_value - returns).square()
    ).mean()
    entropy = distribution.entropy().mean()
    loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
    with torch.no_grad():
        log_ratio = new_logprob - old_logprob
        approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > config.clip).float().mean()
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
    }


def update_policy(model, optimiser, batch, kind, config, device, rng):
    aggregates = []
    stop_early = False
    if kind == "nomem":
        total = config.rollout_steps * config.envs
        units = np.arange(total)
        batches = lambda: (
            units[start:start + config.minibatch_steps]
            for start in range(0, total, config.minibatch_steps)
        )
    else:
        assert config.rollout_steps % config.sequence_length == 0
        pairs = np.asarray([
            (env, start) for env in range(config.envs)
            for start in range(0, config.rollout_steps, config.sequence_length)
        ], np.int64)
        sequences_per_batch = config.minibatch_steps // config.sequence_length
        batches = lambda: (
            (pairs[start:start + sequences_per_batch, 0], pairs[start:start + sequences_per_batch, 1])
            for start in range(0, len(pairs), sequences_per_batch)
        )
        units = pairs

    for _epoch in range(config.epochs):
        rng.shuffle(units)
        for indices in batches():
            optimiser.zero_grad(set_to_none=True)
            loss, metrics = ppo_loss(model, batch, indices, config, device, kind == "gru")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimiser.step()
            metrics["grad_norm"] = float(grad_norm)
            aggregates.append(metrics)
            if metrics["approx_kl"] > config.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    return {
        key: float(np.mean([row[key] for row in aggregates]))
        for key in aggregates[0]
    } | {"early_stop_kl": stop_early, "minibatches": len(aggregates)}


@torch.inference_mode()
def evaluate(model, kind, config, device):
    env = PpoVec(config.eval_envs, EVAL_ENV_BASE, eval_laika=True)
    hidden = model.initial_hidden(config.eval_envs, device)
    starts = np.ones(config.eval_envs, bool)
    channel_total = np.zeros((config.eval_envs, len(CHANNEL_NAMES)), np.float64)
    initial_phi = np.full(config.eval_envs, np.nan)
    max_phi = np.full(config.eval_envs, -np.inf)
    min_path = np.full(config.eval_envs, np.inf)
    fired = np.zeros(config.eval_envs, bool)
    decisions = np.zeros(config.eval_envs, np.int64)
    episodes = []
    try:
        while len(episodes) < config.eval_episodes:
            current_obs = env.obs.copy()
            current_masks = env.masks.astype(bool, copy=True)
            min_path = np.minimum(min_path, current_obs[:, MAP_DIM])
            if kind == "gru":
                hidden = hidden * torch.as_tensor(
                    ~starts, device=device, dtype=torch.float32
                ).view(1, config.eval_envs, 1)
            logits, _value, hidden = model.step(*tensors(current_obs, current_masks, device), hidden)
            action = logits.argmax(-1).cpu().numpy().astype(np.uint16)
            fired |= action % 2 == 1
            decisions += 1
            env.step(action)
            channel_total += env.channels
            phi = env.diagnostics[:, 0]
            first = np.isnan(initial_phi)
            initial_phi[first] = phi[first]
            max_phi = np.maximum(max_phi, phi)
            done = env.dones.astype(bool).copy()
            for index in np.flatnonzero(done):
                winner = int(env.winners[index])
                episodes.append({
                    "outcome": "win" if winner == 0 else "loss" if winner == 1 else "double" if winner == -1 else "timeout",
                    "initiative_gain": float(max_phi[index] - initial_phi[index]),
                    "close_contact": bool(min_path[index] <= 4.0 / 22.0),
                    "fired": bool(fired[index]),
                    "decisions": int(decisions[index]),
                    "channels": channel_total[index].copy(),
                })
                channel_total[index] = 0
                initial_phi[index] = np.nan
                max_phi[index] = -np.inf
                min_path[index] = np.inf
                fired[index] = False
                decisions[index] = 0
            env.reset_done()
            starts = done
    finally:
        env.close()
    episodes = episodes[:config.eval_episodes]
    outcomes = {key: sum(row["outcome"] == key for row in episodes)
                for key in ("win", "loss", "double", "timeout")}
    gains = np.asarray([row["initiative_gain"] for row in episodes])
    episode_channels = np.stack([row["channels"] for row in episodes])
    return {
        "episodes": len(episodes), "outcomes": outcomes,
        "win_rate": outcomes["win"] / len(episodes),
        "initiative_gain_mean": float(gains.mean()),
        "initiative_gain_positive_rate": float((gains >= 0.05).mean()),
        "close_contact_rate": float(np.mean([row["close_contact"] for row in episodes])),
        "episode_fire_rate": float(np.mean([row["fired"] for row in episodes])),
        "decisions_mean": float(np.mean([row["decisions"] for row in episodes])),
        "channel_episode_mean": {
            name: float(episode_channels[:, i].mean()) for i, name in enumerate(CHANNEL_NAMES)
        },
    }


@torch.inference_mode()
def evaluate_hunt(model, kind, config, device, hunt_map="real"):
    """Exactly 100 deterministic episodes on one hunt map.

    The fixed-Laika report is a side channel; this is the metric that says
    whether the policy actually learned to run the maze down and shoot. The
    two maps are reported separately: an average over the mix would hide the
    easy half propping up the hard one.
    """
    env = PpoVec(config.eval_envs, 0, hunt=True, hunt_map=hunt_map)
    hidden = model.initial_hidden(config.eval_envs, device)
    starts = np.ones(config.eval_envs, bool)
    decisions = np.zeros(config.eval_envs, np.int64)
    fired = np.zeros(config.eval_envs, bool)
    bfs_sum = np.zeros(config.eval_envs, np.float64)
    bfs_count = np.zeros(config.eval_envs, np.int64)
    bfs_min = np.full(config.eval_envs, np.inf, np.float64)
    episodes = []
    try:
        while len(episodes) < config.eval_episodes:
            obs = env.obs.copy()
            masks = env.masks.astype(bool, copy=True)
            if kind == "gru":
                hidden = hidden * torch.as_tensor(
                    ~starts, device=device, dtype=torch.float32
                ).view(1, config.eval_envs, 1)
            logits, _value, hidden = model.step(*tensors(obs, masks, device), hidden)
            action = logits.argmax(-1).cpu().numpy().astype(np.uint16)
            fired |= action % 2 == 1
            decisions += 1
            env.step(action)
            bfs = env.obs[:, NAV_OFFSET + 4].astype(np.float64) * 22.0
            settled = decisions % 4 == 0
            bfs_sum[settled] += bfs[settled]
            bfs_count[settled] += 1
            bfs_min[settled] = np.minimum(bfs_min[settled], bfs[settled])
            reward = env.rewards.copy()
            done = env.dones.astype(bool).copy()
            for index in np.flatnonzero(done):
                winner = int(env.winners[index])
                reason = int(env.walking_failure_reasons[index])
                terminal = bool(env.terminals[index])
                if winner == 0:
                    outcome = "kill"
                elif not terminal:
                    outcome = "truncated"
                elif reason in (1, 5):
                    outcome = "rule_failure"
                else:
                    outcome = "suicide"
                episodes.append({
                    "outcome": outcome,
                    "failure_reason": {1: "wall_or_slide", 5: "stop_or_stuck"}.get(reason, "none"),
                    "decisions": int(decisions[index]),
                    "fired": bool(fired[index]),
                    "final_reward": float(reward[index]),
                    "mean_bfs": float(bfs_sum[index] / max(bfs_count[index], 1)),
                    "min_bfs": float(bfs_min[index]) if np.isfinite(bfs_min[index]) else 0.0,
                })
                decisions[index] = 0
                fired[index] = False
                bfs_sum[index] = 0.0
                bfs_count[index] = 0
                bfs_min[index] = np.inf
            starts = done
            env.reset_done()
    finally:
        env.close()
    episodes = episodes[:config.eval_episodes]
    counts = {k: 0 for k in ("kill", "suicide", "rule_failure", "truncated")}
    for row in episodes:
        counts[row["outcome"]] += 1
    reasons = {}
    for row in episodes:
        if row["outcome"] == "rule_failure":
            reasons[row["failure_reason"]] = reasons.get(row["failure_reason"], 0) + 1
    return {
        "episodes": len(episodes),
        "outcomes": counts,
        "kill_rate": counts["kill"] / max(len(episodes), 1),
        "rule_failure_reasons": reasons,
        "mean_decisions": float(np.mean([r["decisions"] for r in episodes])),
        "episode_fire_rate": float(np.mean([r["fired"] for r in episodes])),
        "mean_bfs": float(np.mean([r["mean_bfs"] for r in episodes])),
        "min_bfs_mean": float(np.mean([r["min_bfs"] for r in episodes])),
    }


@torch.inference_mode()
def evaluate_walking(model, kind, config, device, walking_map=1, pursuit=False):
    """Exactly 100 deterministic episodes on the curriculum acceptance map."""
    env = PpoVec(
        config.eval_envs, 0, walking=not pursuit,
        walking_map=walking_map, pursuit=pursuit,
    )
    hidden = model.initial_hidden(config.eval_envs, device)
    starts = np.ones(config.eval_envs, bool)
    decisions = np.zeros(config.eval_envs, np.int64)
    fired = np.zeros(config.eval_envs, bool)
    bfs_sum = np.zeros(config.eval_envs, np.float64)
    bfs_count = np.zeros(config.eval_envs, np.int64)
    bfs_min = np.full(config.eval_envs, np.inf, np.float64)
    bfs_final = np.zeros(config.eval_envs, np.float64)
    episodes = []
    try:
        while len(episodes) < config.eval_episodes:
            obs = env.obs.copy()
            masks = env.masks.astype(bool, copy=True)
            if kind == "gru":
                hidden = hidden * torch.as_tensor(
                    ~starts, device=device, dtype=torch.float32
                ).view(1, config.eval_envs, 1)
            logits, _value, hidden = model.step(*tensors(obs, masks, device), hidden)
            action = logits.argmax(-1).cpu().numpy().astype(np.uint16)
            fired |= action % 2 == 1
            decisions += 1
            env.step(action)
            if pursuit:
                bfs = env.obs[:, NAV_OFFSET + 4].astype(np.float64) * 22.0
                settled = decisions % 4 == 0
                bfs_sum[settled] += bfs[settled]
                bfs_count[settled] += 1
                bfs_min[settled] = np.minimum(bfs_min[settled], bfs[settled])
                bfs_final[:] = bfs
            done = env.dones.astype(bool).copy()
            for index in np.flatnonzero(done):
                winner = int(env.winners[index])
                reason = int(env.walking_failure_reasons[index])
                row = {
                    "outcome": (
                        "failed" if pursuit and winner == 1 else
                        "completed" if pursuit else
                        "arrived" if winner == 0 else
                        "failed" if winner == 1 else "timeout"
                    ),
                    "failure_reason": {1: "wall", 2: "heading", 3: "fire", 4: "timeout", 5: "stop", 6: "route_direction"}.get(reason, "none"),
                    "decisions": int(decisions[index]),
                    "fired": bool(fired[index]),
                }
                if pursuit:
                    count = max(int(bfs_count[index]), 1)
                    row.update({
                        "bfs_mean": float(bfs_sum[index] / count),
                        "bfs_min": float(bfs_min[index]) if bfs_count[index] else float(bfs_final[index]),
                        "bfs_final": float(bfs_final[index]),
                    })
                episodes.append(row)
                decisions[index] = 0
                fired[index] = False
                bfs_sum[index] = 0.0
                bfs_count[index] = 0
                bfs_min[index] = np.inf
                bfs_final[index] = 0.0
            env.reset_done()
            starts = done
    finally:
        env.close()
    episodes = episodes[:config.eval_episodes]
    outcome_keys = ("completed", "failed") if pursuit else ("arrived", "failed", "timeout")
    outcomes = {key: sum(row["outcome"] == key for row in episodes)
                for key in outcome_keys}
    failure_reasons = {key: sum(row["failure_reason"] == key for row in episodes)
                       for key in ("wall", "heading", "fire", "stop", "route_direction", "timeout")}
    result = {
        "episodes": len(episodes),
        "map": (
            "walking-v1-upper-right-room-irregular-laika-seed-20260827"
            if pursuit else
            "walking-v2-seven-by-four-five-turn-seed-20260826"
            if walking_map == 2 else
            "walking-v1-six-by-three-serpentine-seed-20260825"
        ),
        "outcomes": outcomes,
        "failure_reasons": failure_reasons,
        "fire_rate": float(np.mean([row["fired"] for row in episodes])),
        "decisions_mean": float(np.mean([row["decisions"] for row in episodes])),
    }
    if pursuit:
        result.update({
            "completion_rate": outcomes["completed"] / len(episodes),
            "bfs_mean": float(np.mean([row["bfs_mean"] for row in episodes])),
            "bfs_min_mean": float(np.mean([row["bfs_min"] for row in episodes])),
            "bfs_final_mean": float(np.mean([row["bfs_final"] for row in episodes])),
        })
    else:
        result["arrival_rate"] = outcomes["arrived"] / len(episodes)
    return result


def append_jsonl(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def select_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    candidate = torch.device("mps")
    if choice == "mps" or torch.backends.mps.is_available():
        try:
            torch.zeros(1, device=candidate)
            torch.mps.synchronize()
            return candidate
        except RuntimeError:
            if choice == "mps":
                raise
            print("MPS 报告可用但实际初始化失败；本次回退 CPU", flush=True)
    return torch.device("cpu")


def pretrain_pursuit_direction(model, kind, optimiser, config, device):
    """Teach the frozen route-action mapping on one exact 300-frame oracle trace."""
    if kind != "nomem":
        raise ValueError("the pursuit direction warm-up currently requires nomem")
    env = PpoVec(1, 0, pursuit=True)
    observations, masks, targets = [], [], []
    previous_action = 90
    try:
        for frame in range(config.episode_frames):
            observation = env.obs.copy()
            route = observation[0, NAV_OFFSET:NAV_OFFSET + 4]
            if route.max() > 0.5:
                previous_action = int(route.argmax()) * 90
            observations.append(observation[0])
            masks.append(env.masks[0].copy())
            targets.append(previous_action)
            env.step(np.asarray([previous_action], np.uint16))
            if env.dones[0] and frame + 1 != config.episode_frames:
                reason = int(env.walking_failure_reasons[0])
                raise RuntimeError(f"oracle failed at frame {frame + 1}, reason {reason}")
    finally:
        env.close()
    observations = np.asarray(observations, np.float32)
    masks = np.asarray(masks, bool)
    targets = torch.as_tensor(targets, device=device, dtype=torch.long)
    rng = np.random.default_rng(20_260_827)
    model.train()
    for epoch in range(config.direction_pretrain_epochs):
        order = rng.permutation(len(observations))
        for start in range(0, len(order), 64):
            indices = order[start:start + 64]
            optimiser.zero_grad(set_to_none=True)
            logits, _value, _hidden = model.step(
                *tensors(observations[indices], masks[indices], device),
                model.initial_hidden(len(indices), device),
            )
            loss = F.cross_entropy(logits, targets[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimiser.step()
    model.eval()
    with torch.no_grad():
        logits, _value, _hidden = model.step(
            *tensors(observations, masks, device),
            model.initial_hidden(len(observations), device),
        )
        accuracy = float((logits.argmax(-1) == targets).float().mean())
    print(
        f"pursuit direction pretrain: {len(observations)} frames, "
        f"{config.direction_pretrain_epochs} epochs, accuracy={accuracy:.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("nomem", "gru"))
    parser.add_argument("--seed", required=True, type=int, choices=TRAIN_SEEDS)
    parser.add_argument(
        "--curriculum", choices=("static-target", "walking", "pursuit", "hunt"),
        default="static-target",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    device = select_device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = Config()
    walking = args.curriculum == "walking"
    pursuit = args.curriculum == "pursuit"
    hunt = args.curriculum == "hunt"
    locomotion = walking or pursuit
    if walking:
        config = Config(
            stage="walking-v6-transition-context-joystick130",
            training_opponent="inert-walking-goal-varied-starts-same-map-seed-20260825",
            episode_frames=300,
            navigation_total=5.0,
            success_base=10.0,
            speed_bonus_max=0.0,
            failure_reward=-10.0,
            shot_attempt_reward=0.0,
            shot_attempt_cap=0.0,
            map_name="walking-v1-six-by-three-serpentine-seed-20260825",
            step_cost=-0.002,
            failure_rules="wall-or-slide,no-displacement,route-direction-mismatch,displacement-heading-mismatch,fire=-10;timeout=-10",
            initial_checkpoint="outputs/ppo_walking_v5_waypoint_direction_joystick130/nomem/s11/final.pt",
            learning_rate=1e-4,
            total_steps=500_000,
        )
    elif pursuit:
        config = Config(
            stage="pursuit-v11-room-exp-bfs-joystick130",
            training_opponent="unarmed-laika-irregular-two-dimensional-room-patrol",
            episode_frames=300,
            navigation_total=0.0,
            success_base=0.0,
            speed_bonus_max=0.0,
            failure_reward=-10.0,
            shot_attempt_reward=0.0,
            shot_attempt_cap=0.0,
            map_name="walking-v1-upper-right-two-by-two-room-seed-20260827",
            step_cost=0.0,
            failure_rules="wall-or-slide,no-displacement,fire=-10,stop=-10;forward-only-wheel-no-reverse;300-frame-horizon=truncation",
            initial_checkpoint="outputs/ppo_walking_v6_transition_context_joystick130/nomem/s11/final.pt",
            actor_logit_scale_on_init=0.5,
            direction_pretrain_epochs=200,
            learning_rate=1e-9,
            total_steps=16_384,
            entropy_coefficient=0.0,
        )
    elif hunt:
        config = Config(
            stage="hunt-v3-mixed-selfclosed-bfs-joystick258",
            training_opponent="50-50-room-patrol-waypoints-and-scripted-random-walk-real-maze",
            episode_frames=750,
            navigation_total=0.0,
            success_base=50.0,
            speed_bonus_max=0.0,
            failure_reward=-300.0,
            shot_attempt_reward=0.0,
            shot_attempt_cap=0.0,
            map_name="50-50-mix:pursuit-room-seed-20260827-and-real-maze-seed-20260862",
            step_cost=0.0,
            failure_rules=(
                "stop=-300,wall-or-slide=-300,no-displacement=-300;"
                "kill-target=+50;own-ricochet-death=-300;"
                "approach=+1-per-cell-the-policy-itself-closes-settled-every-frame;"
                "fire-is-legal;750-frame-horizon=truncation"
            ),
            initial_checkpoint="",
            learning_rate=3e-4,
            total_steps=5_000_000,
        )
    if args.smoke:
        config = replace(
            config,
            total_steps=config.envs * config.rollout_steps * 2,
            eval_every_updates=0, eval_episodes=100,
        )
    output_root = args.output or Path(
        "outputs/ppo_hunt_v3_mixed_selfclosed_joystick258" if hunt
        else "outputs/ppo_pursuit_v11_room_exp_bfs_joystick130" if pursuit
        else "outputs/ppo_walking_v6_transition_context_joystick130" if walking
        else "outputs/ppo_static_target_fixed_v1_joystick130"
    )
    output = output_root / args.model / f"s{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    if config.actor_logit_scale_on_init == 1.0:
        config_dict.pop("actor_logit_scale_on_init")
    config_dict |= {"model": args.model, "seed": args.seed, "device": str(device)}
    if pursuit:
        config_dict |= {
            "proximity_reward": "every-4-frames:-exp(current_bfs-initial_bfs)",
            "success_reward": "none; reaching Laika never ends the episode",
            "target_motion": "irregular-horizontal-vertical-diagonal-patrol-in-upper-right-two-by-two-room",
        }
    config_path = output / "config.json"
    if (config_path.exists() and json.loads(config_path.read_text()) != config_dict
            and not args.reevaluate):
        raise RuntimeError(f"refusing incompatible resume at {output}")
    config_path.write_text(json.dumps(config_dict, indent=2, ensure_ascii=False))
    if args.reevaluate:
        final_path = output / "final.pt"
        saved = torch.load(final_path, map_location=device, weights_only=False)
        model = make_actor_critic(args.model).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        result = saved["result"]
        result["config"] = config_dict
        result["evaluation"] = evaluate(model, args.model, config, device)
        if locomotion:
            result["curriculum_evaluation"] = evaluate_walking(
                model, args.model, config, device, pursuit=pursuit
            )
        (output / "complete.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        atomic_torch_save({"model": model.state_dict(), "result": result}, final_path)
        print(json.dumps(result["evaluation"], ensure_ascii=False), flush=True)
        return
    if (output / "complete.json").exists():
        print(f"已完成，跳过 {output}")
        return

    model = make_actor_critic(args.model).to(device)
    if locomotion:
        with torch.no_grad():
            model.actor.bias[FIRE_ACTIONS] = -4.0
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    start_update = 0
    total_steps = 0
    checkpoint = output / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        start_update = saved["update"] + 1
        total_steps = saved["total_steps"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "mps" and saved.get("device_rng") is not None:
            torch.mps.set_rng_state(saved["device_rng"])
    else:
        init_checkpoint = args.init_checkpoint or (
            Path(config.initial_checkpoint) if config.initial_checkpoint else None
        )
        if init_checkpoint is not None:
            saved = torch.load(init_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            if config.actor_logit_scale_on_init != 1.0:
                with torch.no_grad():
                    model.actor.weight.mul_(config.actor_logit_scale_on_init)
                    model.actor.bias.mul_(config.actor_logit_scale_on_init)
                    model.actor.bias[FIRE_ACTIONS] = -8.0
                    model.actor.bias[STOP_ACTIONS] = -8.0
        if pursuit and config.direction_pretrain_epochs:
            pretrain_optimiser = torch.optim.Adam(
                model.parameters(),
                lr=config.direction_pretrain_learning_rate,
                eps=1e-5,
            )
            pretrain_pursuit_direction(
                model, args.model, pretrain_optimiser, config, device
            )
            # The supervised direction stage must not leak Adam moments into PPO.
            optimiser = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, eps=1e-5
            )

    steps_per_update = config.envs * config.rollout_steps
    updates = math.ceil(config.total_steps / steps_per_update)
    env_seed = TRAIN_ENV_BASE + args.seed * 10_000 + start_update * config.envs * 100
    env = PpoVec(
        config.envs,
        env_seed,
        static_target=not (locomotion or hunt),
        walking_training=walking,
        pursuit=pursuit,
        hunt=hunt,
        hunt_map="mixed",
    )
    starts = np.ones(config.envs, bool)
    hidden = model.initial_hidden(config.envs, device)
    started = time.perf_counter()
    try:
        for update in range(start_update, updates):
            model.eval()
            rollout, rollout_metrics, starts, hidden = collect_rollout(
                env, model, args.model, config, device, starts, hidden
            )
            model.train()
            update_metrics = update_policy(
                model, optimiser, rollout, args.model, config, device,
                np.random.default_rng(args.seed * 1_000_000 + update),
            )
            total_steps += steps_per_update
            elapsed = time.perf_counter() - started
            record = {
                "update": update, "total_steps": total_steps,
                "steps_per_second_this_run": (total_steps - start_update * steps_per_update) / elapsed,
                "rollout": rollout_metrics, "ppo": update_metrics,
            }
            if config.eval_every_updates and (update + 1) % config.eval_every_updates == 0:
                model.eval()
                record["evaluation"] = evaluate(model, args.model, config, device)
                print(
                    f"{args.model} s{args.seed} u{update + 1}/{updates} "
                    f"steps={total_steps:,} win={record['evaluation']['win_rate']:.3f} "
                    f"engage={record['evaluation']['initiative_gain_positive_rate']:.3f} "
                    f"fire={record['evaluation']['episode_fire_rate']:.3f}", flush=True,
                )
            else:
                print(
                    f"{args.model} s{args.seed} u{update + 1}/{updates} "
                    f"steps={total_steps:,} reward={rollout_metrics['reward_mean']:.5f} "
                    f"H={update_metrics['entropy']:.3f}", flush=True,
                )
            append_jsonl(output / "metrics.jsonl", record)
            atomic_torch_save({
                "model": model.state_dict(), "optimiser": optimiser.state_dict(),
                "update": update, "total_steps": total_steps,
                "torch_rng": torch.get_rng_state(), "config": config_dict,
                "device_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
            }, checkpoint)
    finally:
        env.close()
    model.eval()
    final_eval = evaluate(model, args.model, config, device)
    result = {
        "name": f"ppo-{config.stage}-{args.model}-s{args.seed}",
        "model": args.model, "seed": args.seed, "total_steps": total_steps,
        "seconds_this_run": time.perf_counter() - started,
        "evaluation": final_eval, "config": config_dict,
    }
    if locomotion:
        result["curriculum_evaluation"] = evaluate_walking(
            model, args.model, config, device, pursuit=pursuit
        )
    if hunt:
        result["curriculum_evaluation"] = {
            "room": evaluate_hunt(model, args.model, config, device, hunt_map="room"),
            "real_maze": evaluate_hunt(model, args.model, config, device, hunt_map="real"),
        }
    (output / "complete.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    atomic_torch_save({"model": model.state_dict(), "result": result}, output / "final.pt")


if __name__ == "__main__":
    main()
