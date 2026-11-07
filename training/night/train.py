from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel_env import (
    ACTIONS,
    BULLET_SLOTS,
    DuelVec,
    OBS_DIM,
    OPPONENT_FROZEN,
    OPPONENT_LAIKA,
    OPPONENT_MPC,
)
from hybrid_model import HybridActorCritic, load_default
from teacher_search import choose_search


def expert_batch_labels(env, slots, teacher, opponent, device, horizon, gamma):
    """Depth-1 search labels for several slots, stepped in lockstep so the
    per-search model forwards batch across slots instead of running one at a
    time. Returns one label per input slot (int argmax of branch score)."""
    out = []
    if not slots:
        return out
    branches = ACTIONS
    searches = []
    scores = []
    discount = []
    opp_first = []
    src_obs = []
    src_mask = []
    for slot in slots:
        try:
            s = env.make_search(slot, 0, branches, 0)
        except RuntimeError:
            s = None
        if s is None:
            out.append((slot, None))
            continue
        searches.append(s)
        scores.append(np.zeros(branches, np.float32))
        discount.append(np.zeros(branches, np.float32))
        opp_first.append(None)
        src_obs.append(None)
        src_mask.append(None)
    if not searches:
        return out
    live_searches = list(range(len(searches)))
    for step in range(horizon):
        if not live_searches:
            break
        obs_chunks, mask_chunks, opp_chunks, opp_mask_chunks = [], [], [], []
        for k in live_searches:
            s = searches[k]
            obs_chunks.append(s.obs)
            mask_chunks.append(s.masks.astype(bool, copy=False))
            opp_chunks.append(s.obs_opponent)
            opp_mask_chunks.append(s.masks_opponent.astype(bool, copy=False))
        obs_t = torch.as_tensor(
            np.concatenate(obs_chunks, axis=0), dtype=torch.float32, device=device
        )
        mask_t = torch.as_tensor(
            np.concatenate(mask_chunks, axis=0), dtype=torch.bool, device=device
        )
        opp_obs_t = torch.as_tensor(
            np.concatenate(opp_chunks, axis=0), dtype=torch.float32, device=device
        )
        opp_mask_t = torch.as_tensor(
            np.concatenate(opp_mask_chunks, axis=0), dtype=torch.bool, device=device
        )
        with torch.no_grad():
            s_logits, _ = teacher(obs_t, mask_t)
            opp_logits, _ = opponent(opp_obs_t, opp_mask_t)
        s_argmax = s_logits.argmax(-1).cpu().numpy()
        opp_argmax = opp_logits.argmax(-1).cpu().numpy()
        for ki, k in enumerate(list(live_searches)):
            s = searches[k]
            lo, hi = ki * branches, (ki + 1) * branches
            alive = s.dones == 0
            if step == 0:
                opp_first[k] = int(opp_argmax[lo])
                learner = np.arange(branches, dtype=np.uint16)
            else:
                learner = s_argmax[lo:hi]
            opps = opp_argmax[lo:hi].copy()
            opps[:].fill(opp_first[k]) if step < 1 else None
            discount[k] += (gamma ** step) * alive
            s.step(learner, opps)
            scores[k] += (gamma ** step) * s.rewards * alive.astype(np.float32)
            done = s.dones.astype(bool)
            if done.any():
                for bi in np.flatnonzero(done):
                    if discount[k][bi] >= 0.5:
                        scores[k][bi] = (
                            1.0 if int(s.outcomes[bi]) == 1 else -1.0
                        )
                    alive[bi] = False
                s.dones_done = done
            if not alive.any():
                live_searches.remove(k)
            elif step == horizon - 1:
                live_obs = torch.as_tensor(
                    s.obs[alive], dtype=torch.float32, device=device
                )
                live_mask = torch.as_tensor(
                    s.masks[alive].astype(bool, copy=False), dtype=torch.bool,
                    device=device,
                )
                with torch.no_grad():
                    _, live_value = teacher(live_obs, live_mask)
                leaf = live_value.cpu().numpy() * (gamma ** horizon)
                for j, bi in enumerate(np.flatnonzero(alive)):
                    scores[k][bi] += leaf[j]
                live_searches.remove(k)
    for ki, k in enumerate(range(len(searches))):
        out.append((slots[ki] if slots[ki] is not None else slots[ki], int(np.argmax(scores[k]))))
        searches[k].close()
    return out

THREAT_COUNT_OFFSET = 1010


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=400.0)
    p.add_argument("--total-steps", type=int, default=80_000_000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-clip", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.002)
    p.add_argument("--kl-coef", type=float, default=0.006)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--critic-warmup", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--laika-share", type=float, default=0.375)
    p.add_argument("--frozen-share", type=float, default=0.3125)
    p.add_argument("--best-share", type=float, default=0.0)
    p.add_argument("--pool-share", type=float, default=0.0)
    p.add_argument("--pool-from", nargs="*", type=Path, default=[])
    p.add_argument("--barrage-prob", type=float, default=0.0)
    p.add_argument("--barrage-cooldown", type=int, default=25)
    p.add_argument("--barrage-count", type=int, default=2)
    p.add_argument("--barrage-spread", type=float, default=50.0)
    p.add_argument("--mpc-share", type=float, default=0.0)
    p.add_argument("--kl-anchor", choices=["frozen", "init"], default="frozen",
                   help="which policy the KL regulariser anchors to")
    p.add_argument("--aux-coef", type=float, default=0.0,
                   help=">0 trains an opponent next-action prediction head")
    p.add_argument("--deterministic-frozen", action="store_true")
    p.add_argument("--safety-beta", type=float, default=0.0)
    p.add_argument("--fire-risk-beta", type=float, default=0.0)
    p.add_argument("--group-advnorm", action="store_true",
                   help="normalise advantages within each league group")
    p.add_argument("--double-reward", type=float, default=None)
    p.add_argument("--delay-reward", type=float, default=0.0,
                   help="tiny per-frame reward for low-bullet standoffs")
    p.add_argument("--delay-budget", type=float, default=0.2)
    p.add_argument("--delay-bullets", type=int, default=1)
    p.add_argument("--delay-start", type=int, default=100)
    p.add_argument("--delay-end", type=int, default=500)
    p.add_argument("--delay-motion", type=float, default=0.01)
    p.add_argument("--no-fire", action="store_true")
    p.add_argument("--no-fire-share", type=float, default=1.0)
    p.add_argument("--survival-reward", type=float, default=0.0)
    p.add_argument("--threat-penalty", type=float, default=0.0)
    p.add_argument("--shield-train", action="store_true")
    p.add_argument("--shield-horizon", type=int, default=50)
    p.add_argument("--shield-threshold", type=float, default=0.15)
    p.add_argument("--shield-margin", type=float, default=0.35)
    p.add_argument("--shield-fraction", type=float, default=0.25)
    p.add_argument("--self-share", type=float, default=0.3125)
    p.add_argument("--threads", type=int, default=18)
    p.add_argument("--output", type=Path, default=Path("outputs/night"))
    p.add_argument("--eval-minutes", type=float, default=20.0)
    p.add_argument("--eval-laika-games", type=int, default=100)
    p.add_argument("--eval-mpc-games", type=int, default=50)
    p.add_argument("--eval-h2h-games", type=int, default=100)
    p.add_argument("--eval-matrix-games", type=int, default=48)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=0)
    p.add_argument("--expert-it", action="store_true",
                   help="depth-1 search labels on sampled frozen-lane frames, "
                        "per-frame reward bonus on action match")
    p.add_argument("--expert-it-every", type=int, default=8)
    p.add_argument("--expert-it-envs", type=int, default=8)
    p.add_argument("--expert-it-bonus", type=float, default=0.08)
    p.add_argument("--expert-it-horizon", type=int, default=12)
    p.add_argument("--expert-it-gamma", type=float, default=0.97)
    p.add_argument("--expert-it-threads", type=int, default=2)
    p.add_argument("--kd-npz", type=Path, default=None)
    p.add_argument("--kd-coef", type=float, default=0.25)
    p.add_argument("--kd-gap", type=float, default=0.05)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--frozen-from", type=Path, default=None)
    p.add_argument("--best-from", type=Path, default=None)
    p.add_argument("--freeze-trunk", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def split_counts(total: int, shares) -> list[int]:
    raw = [max(0.0, s) for s in shares]
    norm = sum(raw) or 1.0
    out = [int(round(total * s / norm)) for s in raw]
    while sum(out) < total:
        out[0] += 1
    while sum(out) > total:
        out[-1] -= 1
    return out


class League:
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        seed = config.seed & 0xFFFFFFFF
        pool_from = list(config.pool_from) if config.pool_share > 0.0 else []
        specs: list[tuple[str, float, int, int, bool]] = []
        if config.laika_share > 0.0:
            specs.append(("laika", config.laika_share, seed + 11, OPPONENT_LAIKA, False))
        if config.frozen_share > 0.0:
            specs.append(("frozen", config.frozen_share, seed + 22, OPPONENT_FROZEN, True))
        if config.mpc_share > 0.0:
            specs.append(("mpc", config.mpc_share, seed + 55, OPPONENT_MPC, False))
        if config.self_share > 0.0:
            specs.append(("self", config.self_share, seed + 33, OPPONENT_FROZEN, True))
        if config.best_share > 0.0:
            specs.append(("best", config.best_share, seed + 44, OPPONENT_FROZEN, True))
        if pool_from:
            per_pool = config.pool_share / len(pool_from)
            for i in range(len(pool_from)):
                specs.append(
                    (f"pool{i}", per_pool, seed + 100 + i * 11, OPPONENT_FROZEN, True)
                )
        if not specs:
            raise ValueError("no league lanes configured")
        counts = split_counts(config.envs, [spec[1] for spec in specs])
        threads = max(1, config.threads // max(1, len(specs)))
        self.names = [spec[0] for spec in specs]
        self.counts = counts
        self.offsets = np.cumsum([0] + counts).tolist()
        self.envs = [
            DuelVec(counts[i], spec[2], spec[3], threads)
            for i, spec in enumerate(specs)
        ]
        self.external = [spec[4] for spec in specs]

        # Drop zero-count lanes outright: empty vec envs crash downstream
        # categorical sampling and reward bookkeeping.
        keep = [i for i, c in enumerate(self.counts) if c > 0]
        if len(keep) != len(self.counts):
            self.names = [self.names[i] for i in keep]
            self.counts = [self.counts[i] for i in keep]
            self.envs = [self.envs[i] for i in keep]
            self.external = [self.external[i] for i in keep]
            self.offsets = np.cumsum([0] + self.counts).tolist()

    def close(self):
        for env in self.envs:
            env.close()


def stack_group(envs, field: str, dtype, copy: bool = True):
    parts = []
    for env in envs:
        value = getattr(env, field)
        parts.append(value.copy() if copy else value)
    return np.concatenate(parts, axis=0).astype(dtype, copy=False)


def sample_actions(model, obs, mask, device, sample: bool):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = model(obs_t, mask_t)
        if sample:
            action = Categorical(logits=logits).sample()
        else:
            action = logits.argmax(-1)
    return action, logits, value


@torch.no_grad()
def rollout(league: League, model, frozen, best_model, pool_models, config, device,
            anchor=None, teacher=None):
    tmax = config.rollout
    total = sum(league.counts)
    obs = np.empty((tmax, total, OBS_DIM), np.float32)
    masks = np.empty((tmax, total, BULLET_SLOTS), bool)
    actions = np.empty((tmax, total), np.int64)
    logprob = np.empty((tmax, total), np.float32)
    values = np.empty((tmax, total), np.float32)
    next_values = np.empty((tmax, total), np.float32)
    rewards = np.empty((tmax, total), np.float32)
    dones = np.empty((tmax, total), bool)
    terminals = np.empty((tmax, total), bool)
    ref_logits = np.empty((tmax, total, ACTIONS), np.float32)
    aux_labels = (
        np.full((tmax, total), -1, np.int64) if getattr(config, "aux_coef", 0.0) else None
    )
    outcomes = np.zeros((len(league.envs), 5), np.int64)
    delay_accum = np.zeros(total, np.float32)
    no_fire_env = np.zeros(total, bool)
    if config.no_fire:
        if config.no_fire_share >= 0.999:
            no_fire_env[:] = True
        else:
            no_fire_env = np.random.rand(total) < config.no_fire_share
    model.eval()
    started = time.time()
    total_shield_overrides = 0
    expert_slots: list[tuple[int, int]] = []
    expert_cursor = 0
    expert_labels = None
    if teacher is not None:
        for gi, env in enumerate(league.envs):
            if league.names[gi] == "frozen" and league.counts[gi] >= 2:
                count = league.counts[gi]
                stride = max(1, count // max(config.expert_it_envs, 1))
                for slot in range(0, count, stride):
                    expert_slots.append((gi, slot))
        sel = max(config.expert_it_envs, 1)
        if len(expert_slots) > sel * 3:
            expert_slots = expert_slots[: sel * 3]
        expert_labels = np.full((tmax, total), -1, np.int64)
        print(f"expert-it: {len(expert_slots)} labeled frozen slots, "
              f"every {config.expert_it_every} steps", flush=True)
    expert_labeled = 0
    expert_matched = 0
    for t in range(tmax):
        obs[t] = stack_group(league.envs, "obs", np.float32)
        masks[t] = stack_group(league.envs, "masks", np.bool_)
        obs_t = torch.as_tensor(obs[t], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(masks[t], dtype=torch.bool, device=device)
        logits, value = model(obs_t, mask_t)
        if config.no_fire:
            logits = logits.clone()
            logits[no_fire_env, 1::2] = -1e9
        dist = Categorical(logits=logits)
        action = dist.sample()
        actions[t] = action.cpu().numpy()
        logprob[t] = dist.log_prob(action).cpu().numpy()
        shield_overrides = 0
        if getattr(config, "shield_train", False):
            step = max(1, int(round(1.0 / max(config.shield_fraction, 1e-6))))
            for gi, env in enumerate(league.envs):
                start, end = league.offsets[gi], league.offsets[gi + 1]
                for idx in range(start, end, step):
                    slot = idx - start
                    safety = env.predictive_safety(
                        slot, config.shield_horizon, True,
                    )
                    old_action = int(actions[t, idx])
                    movement = old_action // 2
                    current = float(safety[movement])
                    best = int(np.argmax(safety))
                    if current < config.shield_threshold and safety[best] > current + config.shield_margin:
                        new_action = best * 2 + (old_action & 1)
                        actions[t, idx] = new_action
                        with torch.no_grad():
                            logprob[t, idx] = float(
                                torch.log_softmax(logits[idx], -1)[new_action].item()
                            )
                        shield_overrides += 1
                        total_shield_overrides += 1
        values[t] = value.cpu().numpy()
        with torch.no_grad():
            ref_source = frozen if anchor is None else anchor
            ref_logits[t] = ref_source(obs_t, mask_t)[0].cpu().numpy()
        if expert_labels is not None and t % max(config.expert_it_every, 1) == 0:
            batch_eff = expert_slots[expert_cursor:expert_cursor + max(config.expert_it_envs, 1)]
            expert_cursor += len(batch_eff)
            if expert_cursor >= len(expert_slots):
                expert_cursor = 0
            by_env: dict[int, list[int]] = {}
            for gi, slot in batch_eff:
                by_env.setdefault(gi, []).append(slot)
            for gi, slots in by_env.items():
                env_l = league.envs[gi]
                labels = expert_batch_labels(
                    env_l, slots, teacher, frozen, device,
                    config.expert_it_horizon, config.expert_it_gamma,
                )
                for slot, label in labels:
                    if label is None:
                        continue
                    idx = league.offsets[gi] + slot
                    expert_labels[t, idx] = label
                    exact = int(actions[t, idx]) == int(label)
                    expert_labeled += 1
                    expert_matched += 1 if exact else 0
        opponent_actions = np.zeros(total, np.uint16)
        for gi, env in enumerate(league.envs):
            start, end = league.offsets[gi], league.offsets[gi + 1]
            if not league.external[gi]:
                continue
            opp_obs = env.obs_opponent.copy()
            opp_mask = env.masks_opponent.astype(bool, copy=True)
            opp_obs_t = torch.as_tensor(opp_obs, dtype=torch.float32, device=device)
            opp_mask_t = torch.as_tensor(opp_mask, dtype=torch.bool, device=device)
            with torch.no_grad():
                name = league.names[gi]
                if name == "frozen":
                    opp_model = frozen
                elif name == "best":
                    opp_model = best_model
                elif name.startswith("pool"):
                    opp_model = pool_models[int(name[4:])]
                else:
                    opp_model = model
                opp_logits, _ = opp_model(opp_obs_t, opp_mask_t)
                if name == "frozen" and getattr(config, "deterministic_frozen", False):
                    opp = opp_logits.argmax(-1).cpu().numpy().astype(np.uint16)
                else:
                    opp = Categorical(logits=opp_logits).sample().cpu().numpy().astype(np.uint16)
            opponent_actions[start:end] = opp
            if aux_labels is not None and name == "frozen":
                aux_labels[t, start:end] = opp_logits.argmax(-1).cpu().numpy().astype(np.int64)
        for gi, env in enumerate(league.envs):
            start, end = league.offsets[gi], league.offsets[gi + 1]
            supplied = opponent_actions[start:end] if league.external[gi] else None
            previous_safety = env.safety.copy()
            frames_before = env.episode_frames.copy()
            previous_fire = (
                obs[t, start:end, 895] * obs[t, start:end, 876]
            ) if config.fire_risk_beta != 0.0 else 0.0
            env.step(actions[t, start:end].astype(np.uint16), supplied)
            for i in range(league.counts[gi]):
                if env.dones[i] and env.terminals[i]:
                    outcomes[gi, min(int(env.outcomes[i]), 4)] += 1
            next_obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
            next_mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
            with torch.no_grad():
                _, nv = model(next_obs, next_mask)
            next_values[t, start:end] = nv.cpu().numpy()
            if config.fire_risk_beta != 0.0:
                next_fire = env.obs[:, 895] * env.obs[:, 876]
                previous_effective = previous_safety * (
                    1.0 - config.fire_risk_beta * previous_fire
                )
                next_effective = env.safety * (1.0 - config.fire_risk_beta * next_fire)
            else:
                previous_effective = previous_safety
                next_effective = env.safety
            shaping = config.safety_beta * (
                config.gamma * next_effective.astype(np.float32) - previous_effective
            ) if config.safety_beta != 0.0 else 0.0
            rewards[t, start:end] = env.rewards + shaping
            if expert_labels is not None:
                lab = expert_labels[t, start:end]
                hit = lab >= 0
                if hit.any():
                    match = hit & (actions[t, start:end] == lab)
                    rewards[t, start:end] += (
                        np.where(match, config.expert_it_bonus, 0.0).astype(np.float32)
                    )
            if config.survival_reward != 0.0:
                rewards[t, start:end] += (
                    config.survival_reward
                    * no_fire_env[start:end]
                    * (~env.dones.astype(bool))
                )
            if config.threat_penalty != 0.0:
                rewards[t, start:end] -= (
                    config.threat_penalty * obs[t, start:end, THREAT_COUNT_OFFSET]
                )
            if config.double_reward is not None:
                local_count = end - start
                doubled = (env.outcomes[:local_count] == 3) & env.terminals[:local_count].astype(bool)
                rewards[t, start:end][doubled] = config.double_reward
            if config.delay_reward > 0.0:
                local_count = end - start
                frame = frames_before[:local_count]
                self_xy = obs[t, start:end, 856:858]
                opp_xy = obs[t, start:end, 868:870]
                if t > 0:
                    motion = (np.abs(self_xy - obs[t - 1, start:end, 856:858]).sum(axis=1)
                              + np.abs(opp_xy - obs[t - 1, start:end, 868:870]).sum(axis=1))
                else:
                    motion = np.zeros(local_count, np.float32)
                bullets = masks[t, start:end].sum(axis=1)
                eligible = (
                    (frame >= config.delay_start)
                    & (frame <= config.delay_end)
                    & (bullets <= config.delay_bullets)
                    & (motion <= config.delay_motion)
                    & ~env.terminals[:local_count].astype(bool)
                )
                room = np.maximum(0.0, config.delay_budget - delay_accum[start:end])
                gain = np.minimum(config.delay_reward, room)
                gain[~eligible] = 0.0
                rewards[t, start:end] += gain
                delay_accum[start:end] += gain
                delay_accum[start:end][env.dones[:local_count].astype(bool)] = 0.0
            dones[t, start:end] = env.dones.astype(bool)
            terminals[t, start:end] = env.terminals.astype(bool)
            env.reset_done()
    seconds = time.time() - started
    advantages = np.zeros_like(rewards)
    carry = np.zeros(total, np.float32)
    for t in range(tmax - 1, -1, -1):
        alive = 1.0 - dones[t]
        delta = rewards[t] + config.gamma * next_values[t] * alive - values[t]
        carry = delta + config.gamma * config.gae * alive * carry
        advantages[t] = carry
    if config.group_advnorm:
        for start, end in zip(league.offsets[:-1], league.offsets[1:]):
            if end - start <= 1:
                continue
            block = advantages[:, start:end]
            mean, std = block.mean(), block.std()
            if std > 1e-8:
                advantages[:, start:end] = (block - mean) / (std + 1e-8)
    returns = advantages + values
    batch = {
        "obs": obs,
        "masks": masks,
        "actions": actions,
        "logprob": logprob,
        "values": values,
        "advantages": advantages,
        "returns": returns,
        "ref_logits": ref_logits,
        "aux_labels": aux_labels,
        "no_fire": np.broadcast_to(no_fire_env, (tmax, total)).copy(),
    }
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "fire_rate": float((actions % 2 == 1).mean()),
        "shield_overrides": int(total_shield_overrides),
        "expert_labeled": int(expert_labeled),
        "expert_match": (expert_matched / expert_labeled) if expert_labeled else 0.0,
        "done_count": int(dones.sum()),
        "sps": float(tmax * total / max(seconds, 1e-9)),
        "outcomes": {
            league.names[gi]: {
                "win": int(row[1]),
                "loss": int(row[2]),
                "double": int(row[3]),
                "draw": int(row[4]),
            }
            for gi, row in enumerate(outcomes)
        },
    }
    return batch, metrics


def update_policy(model, optimiser, batch, config, device, ref_model, update_index, kd=None):
    model.train()
    total = batch["obs"].shape[0] * batch["obs"].shape[1]
    obs = batch["obs"].reshape(total, OBS_DIM)
    masks = batch["masks"].reshape(total, BULLET_SLOTS)
    actions = batch["actions"].reshape(total)
    old_logprob = batch["logprob"].reshape(total)
    old_values = batch["values"].reshape(total)
    advantages = batch["advantages"].reshape(total)
    returns = batch["returns"].reshape(total)
    ref_logits = batch["ref_logits"].reshape(total, ACTIONS)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
    action_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
    old_logprob_t = torch.as_tensor(old_logprob, dtype=torch.float32, device=device)
    old_value_t = torch.as_tensor(old_values, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    ref_logits_t = torch.as_tensor(ref_logits, dtype=torch.float32, device=device)
    no_fire_t = torch.as_tensor(
        batch["no_fire"].reshape(total), dtype=torch.bool, device=device
    )
    aux_labels_t = None
    if batch.get("aux_labels") is not None:
        aux_labels_t = torch.as_tensor(
            batch["aux_labels"].reshape(total), dtype=torch.int64, device=device
        )
    size = total // config.minibatches
    indices = np.arange(total)
    aggregates = []
    stop = False
    critic_only = update_index < config.critic_warmup
    for _ in range(config.epochs):
        np.random.shuffle(indices)
        for start in range(0, total, size):
            sel = torch.as_tensor(indices[start:start + size], device=device)
            if critic_only:
                features = model.features(obs_t[sel], mask_t[sel]).detach()
                value = model.critic(features).squeeze(-1)
                clipped_value = old_value_t[sel] + (value - old_value_t[sel]).clamp(
                    -config.value_clip, config.value_clip
                )
                value_loss = 0.5 * torch.maximum(
                    (value - ret_t[sel]).square(), (clipped_value - ret_t[sel]).square()
                ).mean()
                loss = value_loss
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad)
                optimiser.step()
                aggregates.append({
                    "policy_loss": 0.0,
                    "value_loss": float(value_loss.detach()),
                    "entropy": 0.0,
                    "kl_ref": 0.0,
                    "approx_kl": 0.0,
                    "clipfrac": 0.0,
                    "grad_norm": float(grad_norm),
                })
                continue
            logits, value, feat_f = model(obs_t[sel], mask_t[sel], want_features=True)
            nf = no_fire_t[sel]
            if config.no_fire:
                logits = logits.clone()
                logits[nf, 1::2] = -1e9
            dist = Categorical(logits=logits)
            logprob = dist.log_prob(action_t[sel])
            ratio = (logprob - old_logprob_t[sel]).exp()
            policy_loss = -torch.minimum(
                ratio * adv_t[sel],
                ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * adv_t[sel],
            ).mean()
            sel_value = value
            clipped_value = old_value_t[sel] + (sel_value - old_value_t[sel]).clamp(
                -config.value_clip, config.value_clip
            )
            value_loss = 0.5 * torch.maximum(
                (sel_value - ret_t[sel]).square(), (clipped_value - ret_t[sel]).square()
            ).mean()
            entropy = dist.entropy().mean()
            ref_sel = ref_logits_t[sel]
            if config.no_fire:
                ref_sel = ref_sel.clone()
                ref_sel[nf, 1::2] = -1e9
            ref_prob = F.softmax(ref_sel, dim=-1)
            kl_ref = F.kl_div(
                F.log_softmax(logits, dim=-1), ref_prob, reduction="batchmean"
            )
            aux = torch.zeros((), device=device)
            if aux_labels_t is not None and aux_labels_t[sel].ge(0).any():
                opp_pred = model.opp_head(feat_f)
                aux = F.cross_entropy(opp_pred, aux_labels_t[sel], ignore_index=-1)
            kd_loss = torch.zeros((), device=device)
            if kd is not None:
                idx = torch.randint(0, kd["obs"].shape[0], (sel.shape[0],), device=device)
                kd_logits, _ = model(kd["obs"][idx], kd["mask"][idx])
                kd_loss = F.cross_entropy(kd_logits, kd["action"][idx], reduction="none")
                kd_loss = (kd_loss * kd["weight"][idx]).sum() / kd["weight"][idx].sum().clamp_min(1.0)
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
                + config.kl_coef * kl_ref
                + config.aux_coef * aux
                + config.kd_coef * kd_loss
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad)
            optimiser.step()
            with torch.no_grad():
                log_ratio = logprob - old_logprob_t[sel]
                approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config.clip).float().mean()
            aggregates.append({
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "kl_ref": float(kl_ref.detach()),
                "aux_loss": float(aux.detach()),
                "kd_loss": float(kd_loss.detach()),                "approx_kl": float(approx_kl),
                "clipfrac": float(clipfrac),
                "grad_norm": float(grad_norm),
            })
            if float(approx_kl) > config.target_kl:
                stop = True
                break
        if stop:
            break
    return {
        key: float(np.mean([row[key] for row in aggregates])) for key in aggregates[0]
    } | {"early_stop_kl": stop, "minibatches": len(aggregates), "critic_only": critic_only}


@torch.no_grad()
def evaluate(model, opponent, games, envs, seed, device, threads=20):
    env = DuelVec(envs, seed, opponent, min(threads, envs))
    wins = losses = doubles = draws = episodes = 0
    started = time.time()
    while episodes < games:
        obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
        mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
        logits, _ = model(obs, mask)
        actions = logits.argmax(-1).cpu().numpy().astype(np.uint16)
        if opponent == OPPONENT_FROZEN:
            opp = None
        else:
            opp = None
        env.step(actions, opp)
        for i in range(envs):
            if env.dones[i] and env.terminals[i]:
                o = int(env.outcomes[i])
                wins += o == 1
                losses += o == 2
                doubles += o == 3
                draws += o == 4
                episodes += 1
                if episodes >= games:
                    break
        env.reset_done()
    env.close()
    return {
        "games": episodes,
        "wins": wins,
        "losses": losses,
        "doubles": doubles,
        "draws": draws,
        "win_rate": wins / max(episodes, 1),
        "seconds": time.time() - started,
    }


@torch.no_grad()
def evaluate_h2h(model, opponent_model, games, envs, seed, device, threads=20):
    env = DuelVec(envs, seed, OPPONENT_FROZEN, min(threads, envs))
    wins = losses = doubles = draws = episodes = 0
    started = time.time()
    while episodes < games:
        obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
        mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
        logits, _ = model(obs, mask)
        actions = logits.argmax(-1).cpu().numpy().astype(np.uint16)
        oo = torch.as_tensor(env.obs_opponent.copy(), dtype=torch.float32, device=device)
        om = torch.as_tensor(env.masks_opponent.astype(bool, copy=True), device=device)
        opp_logits, _ = opponent_model(oo, om)
        opp_actions = opp_logits.argmax(-1).cpu().numpy().astype(np.uint16)
        env.step(actions, opp_actions)
        for i in range(envs):
            if env.dones[i] and env.terminals[i]:
                o = int(env.outcomes[i])
                wins += o == 1
                losses += o == 2
                doubles += o == 3
                draws += o == 4
                episodes += 1
                if episodes >= games:
                    break
        env.reset_done()
    env.close()
    return {
        "games": episodes,
        "wins": wins,
        "losses": losses,
        "doubles": doubles,
        "draws": draws,
        "win_rate": wins / max(episodes, 1),
        "seconds": time.time() - started,
    }


def main():
    config = parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "checkpoints").mkdir(exist_ok=True)
    print(f"device={device} output={config.output}", flush=True)
    model = load_default(device)
    if config.random_init:
        model = HybridActorCritic(gated_dodge=True, gated_ammo=True).to(device)
        print("random initialisation", flush=True)
    elif config.init_from and config.init_from.exists():
        payload = torch.load(config.init_from, map_location=device, weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=False)
        print(f"initialised from {config.init_from}", flush=True)
    if config.frozen_from and config.frozen_from.exists():
        payload = torch.load(config.frozen_from, map_location=device, weights_only=False)
        frozen = load_default(device)
        frozen.load_state_dict(payload.get("model", payload), strict=False)
        print(f"frozen opponent from {config.frozen_from}", flush=True)
    else:
        frozen = copy.deepcopy(load_default(device))
    frozen = frozen.to(device).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    anchor = None
    if config.kl_anchor == "init":
        anchor = copy.deepcopy(model)
        for p in anchor.parameters():
            p.requires_grad_(False)
        anchor.eval()
        print("KL anchor = starting weights", flush=True)
    teacher = None
    if config.expert_it:
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        print("expert-iteration teacher = starting-weights depth-1 search", flush=True)
    best_model = None
    if config.best_share > 0.0:
        best_model = load_default(device)
        if config.best_from and config.best_from.exists():
            payload = torch.load(config.best_from, map_location=device, weights_only=False)
            best_model.load_state_dict(payload.get("model", payload), strict=False)
            print(f"best opponent from {config.best_from}", flush=True)
        best_model = best_model.to(device).eval()
        for p in best_model.parameters():
            p.requires_grad_(False)
    pool_models = []
    if config.pool_share > 0.0 and config.pool_from:
        for path in config.pool_from:
            if not path.exists():
                raise FileNotFoundError(path)
            pool_net = load_default(device)
            if path.suffix == ".json":
                pool_net.load_from_assets(path, path.with_suffix(".bin"))
            else:
                payload = torch.load(path, map_location=device, weights_only=False)
                pool_net.load_state_dict(payload.get("model", payload), strict=False)
            pool_net = pool_net.to(device).eval()
            for p in pool_net.parameters():
                p.requires_grad_(False)
            pool_models.append(pool_net)
            print(f"pool opponent from {path}", flush=True)
    matrix_items = [("v1_hybrid", frozen)]
    for p, net in zip(config.pool_from, pool_models):
        matrix_items.append((Path(p).stem, net))
    if config.freeze_trunk:
        frozen_prefixes = ("map0.", "map2.", "map5.", "bullets0.", "bullets2.",
                           "scalars0.", "trunk0.")
        frozen_count = 0
        for name, parameter in model.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad_(False)
                frozen_count += 1
        print(f"frozen trunk parameters: {frozen_count}", flush=True)
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config.lr, eps=1e-5
    )
    kd = None
    if config.kd_npz and config.kd_npz.exists():
        data = np.load(config.kd_npz, allow_pickle=False)
        obs_np = data["obs"].astype(np.float32)
        mask_np = data["masks"].astype(bool)
        act_np = data["action"].astype(np.int64)
        q_np = data["q"].astype(np.float32)
        with torch.no_grad():
            model.eval()
            n = obs_np.shape[0]
            stud = []
            for off in range(0, n, 8192):
                o = torch.as_tensor(obs_np[off:off + 8192], device=device)
                m = torch.as_tensor(mask_np[off:off + 8192], device=device)
                stud.append(model(o, m)[0])
            study = torch.cat(stud).argmax(-1)
        stud_np = study.cpu().numpy()
        teacher = q_np.argmax(-1)
        gap = q_np.max(-1) - np.sort(q_np, axis=-1)[:, -2]
        keep = (teacher != stud_np) | (gap >= config.kd_gap)
        keep_idx = np.nonzero(keep)[0]
        obs_np, mask_np = obs_np[keep_idx], mask_np[keep_idx]
        act_np, q_np = act_np[keep_idx], q_np[keep_idx]
        stud_np = stud_np[keep_idx]
        weight = np.ones(keep_idx.shape[0], np.float32)
        weight[q_np.argmax(-1) == stud_np] = 0.35
        kd = {
            "obs": torch.as_tensor(obs_np, device=device),
            "mask": torch.as_tensor(mask_np, device=device),
            "action": torch.as_tensor(act_np, device=device),
            "weight": torch.as_tensor(weight, device=device),
        }
        print(f"kd npz: kept {keep_idx.shape[0]} frames", flush=True)
    start_update = 0
    total_steps = 0
    best_score = -1.0
    if config.resume and config.resume.exists():
        payload = torch.load(config.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimiser.load_state_dict(payload["optimiser"])
        start_update = int(payload.get("update", 0)) + 1
        total_steps = int(payload.get("total_steps", 0))
        best_score = float(payload.get("best_score", -1.0))
        print(f"resumed {config.resume} update={start_update} steps={total_steps}", flush=True)
    league = League(config, device)
    if config.barrage_prob > 0.0:
        for env in league.envs:
            env.set_barrage(
                config.barrage_prob,
                config.barrage_cooldown,
                config.barrage_count,
                config.barrage_spread,
            )
    steps_per_update = config.rollout * config.envs
    deadline = time.time() + config.minutes * 60.0
    last_eval = 0.0
    update = start_update
    metrics_path = config.output / "metrics.jsonl"
    while update < 10_000_000 and total_steps < config.total_steps:
        if time.time() > deadline:
            print("time budget reached", flush=True)
            break
        progress = min(total_steps / max(config.total_steps, 1), 1.0)
        lr = config.lr * max(0.08, 1.0 - progress)
        entropy_coef = config.entropy_coef * max(0.1, 1.0 - progress)
        for group in optimiser.param_groups:
            group["lr"] = lr
        old_entropy = config.entropy_coef
        config.entropy_coef = entropy_coef
        batch, rollout_metrics = rollout(
            league, model, frozen, best_model, pool_models, config, device, anchor,
            teacher,
        )
        config.entropy_coef = old_entropy
        update_metrics = update_policy(model, optimiser, batch, config, device, anchor, update, kd=kd)
        total_steps += steps_per_update
        record = {
            "update": update,
            "total_steps": total_steps,
            "lr": lr,
            "rollout": rollout_metrics,
            "ppo": update_metrics,
            "elapsed": time.time() - (deadline - config.minutes * 60.0),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"u{update} steps={total_steps:,} sps={rollout_metrics['sps']:.0f} "
            f"R={rollout_metrics['reward_mean']:.4f} H={update_metrics['entropy']:.3f} "
            f"KL={update_metrics['approx_kl']:.4f} refKL={update_metrics['kl_ref']:.4f} "
            f"VF={update_metrics['value_loss']:.4f}"
            + (" [critic]" if update_metrics["critic_only"] else ""),
            flush=True,
        )
        if update % config.save_every == 0 or update == start_update:
            torch.save({
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "update": update,
                "total_steps": total_steps,
                "best_score": best_score,
                "config": vars(config) | {"output": str(config.output)},
            }, config.output / "checkpoints" / "last.pt")
        if config.snapshot_every and update % config.snapshot_every == 0:
            snapdir = config.output / "checkpoints" / "snap"
            snapdir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "update": update,
                "total_steps": total_steps,
            }, snapdir / f"u{update}.pt")
        if time.time() - last_eval >= config.eval_minutes * 60.0:
            last_eval = time.time()
            model.eval()
            ev_laika = evaluate(model, OPPONENT_LAIKA, config.eval_laika_games, 32, config.seed + 1000, device)
            ev_mpc = evaluate(model, OPPONENT_MPC, config.eval_mpc_games, 8, config.seed + 2000, device)
            ev_h2h = evaluate_h2h(model, frozen, config.eval_h2h_games, 32, config.seed + 3000, device)
            matrix = {}
            for mi, (label, opp_model) in enumerate(matrix_items):
                matrix[label] = evaluate_h2h(
                    model, opp_model, config.eval_matrix_games, 16,
                    config.seed + 5000 + mi, device,
                )
            score = 0.4 * ev_laika["win_rate"] + 0.3 * ev_mpc["win_rate"] + 0.3 * ev_h2h["win_rate"]
            matrix_text = " ".join(
                f"{label}={ev['win_rate']:.3f}" for label, ev in matrix.items()
            )
            print(
                f"EVAL u{update} laika={ev_laika['win_rate']:.3f} mpc={ev_mpc['win_rate']:.3f} "
                f"h2h={ev_h2h['win_rate']:.3f} score={score:.4f}"
                + (f" | {matrix_text}" if matrix_text else ""),
                flush=True,
            )
            with (config.output / "evals.jsonl").open("a") as handle:
                handle.write(json.dumps({
                    "update": update, "total_steps": total_steps, "score": score,
                    "laika": ev_laika, "mpc": ev_mpc, "h2h": ev_h2h,
                    "matrix": {label: ev for label, ev in matrix.items()},
                }) + "\n")
            if score > best_score:
                best_score = score
                torch.save({
                    "model": model.state_dict(),
                    "update": update,
                    "total_steps": total_steps,
                    "best_score": best_score,
                    "evals": {"laika": ev_laika, "mpc": ev_mpc, "h2h": ev_h2h,
                              "matrix": dict(matrix)},
                }, config.output / "checkpoints" / "best.pt")
        update += 1
    torch.save({
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "update": update,
        "total_steps": total_steps,
        "best_score": best_score,
        "config": vars(config) | {"output": str(config.output)},
    }, config.output / "checkpoints" / "final.pt")
    league.close()
    print(f"done update={update} steps={total_steps:,} best={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
