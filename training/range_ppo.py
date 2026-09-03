"""PPO on the shooting-range curriculum.

Deliberately small. The observation is 100 hand-built features and the action
space is the planner's 18, so there is no CNN and no 258-wide action one-hot to
encode — the whole model is an MLP plus a shared bullet encoder.

The bullet encoder is the one piece that is not a plain MLP, and it has to be:
the ten bullet rows arrive in engine creation order, which shifts as rounds are
fired and expire. Feeding that to a dense layer would make the policy sensitive
to storage order, so the rows go through one shared encoder and are pooled with
a mask.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Layout mirrored from engine/src/range_obs.rs. A mismatch is caught at startup
# against the engine's own reported dimensions rather than trusted.
OBS_DIM = 100
BULLET_SLOTS = 10
BULLET_DIM = 7
BULLET_OFFSET = 26
SCALAR_DIM = OBS_DIM - BULLET_SLOTS * BULLET_DIM
ACTIONS = 18
OBS_SCHEMA_VERSION = 11
ARCH = "range_mlp_v1"


@dataclass(frozen=True)
class Config:
    total_steps: int = 3_000_000
    envs: int = 32
    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.003
    max_grad_norm: float = 0.5
    seed: int = 11


class RangeVec:
    """ctypes wrapper over the Rust vectorised range environment."""

    def __init__(self, count: int, seed: int,
                 library=Path("engine/target/release/libkf_engine.dylib")):
        self.count = count
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.lib.kf_vec_new_range.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.lib.kf_vec_new_range.restype = ctypes.c_void_p
        for name in ("kf_vec_obs_dim", "kf_vec_bullet_slots",
                     "kf_vec_action_count", "kf_vec_episode_frames",
                     "kf_vec_obs_schema_version"):
            getattr(self.lib, name).restype = ctypes.c_uint32

        native = (int(self.lib.kf_vec_obs_dim()), int(self.lib.kf_vec_bullet_slots()),
                  int(self.lib.kf_vec_action_count()),
                  int(self.lib.kf_vec_obs_schema_version()))
        expected = (OBS_DIM, BULLET_SLOTS, ACTIONS, OBS_SCHEMA_VERSION)
        if native != expected:
            raise RuntimeError(f"engine/python schema mismatch: {native} != {expected}")
        self.episode_frames = int(self.lib.kf_vec_episode_frames())

        self.handle = self.lib.kf_vec_new_range(count, seed)
        self.lib.kf_vec_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)]
        self.lib.kf_vec_reset_done.argtypes = [ctypes.c_void_p]
        self.lib.kf_vec_free.argtypes = [ctypes.c_void_p]

        self.obs = self._view("kf_vec_obs", ctypes.c_float, (count, OBS_DIM))
        self.masks = self._view("kf_vec_masks", ctypes.c_uint8, (count, BULLET_SLOTS))
        self.rewards = self._view("kf_vec_rewards", ctypes.c_float, (count,))
        self.dones = self._view("kf_vec_dones", ctypes.c_uint8, (count,))
        self.terminals = self._view("kf_vec_terminals", ctypes.c_uint8, (count,))
        self.kills = self._view("kf_vec_kills", ctypes.c_uint32, (count,))
        self.shots = self._view("kf_vec_shots", ctypes.c_uint32, (count,))
        self.good_shots = self._view("kf_vec_good_shots", ctypes.c_uint32, (count,))

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


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.bullets = nn.Sequential(
            nn.Linear(BULLET_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.scalars = nn.Sequential(nn.Linear(SCALAR_DIM, 128), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(128 + 64, 128), nn.Tanh())
        self.actor = nn.Linear(128, ACTIONS)
        self.critic = nn.Linear(128, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs, mask):
        rows = obs[:, BULLET_OFFSET:BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM]
        rows = rows.reshape(-1, BULLET_SLOTS, BULLET_DIM)
        encoded = self.bullets(rows)
        m = mask.unsqueeze(-1)
        mean = (encoded * m).sum(1) / m.sum(1).clamp(min=1)
        peak = torch.amax(encoded.masked_fill(~m, -torch.inf), dim=1)
        peak = torch.where((~mask.any(1))[:, None], torch.zeros_like(peak), peak)
        scalars = torch.cat(
            (obs[:, :BULLET_OFFSET], obs[:, BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM:]), 1
        )
        features = self.trunk(torch.cat((self.scalars(scalars), mean, peak), 1))
        return self.actor(features), self.critic(features).squeeze(-1)


def tensors(env, device):
    return (
        torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device),
        torch.as_tensor(env.masks.astype(bool), dtype=torch.bool, device=device),
    )


@torch.inference_mode()
def evaluate(model, device, episodes=30, seed=9_000):
    """Deterministic argmax episodes; the honest read of what was learnt."""
    env = RangeVec(16, seed)
    rewards, kills, good, deaths, frames = [], [], [], 0, []
    running = np.zeros(env.count)
    running_frames = np.zeros(env.count, np.int64)
    ek = np.zeros(env.count, np.int64)
    eg = np.zeros(env.count, np.int64)
    try:
        while len(rewards) < episodes:
            logits, _ = model(*tensors(env, device))
            action = logits.argmax(-1).cpu().numpy().astype(np.uint16)
            env.step(action)
            running += env.rewards
            running_frames += 1
            ek += env.kills
            eg += env.good_shots
            for i in np.flatnonzero(env.dones.astype(bool)):
                rewards.append(float(running[i]))
                kills.append(int(ek[i]))
                good.append(int(eg[i]))
                frames.append(int(running_frames[i]))
                deaths += int(env.terminals[i])
                running[i] = 0.0
                running_frames[i] = 0
                ek[i] = eg[i] = 0
            env.reset_done()
    finally:
        env.close()
    n = max(len(rewards), 1)
    return {
        "episodes": len(rewards),
        "reward_mean": float(np.mean(rewards)),
        "reward_median": float(np.median(rewards)),
        "reward_max": float(np.max(rewards)),
        "kills_per_episode": float(np.mean(kills)),
        "good_shots_per_episode": float(np.mean(good)),
        "death_rate": deaths / n,
        "frames_mean": float(np.mean(frames)),
    }


def save_live(output, model, config, steps, update, started):
    """Publish the current weights for the viewer to pick up.

    Written to a fixed path so the server can key its cache on mtime and the
    page keeps one stable URL. Both files go out through a temp file and
    `os.replace`, which is atomic on the same filesystem — a reader can never
    observe a half-written checkpoint, and the manifest is written *after* the
    weights so a fresh manifest always describes weights already on disk.
    """
    payload = {"model": model.state_dict(), "arch": ARCH}
    tmp = output / "live.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, output / "live.pt")

    manifest = {
        "arch": ARCH,
        "schema_version": OBS_SCHEMA_VERSION,
        "obs_dim": OBS_DIM,
        "bullet_slots": BULLET_SLOTS,
        "action_count": ACTIONS,
        "steps": steps,
        "update": update,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "seed": config.seed,
        "timestamp": time.time(),
    }
    tmp = output / "live.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, output / "live.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.total_steps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ppo_range_v1"))
    parser.add_argument("--save-every", type=int, default=200_000,
                        help="publish live.pt/live.json every N environment "
                             "steps; 0 disables periodic publishing")
    args = parser.parse_args()
    config = Config(total_steps=args.steps, seed=args.seed)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output = args.output / f"s{config.seed}"
    output.mkdir(parents=True, exist_ok=True)

    env = RangeVec(config.envs, 1_000_000 + config.seed * 977)
    model = ActorCritic().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    batch = config.envs * config.rollout_steps
    updates = max(1, config.total_steps // batch)
    print(f"range PPO · {device} · {updates} updates x {batch:,} steps "
          f"= {updates * batch:,} · horizon {env.episode_frames} frames", flush=True)

    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("")
    started = time.perf_counter()
    total = 0
    published = 0
    if args.save_every:
        # Publish the untrained network immediately. The page then has a model
        # from the first second, and what you watch at step 0 is the honest
        # baseline every later checkpoint is compared against.
        save_live(output, model, config, 0, 0, started)

    for update in range(updates):
        shape = (config.rollout_steps, config.envs)
        obs_buf = np.empty(shape + (OBS_DIM,), np.float32)
        mask_buf = np.empty(shape + (BULLET_SLOTS,), bool)
        act_buf = np.empty(shape, np.int64)
        logp_buf = np.empty(shape, np.float32)
        val_buf = np.empty(shape, np.float32)
        rew_buf = np.empty(shape, np.float32)
        done_buf = np.empty(shape, bool)
        episode_kills = 0
        episode_good = 0
        finished = 0

        model.eval()
        with torch.inference_mode():
            for t in range(config.rollout_steps):
                obs_t, mask_t = tensors(env, device)
                obs_buf[t] = env.obs
                mask_buf[t] = env.masks.astype(bool)
                logits, value = model(obs_t, mask_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                act_buf[t] = action.cpu().numpy()
                logp_buf[t] = dist.log_prob(action).cpu().numpy()
                val_buf[t] = value.cpu().numpy()
                env.step(act_buf[t].astype(np.uint16))
                rew_buf[t] = env.rewards
                done_buf[t] = env.dones.astype(bool)
                episode_kills += int(env.kills.sum())
                episode_good += int(env.good_shots.sum())
                finished += int(done_buf[t].sum())
                env.reset_done()
            obs_t, mask_t = tensors(env, device)
            _, last_value = model(obs_t, mask_t)
            last_value = last_value.cpu().numpy()

        advantages = np.zeros(shape, np.float32)
        gae = np.zeros(config.envs, np.float32)
        for t in reversed(range(config.rollout_steps)):
            nxt = last_value if t + 1 == config.rollout_steps else val_buf[t + 1]
            alive = 1.0 - done_buf[t]
            delta = rew_buf[t] + config.gamma * nxt * alive - val_buf[t]
            gae = delta + config.gamma * config.gae_lambda * alive * gae
            advantages[t] = gae
        returns = advantages + val_buf

        flat = lambda a: torch.as_tensor(a.reshape((batch,) + a.shape[2:]), device=device)
        b_obs = flat(obs_buf).float()
        b_mask = flat(mask_buf).bool()
        b_act = flat(act_buf).long()
        b_logp = flat(logp_buf).float()
        b_adv = flat(advantages).float()
        b_ret = flat(returns).float()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        model.train()
        indices = np.arange(batch)
        size = batch // config.minibatches
        entropy_seen = 0.0
        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch, size):
                sel = torch.as_tensor(indices[start:start + size], device=device)
                logits, value = model(b_obs[sel], b_mask[sel])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act[sel])
                ratio = (logp - b_logp[sel]).exp()
                adv = b_adv[sel]
                policy_loss = -torch.min(
                    ratio * adv,
                    ratio.clamp(1 - config.clip, 1 + config.clip) * adv,
                ).mean()
                value_loss = 0.5 * (value - b_ret[sel]).pow(2).mean()
                entropy = dist.entropy().mean()
                entropy_seen = float(entropy.detach())
                loss = (policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimiser.step()

        total += batch
        record = {
            "update": update,
            "steps": total,
            "reward_per_step": float(rew_buf.mean()),
            "kills": episode_kills,
            "good_shots": episode_good,
            "episodes": finished,
            "entropy": entropy_seen,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"u{update + 1}/{updates} steps={total:,} "
              f"reward/step={record['reward_per_step']:+.4f} "
              f"kills={episode_kills} good={episode_good} "
              f"H={entropy_seen:.3f}", flush=True)

        if args.save_every and total - published >= args.save_every:
            published = total
            save_live(output, model, config, total, update + 1, started)

    env.close()
    model.eval()
    result = {
        "name": f"ppo-range-v1-joystick18-s{config.seed}",
        "steps": total,
        "seconds": time.perf_counter() - started,
        "config": asdict(config),
        "evaluation": evaluate(model, device),
    }
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    torch.save({"model": model.state_dict(), "result": result}, output / "final.pt")
    if args.save_every:
        save_live(output, model, config, total, updates, started)
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
