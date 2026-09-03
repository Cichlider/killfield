"""PPO on the duel curriculum: the real game, win/loss only.

The reward carries no information until the round is over, so the network's
whole job is to read a 1010-dimension observation and the critic's whole job is
to predict an outcome this project's own ablations found barely predictable
(learned `V(s)` scored R^2 near zero six times). Three consequences shape the
configuration below, and each of them is a scar:

* **A learning-rate schedule is mandatory.** A constant rate ran to 60M steps
  once and collapsed irreversibly at around 2M, never recovering.
* **The critic gets a head start.** Updating a policy against a value function
  that has not fitted yet is the classic way to diverge, and a sparse terminal
  reward makes the critic the slow half by construction.
* **Entropy decays with the rate rather than staying put.** Nothing rewards
  exploration here except finding a win, so the search has to stay wide for
  much longer than a shaped curriculum needs.

Checkpoints are published every `--save-every` steps for the viewer to pick up,
so the honest read of progress is watching it play rather than reading a curve.
"""

from __future__ import annotations

import argparse
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

from duel_env import (
    ACTIONS,
    OPPONENT_NAMES,
    BULLET_DIM,
    BULLET_OFFSET,
    BULLET_SLOTS,
    MAP_CHANNELS,
    MAP_DIM,
    MAP_H,
    MAP_W,
    OBS_DIM,
    OBS_SCHEMA_VERSION,
    SCALAR_DIM,
    DuelVec,
)

ARCH = "duel_cnn_v1"


@dataclass(frozen=True)
class Config:
    total_steps: int = 200_000_000
    envs: int = 256
    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 8
    learning_rate: float = 3e-4
    # The reward lands once, up to 875 frames away. 0.999^875 = 0.42.
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    # Opponent pool weights: scripted AI, planner, frozen checkpoint.
    laika_weight: float = 0.4
    mpc_weight: float = 0.4
    frozen_weight: float = 0.2
    seed: int = 11
    critic_warmup_updates: int = 20


class ActorCritic(nn.Module):
    """A small CNN over the maze, a shared encoder over the bullets, an MLP
    over everything else.

    The bullet rows arrive in engine creation order, which shifts as rounds are
    fired and expire. Feeding that to a dense layer would make the policy
    sensitive to storage order, so the rows go through one shared encoder and
    are pooled with a mask.
    """

    def __init__(self):
        super().__init__()
        self.map = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ((MAP_W + 1) // 2) * ((MAP_H + 1) // 2), 128), nn.Tanh(),
        )
        self.bullets = nn.Sequential(
            nn.Linear(BULLET_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.scalars = nn.Sequential(nn.Linear(SCALAR_DIM, 128), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(128 + 128 + 64, 256), nn.Tanh())
        self.actor = nn.Linear(256, ACTIONS)
        self.critic = nn.Linear(256, 1)
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def features(self, obs, mask):
        grid = obs[:, :MAP_DIM].reshape(-1, MAP_W, MAP_H, MAP_CHANNELS)
        grid = grid.permute(0, 3, 1, 2).contiguous()

        rows = obs[:, BULLET_OFFSET:BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM]
        rows = rows.reshape(-1, BULLET_SLOTS, BULLET_DIM)
        encoded = self.bullets(rows)
        m = mask.unsqueeze(-1)
        mean = (encoded * m).sum(1) / m.sum(1).clamp(min=1)
        peak = torch.amax(encoded.masked_fill(~m, -torch.inf), dim=1)
        peak = torch.where((~mask.any(1))[:, None], torch.zeros_like(peak), peak)

        scalars = torch.cat(
            (
                obs[:, MAP_DIM:BULLET_OFFSET],
                obs[:, BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM:],
            ),
            dim=1,
        )
        return self.trunk(
            torch.cat((self.map(grid), self.scalars(scalars), mean, peak), dim=1)
        )

    def forward(self, obs, mask):
        features = self.features(obs, mask)
        return self.actor(features), self.critic(features).squeeze(-1)


def tensors(env, device):
    return (
        torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device),
        torch.as_tensor(env.masks.astype(bool), dtype=torch.bool, device=device),
    )


def pick_device(model) -> torch.device:
    """Time both and take the faster.

    This network is a few hundred thousand parameters, which on Apple silicon
    is small enough that GPU dispatch latency can outweigh the arithmetic
    entirely — a 32k-parameter model measured 2.9x *faster* on the CPU. Rather
    than hardcode a guess that goes stale when the architecture changes, run
    twenty forward passes on each.
    """
    candidates = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        candidates.append(torch.device("mps"))
    if len(candidates) == 1:
        return candidates[0]

    best, best_time = candidates[0], float("inf")
    for device in candidates:
        probe = model.to(device)
        obs = torch.zeros(256, OBS_DIM, device=device)
        mask = torch.zeros(256, BULLET_SLOTS, dtype=torch.bool, device=device)
        with torch.inference_mode():
            for _ in range(3):
                probe(obs, mask)
            started = time.perf_counter()
            for _ in range(20):
                probe(obs, mask)
            elapsed = time.perf_counter() - started
        print(f"  {str(device):<5} {1000 * elapsed / 20:.2f} ms/forward", flush=True)
        if elapsed < best_time:
            best, best_time = device, elapsed
    return best


def save_live(output: Path, model, config: Config, steps: int, update: int, started: float):
    """Publish the current weights for the viewer.

    Written to a fixed path so the server can key its cache on mtime and the
    page keeps one stable URL. Both files go out through a temp file and
    `os.replace`, which is atomic on the same filesystem, and the manifest is
    written after the weights so a fresh manifest always describes weights
    already on disk.
    """
    tmp = output / "live.pt.tmp"
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "arch": ARCH}, tmp)
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
        "pool": {"laika": config.laika_weight, "mpc": config.mpc_weight,
                 "frozen": config.frozen_weight},
        "timestamp": time.time(),
    }
    tmp = output / "live.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, output / "live.json")


class Tally:
    """Results since the last report, split by opponent.

    Win rate against the scripted AI is the number this project has always
    reported, so it stays comparable with every historical run; the planner
    column is new and much harder.
    """

    NAMES = tuple(OPPONENT_NAMES.values())

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {name: [0, 0, 0, 0] for name in self.NAMES}  # win/loss/double/draw
        self.frames = []

    def record(self, outcome: int, opponent: int, frames: int):
        row = self.counts[OPPONENT_NAMES[min(opponent, 2)]]
        row[min(outcome, 4) - 1] += 1
        self.frames.append(frames)

    def summary(self) -> dict:
        out = {}
        for name, (win, loss, double, draw) in self.counts.items():
            total = win + loss + double + draw
            out[name] = {
                "rounds": total,
                "win_rate": win / total if total else None,
                "draw_rate": draw / total if total else None,
                "double_rate": double / total if total else None,
            }
        out["frames_mean"] = float(np.mean(self.frames)) if self.frames else None
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=Config.total_steps)
    parser.add_argument("--envs", type=int, default=Config.envs)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--mix", type=float, nargs=3,
                        default=(Config.laika_weight, Config.mpc_weight,
                                 Config.frozen_weight),
                        metavar=("LAIKA", "MPC", "FROZEN"),
                        help="opponent pool weights, normalised")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="warm-start the learner from this checkpoint")
    parser.add_argument("--frozen-from", type=Path, default=None,
                        help="checkpoint driving the pool's frozen slots; "
                             "defaults to --init-from")
    parser.add_argument("--threads", type=int, default=0,
                        help="engine worker threads; 0 asks the OS")
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_duel_v1"))
    parser.add_argument("--save-every", type=int, default=200_000,
                        help="publish live.pt/live.json every N steps; 0 disables")
    args = parser.parse_args()

    config = Config(
        total_steps=args.steps,
        envs=args.envs,
        seed=args.seed,
        laika_weight=args.mix[0],
        mpc_weight=args.mix[1],
        frozen_weight=args.mix[2],
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output = args.output / f"s{config.seed}"
    output.mkdir(parents=True, exist_ok=True)

    model = ActorCritic()
    parameters = sum(p.numel() for p in model.parameters())
    print(f"duel PPO · {parameters:,} parameters · picking a device:", flush=True)
    device = pick_device(model)
    model = model.to(device)

    if args.init_from:
        payload = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        print(f"warm start from {args.init_from}", flush=True)

    # The pool's frozen slots. Kept in eval mode and never updated: it is a
    # fixed rung to climb, not a moving target, so a rising win rate against it
    # means the learner improved rather than that both drifted together.
    frozen = None
    frozen_source = args.frozen_from or args.init_from
    if config.frozen_weight > 0:
        if frozen_source is None:
            raise SystemExit("--mix gives the frozen pool weight but no "
                             "--frozen-from/--init-from checkpoint to fill it")
        payload = torch.load(frozen_source, map_location=device, weights_only=False)
        frozen = ActorCritic().to(device)
        frozen.load_state_dict(payload["model"])
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        print(f"pool opponent frozen at {frozen_source}", flush=True)

    env = DuelVec(config.envs, 1_000_000 + config.seed * 977,
                  (config.laika_weight, config.mpc_weight, config.frozen_weight),
                  args.threads)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    batch = config.envs * config.rollout_steps
    updates = max(1, config.total_steps // batch)
    pool = " / ".join(f"{name} {share:.0%}"
                      for name, share in zip(OPPONENT_NAMES.values(), env.weights))
    print(f"device {device} · {updates} updates x {batch:,} steps = "
          f"{updates * batch:,} · horizon {env.episode_frames}+{env.grace_frames} frames · "
          f"对手池 {pool}", flush=True)

    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("")
    started = time.perf_counter()
    total = 0
    published = 0
    tally = Tally()
    if args.save_every:
        # Publish the untrained network at once, so the page has something from
        # the first second and step 0 is the baseline you compare against.
        save_live(output, model, config, 0, 0, started)

    for update in range(updates):
        # Linear decay to zero. A constant rate collapsed irreversibly at ~2M
        # steps once and never came back.
        progress = update / updates
        lr = config.learning_rate * (1.0 - progress)
        for group in optimiser.param_groups:
            group["lr"] = lr
        entropy_coefficient = config.entropy_coefficient * (1.0 - progress)
        # Until the critic has something to say, moving the policy against its
        # advantage estimates is noise amplification.
        critic_only = update < config.critic_warmup_updates

        shape = (config.rollout_steps, config.envs)
        obs_buf = np.empty(shape + (OBS_DIM,), np.float32)
        mask_buf = np.empty(shape + (BULLET_SLOTS,), bool)
        act_buf = np.empty(shape, np.int64)
        logp_buf = np.empty(shape, np.float32)
        val_buf = np.empty(shape, np.float32)
        rew_buf = np.empty(shape, np.float32)
        done_buf = np.empty(shape, bool)

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
                # The pool's frozen slots are driven from here, off the
                # observation the engine publishes for tank 1. Sampled, not
                # argmax: a deterministic pool opponent is a script to
                # memorise rather than an opponent to beat.
                opponent_action = None
                if frozen is not None and env.needs_action.any():
                    theirs = torch.as_tensor(env.obs_opponent.copy(),
                                             dtype=torch.float32, device=device)
                    their_mask = torch.as_tensor(env.masks_opponent.astype(bool),
                                                 dtype=torch.bool, device=device)
                    logits_o, _ = frozen(theirs, their_mask)
                    opponent_action = (
                        Categorical(logits=logits_o).sample()
                        .cpu().numpy().astype(np.uint16)
                    )
                env.step(act_buf[t].astype(np.uint16), opponent_action)
                rew_buf[t] = env.rewards
                done_buf[t] = env.dones.astype(bool)
                for i in np.flatnonzero(env.terminals):
                    tally.record(int(env.outcomes[i]), int(env.opponents[i]),
                                 int(env.frames[i]))
                env.reset_done()
            obs_t, mask_t = tensors(env, device)
            _, last_value = model(obs_t, mask_t)
            last_value = last_value.cpu().numpy()

        advantages = np.zeros(shape, np.float32)
        gae = np.zeros(config.envs, np.float32)
        for t in reversed(range(config.rollout_steps)):
            nxt = last_value if t + 1 == config.rollout_steps else val_buf[t + 1]
            # Every `done` here is a real terminal — even the draw, which is a
            # result with its own reward rather than a truncation — so the
            # bootstrap is cut in every case.
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
        b_val = flat(val_buf).float()
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
                if critic_only:
                    loss = value_loss
                else:
                    loss = (policy_loss
                            + config.value_coefficient * value_loss
                            - entropy_coefficient * entropy)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimiser.step()

        total += batch
        # How much of the outcome the critic actually explains. This project's
        # ablations put it near zero; if it stays there, the sparse reward is
        # not going to carry a policy and that is worth knowing early.
        variance = float(b_ret.var())
        explained = (
            float(1.0 - (b_ret - b_val).var() / variance) if variance > 1e-8 else 0.0
        )
        record = {
            "update": update,
            "steps": total,
            "lr": lr,
            "critic_only": critic_only,
            "reward_per_step": float(rew_buf.mean()),
            "explained_variance": explained,
            "entropy": entropy_seen,
            **tally.summary(),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        rate = lambda side: (
            "—" if record[side]["win_rate"] is None
            else f"{record[side]['win_rate']:.0%}({record[side]['rounds']})"
        )
        shown = " ".join(f"{name}={rate(name)}" for name in Tally.NAMES
                         if record[name]["rounds"])
        print(
            f"u{update + 1}/{updates} steps={total:,} {shown} "
            f"EV={explained:+.2f} H={entropy_seen:.2f}"
            + (" [critic warmup]" if critic_only else ""),
            flush=True,
        )
        tally.reset()

        if args.save_every and total - published >= args.save_every:
            published = total
            save_live(output, model, config, total, update + 1, started)

    env.close()
    result = {
        "name": f"ppo-duel-v1-joystick18-s{config.seed}",
        "steps": total,
        "seconds": time.perf_counter() - started,
        "config": asdict(config),
    }
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    torch.save({"model": model.state_dict(), "arch": ARCH, "result": result},
               output / "final.pt")
    if args.save_every:
        save_live(output, model, config, total, updates, started)


if __name__ == "__main__":
    main()
