from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel_env import ACTIONS, BULLET_SLOTS, DuelVec, OBS_DIM, OPPONENT_FROZEN
from hybrid_model import HybridActorCritic, load_default
from train import evaluate_h2h


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device)


def policy_view(env, start: int, end: int):
    obs = np.concatenate(
        [env.obs[:start], env.obs_opponent[start:end]], axis=0,
    )
    masks = np.concatenate(
        [env.masks[:start], env.masks_opponent[start:end]], axis=0,
    ).astype(bool, copy=False)
    return obs, masks


def frozen_view(env, start: int, end: int):
    obs = np.concatenate(
        [env.obs_opponent[:start], env.obs[start:end]], axis=0,
    )
    masks = np.concatenate(
        [env.masks_opponent[:start], env.masks[start:end]], axis=0,
    ).astype(bool, copy=False)
    return obs, masks


def acting(model, obs, masks, device, sample: bool):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = model(obs_t, mask_t)
        if sample:
            action = Categorical(logits=logits).sample()
        else:
            action = logits.argmax(-1)
    return action, logits, value


def terminal_reward_tank1(env, index: int) -> float:
    outcome = int(env.outcomes[index])
    if outcome == 2:
        return 1.0
    if outcome == 1:
        return -1.0
    if outcome == 3:
        return -0.1
    if outcome == 4:
        return -1.0
    return 0.0


def seat_outcome(env, seat: int, index: int) -> int:
    outcome = int(env.outcomes[index])
    if seat == 1:
        if outcome == 1:
            return 2
        if outcome == 2:
            return 1
    return outcome


def seat_reward(env, seat: int, index: int, double_reward: float) -> float:
    outcome = seat_outcome(env, seat, index)
    if outcome == 1:
        return 1.0
    if outcome == 2:
        return -1.0
    if outcome == 3:
        return double_reward
    if outcome == 4:
        return -1.0
    return 0.0


@torch.no_grad()
def rollout(env, policy, frozen, config, device, start: int, end: int, anchor=None):
    total = end
    steps = config.rollout
    obs = np.empty((steps, total, OBS_DIM), np.float32)
    masks = np.empty((steps, total, BULLET_SLOTS), bool)
    actions = np.empty((steps, total), np.int64)
    logprob = np.empty((steps, total), np.float32)
    values = np.empty((steps, total), np.float32)
    next_values = np.empty((steps, total), np.float32)
    rewards = np.empty((steps, total), np.float32)
    dones = np.empty((steps, total), bool)
    terminals = np.empty((steps, total), bool)
    ref_logits = (
        np.empty((steps, total, ACTIONS), np.float32) if config.kl_coef != 0.0 else None
    )
    outcomes = np.zeros(5, np.int64)
    started = time.time()
    for t in range(steps):
        p_obs, p_mask = policy_view(env, start, end)
        f_obs, f_mask = frozen_view(env, start, end)
        obs[t] = p_obs
        masks[t] = p_mask
        action, logits, value = acting(policy, p_obs, p_mask, device, True)
        actions[t] = action.cpu().numpy()
        logprob[t] = Categorical(logits=logits).log_prob(action).cpu().numpy()
        values[t] = value.cpu().numpy()
        if ref_logits is not None:
            with torch.no_grad():
                ref_source = frozen if anchor is None else anchor
                ref_l, _ = acting(ref_source, p_obs, p_mask, device, False)[1:3]
            ref_logits[t] = ref_l.cpu().numpy()
        f_action, _, _ = acting(frozen, f_obs, f_mask, device, False)
        f_action = f_action.cpu().numpy().astype(np.uint16)
        p_action = actions[t].astype(np.uint16)
        self_action = acting(policy, f_obs, f_mask, device, True)[0]
        self_action = self_action.cpu().numpy().astype(np.uint16)
        self_mask = (np.arange(total) % 4) < int(round(4 * config.self_share))
        a0 = np.empty(total, np.uint16)
        a1 = np.empty(total, np.uint16)
        split = start
        a0[:split] = p_action[:split]
        a1[:split] = np.where(self_mask[:split], self_action[:split], f_action[:split])
        a0[split:] = np.where(self_mask[split:], self_action[split:], f_action[split:])
        a1[split:] = p_action[split:]
        env.step(a0, a1)
        r = np.empty(total, np.float32)
        for i in range(total):
            r[i] = seat_reward(env, 0 if i < split else 1, i, config.double_reward)
        rewards[t] = r
        dones[t] = env.dones.astype(bool)
        terminals[t] = env.terminals.astype(bool)
        for i in range(total):
            if env.dones[i] and env.terminals[i]:
                outcome = seat_outcome(env, 0 if i < split else 1, i)
                outcomes[min(outcome, 4)] += 1
        n_obs, n_mask = policy_view(env, start, end)
        _, _, nv = acting(policy, n_obs, n_mask, device, False)
        next_values[t] = nv.cpu().numpy()
        env.reset_done()
    advantages = np.zeros_like(rewards)
    carry = np.zeros(total, np.float32)
    for t in range(steps - 1, -1, -1):
        alive = 1.0 - dones[t]
        delta = rewards[t] + config.gamma * next_values[t] * alive - values[t]
        carry = delta + config.gamma * config.gae * alive * carry
        advantages[t] = carry
    returns = advantages + values
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "done_count": int(dones.sum()),
        "outcomes": {str(i): int(outcomes[i]) for i in range(5)},
        "sps": float(steps * total / max(1e-9, time.time() - started)),
    }
    batch = {
        "obs": obs.reshape(steps * total, OBS_DIM),
        "masks": masks.reshape(steps * total, BULLET_SLOTS),
        "actions": actions.reshape(steps * total),
        "logprob": logprob.reshape(steps * total),
        "values": values.reshape(steps * total),
        "advantages": advantages.reshape(steps * total),
        "returns": returns.reshape(steps * total),
    }
    if ref_logits is not None:
        batch["ref_logits"] = ref_logits.reshape(steps * total, ACTIONS)
    return batch, metrics


def ppo_update(policy, optimiser, batch, config, device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch["masks"], dtype=torch.bool, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    old_logprob = torch.as_tensor(batch["logprob"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    old_values = torch.as_tensor(batch["values"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-6)
    ref_prob = (
        F.softmax(torch.as_tensor(batch["ref_logits"], dtype=torch.float32, device=device), dim=-1)
        if "ref_logits" in batch else None
    )
    n = len(obs)
    order = np.arange(n)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    updates = 0
    stop = False
    for _ in range(config.epochs):
        if stop:
            break
        np.random.shuffle(order)
        for begin in range(0, n, config.batch):
            sel = torch.as_tensor(order[begin:begin + config.batch], device=device)
            logits, values = policy(obs[sel], masks[sel])
            dist = Categorical(logits=logits)
            new_logprob = dist.log_prob(actions[sel])
            ratio = (new_logprob - old_logprob[sel]).exp()
            unclipped = ratio * advantages[sel]
            clipped = torch.clamp(ratio, 1.0 - config.clip, 1.0 + config.clip) * advantages[sel]
            policy_loss = -torch.min(unclipped, clipped).mean()
            if config.value_clip > 0.0:
                clipped_value = old_values[sel] + (
                    values - old_values[sel]
                ).clamp(-config.value_clip, config.value_clip)
                value_loss = 0.5 * torch.maximum(
                    (values - returns[sel]).square(),
                    (clipped_value - returns[sel]).square(),
                ).mean()
            else:
                value_loss = F.mse_loss(values, returns[sel])
            entropy = dist.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            kl_ref = torch.zeros((), device=device)
            if ref_prob is not None:
                log_softmax = F.log_softmax(logits, dim=-1)
                kl_ref = F.kl_div(log_softmax, ref_prob[sel], reduction="batchmean")
                loss = loss + config.kl_coef * kl_ref
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad)
            optimiser.step()
            with torch.no_grad():
                log_ratio = new_logprob - old_logprob[sel]
                approx_kl = (((log_ratio.exp() - 1.0) - log_ratio).mean())
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["approx_kl"] += float(approx_kl)
            updates += 1
            if config.target_kl > 0.0 and float(approx_kl) > config.target_kl:
                stop = True
                break
    count = max(1, updates)
    return {key: value / count for key, value in stats.items()}


def eval_h2h(policy, baseline, games, envs, seed, device):
    a = evaluate_h2h(policy, baseline, games, envs, seed, device)
    b = evaluate_h2h(baseline, policy, games, envs, seed + 1, device)
    wins = a["wins"] + b["losses"]
    total = a["games"] + b["games"]
    return {
        "as_tank0": a,
        "as_tank1": {
            "games": b["games"],
            "wins": b["losses"],
            "losses": b["wins"],
            "doubles": b["doubles"],
            "draws": b["draws"],
            "win_rate": b["losses"] / max(b["games"], 1),
        },
        "combined": {
            "wins": wins,
            "games": total,
            "win_rate": wins / max(total, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.0,
                        help=">0 enables clipped value loss")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0005)
    parser.add_argument("--kl-coef", type=float, default=0.0,
                        help=">0 enables KL anchor to the frozen model on the policy stream")
    parser.add_argument("--kl-anchor", choices=["frozen", "init"], default="frozen",
                        help="which policy the KL regulariser anchors to")
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-grad", type=float, default=0.5)
    parser.add_argument("--double-reward", type=float, default=-1.0)
    parser.add_argument("--self-share", type=float, default=0.25)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--frozen-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-envs", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/night_seats"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    config = parser.parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "checkpoints").mkdir(exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    policy = load_default(device)
    if config.init_from and config.init_from.exists():
        payload = torch.load(config.init_from, map_location=device, weights_only=False)
        policy.load_state_dict(payload.get("model", payload), strict=False)
        print(f"initialised from {config.init_from}", flush=True)
    policy.train()
    frozen = load_default(device)
    if config.frozen_from and config.frozen_from.exists():
        payload = torch.load(config.frozen_from, map_location=device, weights_only=False)
        frozen.load_state_dict(payload.get("model", payload), strict=False)
        print(f"frozen opponent from {config.frozen_from}", flush=True)
    frozen = frozen.to(device).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    anchor = None
    if config.kl_anchor == "init" and config.kl_coef != 0.0:
        anchor = copy.deepcopy(policy)
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        anchor.eval()
        print("KL anchor = starting weights", flush=True)
    optimiser = torch.optim.Adam(policy.parameters(), lr=config.lr, eps=1e-5)
    env = DuelVec(config.envs, config.seed, OPPONENT_FROZEN, config.threads)
    split = config.envs // 2
    best = -1.0
    started = time.time()
    update = 0
    deadline = started + config.minutes * 60.0
    while time.time() < deadline:
        batch, metrics = rollout(env, policy, frozen, config, device, split, config.envs,
                                 anchor)
        ppo = ppo_update(policy, optimiser, batch, config, device)
        update += 1
        elapsed = time.time() - started
        print(
            f"u{update} steps={update * config.rollout * config.envs:,} "
            f"R={metrics['reward_mean']:.4f} out={metrics['outcomes']} "
            f"pi={ppo['policy_loss']:.4f} vf={ppo['value_loss']:.4f} "
            f"H={ppo['entropy']:.3f} t={elapsed:.0f}s", flush=True,
        )
        with (config.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "update": update, "steps": update * config.rollout * config.envs,
                "rollout": metrics, "ppo": ppo, "elapsed": elapsed,
            }) + "\n")
        if update % config.eval_every == 0:
            policy.eval()
            result = eval_h2h(
                policy, frozen, config.eval_games, config.eval_envs,
                config.seed + 7000 + update, device,
            )
            policy.train()
            print(f"EVAL u{update} {json.dumps(result['combined'])} "
                  f"tank0={result['as_tank0']['win_rate']:.3f} "
                  f"tank1={result['as_tank1']['win_rate']:.3f}", flush=True)
            with (config.output / "evals.jsonl").open("a") as handle:
                handle.write(json.dumps({"update": update, **result}) + "\n")
            score = result["combined"]["win_rate"]
            payload = {
                "model": policy.state_dict(),
                "update": update,
                "total_steps": update * config.rollout * config.envs,
                "h2h": result,
            }
            torch.save(payload, config.output / "checkpoints" / "last.pt")
            if score > best:
                best = score
                torch.save(payload, config.output / "checkpoints" / "best.pt")
                print(f"new best {best:.4f}", flush=True)
    env.close()
    torch.save({
        "model": policy.state_dict(),
        "update": update,
        "total_steps": update * config.rollout * config.envs,
        "best": best,
    }, config.output / "checkpoints" / "final.pt")
    print(f"done best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
