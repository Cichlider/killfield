from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from duel_env import (
    BULLET_SLOTS,
    DuelVec,
    OBS_DIM,
    OPPONENT_FROZEN,
    OPPONENT_LAIKA,
)
from hybrid_model import HybridActorCritic


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def collect(args, device):
    model = load(args.model, device)
    initial = load(args.initial, device)
    per_group = args.envs // 3
    counts = [per_group, per_group, args.envs - 2 * per_group]
    offsets = np.cumsum([0] + counts).tolist()
    envs = [
        DuelVec(counts[0], args.seed + 11, OPPONENT_LAIKA, max(1, args.threads // 3)),
        DuelVec(counts[1], args.seed + 22, OPPONENT_FROZEN, max(1, args.threads // 3)),
        DuelVec(counts[2], args.seed + 33, OPPONENT_FROZEN, max(1, args.threads // 3)),
    ]
    opponent_models = [None, initial, model]
    observations, masks, actions, weights = [], [], [], []
    overrides = 0
    total = 0
    for step in range(args.steps):
        obs_parts, mask_parts = [], []
        for env in envs:
            obs_parts.append(env.obs.copy())
            mask_parts.append(env.masks.astype(bool, copy=True))
        obs = np.concatenate(obs_parts, axis=0)
        mask = np.concatenate(mask_parts, axis=0)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
        logits, _ = model(obs_t, mask_t)
        chosen = logits.argmax(-1).cpu().numpy().astype(np.int64)

        teacher = chosen.copy()
        for gi, env in enumerate(envs):
            start, end = offsets[gi], offsets[gi + 1]
            for i in range(counts[gi]):
                safety = env.predictive_safety(
                    i, args.horizon, args.predict_fire != 0
                )
                movement = int(chosen[start + i]) // 2
                current = safety[movement]
                best = int(np.argmax(safety))
                if current < args.threshold and safety[best] > current + args.margin:
                    teacher[start + i] = best * 2 + (int(chosen[start + i]) & 1)

        changed = teacher != chosen
        overrides += int(changed.sum())
        total += len(chosen)
        keep = changed | (np.arange(len(chosen)) % args.record_every == 0)
        for i in np.flatnonzero(keep):
            observations.append(obs[i])
            masks.append(mask[i])
            actions.append(teacher[i])
            weights.append(args.override_weight if changed[i] else args.normal_weight)

        opponent_actions = np.zeros(len(chosen), np.uint16)
        for gi, env in enumerate(envs):
            start, end = offsets[gi], offsets[gi + 1]
            if opponent_models[gi] is None:
                continue
            oo = torch.as_tensor(
                env.obs_opponent.copy(), dtype=torch.float32, device=device
            )
            om = torch.as_tensor(
                env.masks_opponent.astype(bool, copy=True), device=device
            )
            opp, _ = opponent_models[gi](oo, om)
            opponent_actions[start:end] = (
                opp.argmax(-1).cpu().numpy().astype(np.uint16)
            )
        for gi, env in enumerate(envs):
            start, end = offsets[gi], offsets[gi + 1]
            supplied = opponent_actions[start:end] if opponent_models[gi] is not None else None
            env.step(chosen[start:end].astype(np.uint16), supplied)
            env.reset_done()
    for env in envs:
        env.close()
    data = {
        "obs": np.asarray(observations, np.float32),
        "mask": np.asarray(masks, bool),
        "action": np.asarray(actions, np.int64),
        "weight": np.asarray(weights, np.float32),
    }
    meta = {
        "steps": args.steps, "envs": args.envs, "samples": len(actions),
        "overrides": overrides, "override_rate": overrides / max(total, 1),
        "horizon": args.horizon, "threshold": args.threshold,
        "margin": args.margin, "predict_fire": bool(args.predict_fire),
    }
    return data, meta


def train(args, data, device):
    model = load(args.model, device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    obs = torch.as_tensor(data["obs"], device=device)
    mask = torch.as_tensor(data["mask"], device=device)
    action = torch.as_tensor(data["action"], device=device)
    weight = torch.as_tensor(data["weight"], device=device)
    n = len(obs)
    order = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(order)
        total_loss = 0.0
        batches = 0
        for start in range(0, n, args.batch):
            sel = torch.as_tensor(order[start:start + args.batch], device=device)
            logits, _ = model(obs[sel], mask[sel])
            loss = F.cross_entropy(logits, action[sel], reduction="none")
            loss = (loss * weight[sel]).sum() / weight[sel].sum().clamp(min=1e-8)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimiser.step()
            total_loss += float(loss.detach())
            batches += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / max(batches, 1):.4f}", flush=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("outputs/night_run1/checkpoints/hybrid_night_final.pt"))
    parser.add_argument("--initial", type=Path, default=Path("outputs/night_run1/checkpoints/initial.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/night_shield/student_shield.pt"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--envs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--predict-fire", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=8)
    parser.add_argument("--override-weight", type=float, default=4.0)
    parser.add_argument("--normal-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data, meta = collect(args, device)
    np.savez_compressed(args.output.with_suffix(".data.npz"), **data)
    (args.output.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    model = train(args, data, device)
    torch.save({
        "model": model.state_dict(), "meta": meta, "source": str(args.model),
    }, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
