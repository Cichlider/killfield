from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hybrid_model import HybridActorCritic


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/night_run1/checkpoints/hybrid_night_final.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/night_shield/student_balanced.pt"))
    parser.add_argument("--override-weight", type=float, default=0.35)
    parser.add_argument("--normal-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    data = np.load(args.data)
    obs = torch.as_tensor(data["obs"], device=device)
    mask = torch.as_tensor(data["mask"], device=device)
    action = torch.as_tensor(data["action"], device=device)
    original = torch.as_tensor(data["weight"], device=device)
    weight = torch.where(original > 1.0, torch.full_like(original, args.override_weight),
                         torch.full_like(original, args.normal_weight))
    model = load(args.model, device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    n = len(obs)
    order = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(order)
        total = 0.0
        count = 0
        for start in range(0, n, args.batch):
            sel = torch.as_tensor(order[start:start + args.batch], device=device)
            logits, _ = model(obs[sel], mask[sel])
            loss = F.cross_entropy(logits, action[sel], reduction="none")
            loss = (loss * weight[sel]).sum() / weight[sel].sum().clamp(min=1e-8)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimiser.step()
            total += float(loss.detach())
            count += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(count, 1):.4f}", flush=True)
    model.eval()
    torch.save({"model": model.state_dict(), "source": str(args.model),
                "data": str(args.data), "override_weight": args.override_weight,
                "normal_weight": args.normal_weight}, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
