from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_model import HybridActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--init", type=Path, default=Path(
        "outputs/night_run1/checkpoints/hybrid_night_final.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/night_crd/checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--soft-weight", type=float, default=0.5)
    parser.add_argument("--regret-alpha", type=float, default=2.0)
    parser.add_argument("--freeze-trunk", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    data = np.load(args.data)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(data["masks"], dtype=torch.bool, device=device)
    teacher_actions = torch.as_tensor(data["action"], dtype=torch.long, device=device)
    q = torch.as_tensor(data["q"], dtype=torch.float32, device=device)
    q = q - q.max(dim=1, keepdim=True).values
    soft = torch.softmax(q / args.temperature, dim=1)
    one_hot = F.one_hot(teacher_actions, num_classes=q.shape[1]).float()
    target = args.soft_weight * soft + (1.0 - args.soft_weight) * one_hot
    payload = torch.load(args.init, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    model = model.to(device)
    if args.freeze_trunk:
        frozen = ("map0.", "map2.", "map5.", "bullets0.", "bullets2.",
                  "scalars0.", "trunk0.")
        for name, parameter in model.named_parameters():
            if name.startswith(frozen):
                parameter.requires_grad_(False)
    model.train()
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, eps=1e-5,
    )
    n = len(obs)
    order = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(order)
        total = 0.0
        count = 0
        for begin in range(0, n, args.batch):
            sel = torch.as_tensor(order[begin:begin + args.batch], device=device)
            logits, _ = model(obs[sel], masks[sel])
            logp = F.log_softmax(logits, dim=1)
            student_action = logits.argmax(dim=1)
            q_student = q[sel].gather(1, student_action[:, None]).squeeze(1)
            q_best = q[sel].max(dim=1).values
            regret = (q_best - q_student).clamp(min=0.0, max=1.0)
            weight = 1.0 + args.regret_alpha * regret
            loss = -(target[sel] * logp).sum(dim=1)
            loss = (loss * weight).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimiser.step()
            total += float(loss.detach())
            count += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(count, 1):.4f}")
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "data": str(args.data),
        "temperature": args.temperature,
        "soft_weight": args.soft_weight,
        "regret_alpha": args.regret_alpha,
    }, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
