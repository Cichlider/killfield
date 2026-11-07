from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

OBS_DIM = 1028
BULLET_SLOTS = 10
BULLET_OFFSET = 900
BULLET_DIM = 10
DODGE_OFFSET = 1018
DODGE_DIM = 9
IDLE_OFFSET = 1027
ACTIONS = 18
MAP_W = 12
MAP_H = 10
MAP_C = 7
MAP_DIM = MAP_W * MAP_H * MAP_C


class WideActorCritic(nn.Module):
    def __init__(self, hidden: int = 256, trunk: int = 512, bullet: int = 64):
        super().__init__()
        self.hidden = hidden
        self.trunk_width = trunk
        self.bullet_width = bullet
        self.gated_dodge = True
        self.gated_ammo = True
        self.map0 = nn.Conv2d(MAP_C, 16, 3, padding=1)
        self.map2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.map5 = nn.Linear(32 * 6 * 5, hidden)
        self.bullets0 = nn.Linear(BULLET_DIM, bullet)
        self.bullets2 = nn.Linear(bullet, bullet)
        self.scalars0 = nn.Linear(88, hidden)
        self.trunk0 = nn.Linear(2 * hidden + 2 * bullet, trunk)
        self.actor = nn.Linear(trunk, ACTIONS)
        self.critic = nn.Linear(trunk, 1)
        self.dodge0 = nn.Linear(trunk, 2 * 64)
        self.dodge2 = nn.Linear(2 * 64, 1)
        self.ammo0 = nn.Linear(trunk, 2 * 64)
        self.ammo2 = nn.Linear(2 * 64, 4)
        self.dodge_alpha_old = nn.Parameter(torch.zeros(()))
        self.ammo_alpha_old = nn.Parameter(torch.zeros(4))
        self.idle_logit_penalty = nn.Parameter(torch.zeros(()))
        self.dodge_scale_flat = nn.Parameter(torch.zeros(()))
        self.ammo_scale_flat = nn.Parameter(torch.zeros(()))
        self.shot_quality_scale = nn.Parameter(torch.zeros(()))
        self.ammo_lock_scale = nn.Parameter(torch.zeros(()))
        self.suicide_scale = nn.Parameter(torch.zeros(()))
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def features(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        original = obs.shape[:-1]
        flat = obs.reshape(-1, OBS_DIM)
        flat_mask = mask.reshape(-1, BULLET_SLOTS).bool()
        grid = flat[:, :MAP_DIM].reshape(-1, MAP_W, MAP_H, MAP_C).permute(0, 3, 1, 2)
        m = torch.relu(self.map0(grid))
        m = torch.relu(self.map2(m))
        m = torch.tanh(self.map5(m.flatten(1)))
        rows = flat[:, BULLET_OFFSET:BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM]
        rows = rows.reshape(-1, BULLET_SLOTS, BULLET_DIM)
        encoded = torch.relu(self.bullets0(rows))
        encoded = torch.relu(self.bullets2(encoded))
        mb = flat_mask.unsqueeze(-1)
        count = mb.sum(1).clamp(min=1)
        mean = (encoded * mb).sum(1) / count
        peak = torch.amax(encoded.masked_fill(~mb, -torch.inf), dim=1)
        peak = torch.where((~flat_mask.any(1))[:, None], torch.zeros_like(peak), peak)
        scalars = torch.cat((flat[:, MAP_DIM:BULLET_OFFSET], flat[:, BULLET_OFFSET + 100:]), dim=1)
        s = torch.tanh(self.scalars0(scalars))
        joined = torch.cat((m, s, mean, peak), dim=1)
        return torch.tanh(self.trunk0(joined)).reshape(*original, self.trunk_width)

    def logits_and_value(self, obs, mask, dodge=None):
        f = self.features(obs, mask)
        flat_obs = obs.reshape(-1, OBS_DIM)
        if dodge is None:
            dodge = flat_obs[:, DODGE_OFFSET:DODGE_OFFSET + DODGE_DIM]
        else:
            dodge = dodge.reshape(-1, DODGE_DIM)
        logits = self.actor(f)
        scale = self.dodge_alpha_old + self.dodge2(torch.tanh(self.dodge0(f))).squeeze(-1)
        logits = logits + scale.unsqueeze(1) * dodge.repeat_interleave(2, dim=1)
        scales = self.ammo_alpha_old.unsqueeze(0) + self.ammo2(torch.tanh(self.ammo0(f)))
        ammo = flat_obs[:, 863]
        hit = flat_obs[:, 890]
        suicide = flat_obs[:, 891]
        eta = (flat_obs[:, 893] * 3.0).clamp(0.0, 1.0)
        hit_soon = hit * (1.0 - eta)
        fire_bias = (
            scales[:, 0] * ammo
            + scales[:, 1] * hit_soon
            - scales[:, 2] * ((1.0 - ammo) ** 2) * (1.0 - hit_soon)
            - scales[:, 3] * suicide
        )
        odd = torch.arange(ACTIONS, device=logits.device) % 2 == 1
        logits = logits + fire_bias.unsqueeze(1) * odd.unsqueeze(0)
        idle_pressure = ((flat_obs[:, IDLE_OFFSET] * 25.0 - 8.0) / 17.0).clamp(0.0, 1.0)
        idle_penalty = idle_pressure * self.idle_logit_penalty
        logits = logits.clone()
        logits[:, 8] = logits[:, 8] - idle_penalty
        logits[:, 9] = logits[:, 9] - idle_penalty
        value = self.critic(f).squeeze(-1)
        return logits.reshape(*obs.shape[:-1], ACTIONS), value.reshape(*obs.shape[:-1])

    def forward(self, obs, mask, dodge=None):
        return self.logits_and_value(obs, mask, dodge)


def dup_rows(w: torch.Tensor) -> torch.Tensor:
    return torch.cat((w, w), dim=0)


def dup_cols(w: torch.Tensor) -> torch.Tensor:
    return torch.cat((w, w), dim=1)


def dup_both(w: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((w, w), dim=1), torch.cat((w, w), dim=1)), dim=0)


def expand_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key == "map5.weight":
            out[key] = dup_rows(value)
        elif key == "map5.bias":
            out[key] = dup_rows(value)
        elif key == "scalars0.weight":
            out[key] = dup_rows(value)
        elif key == "scalars0.bias":
            out[key] = dup_rows(value)
        elif key in ("bullets0.weight", "bullets0.bias"):
            out[key] = dup_rows(value)
        elif key == "bullets2.weight":
            z = torch.zeros_like(value)
            out[key] = dup_rows(torch.cat((value, z), dim=1))
        elif key == "bullets2.bias":
            out[key] = dup_rows(value)
        elif key == "trunk0.weight":
            z = torch.zeros_like(value)
            cols = torch.cat((
                value[:, :128], z[:, :128],
                value[:, 128:256], z[:, 128:256],
                value[:, 256:288], z[:, 256:288],
                value[:, 288:], z[:, 288:],
            ), dim=1)
            out[key] = dup_rows(cols)
        elif key == "trunk0.bias":
            out[key] = dup_rows(value)
        elif key in ("actor.weight", "critic.weight"):
            z = torch.zeros_like(value)
            out[key] = torch.cat((value, z), dim=1)
        elif key in ("dodge0.weight", "ammo0.weight"):
            z = torch.zeros_like(value)
            out[key] = dup_rows(torch.cat((value, z), dim=1))
        elif key in ("dodge0.bias", "ammo0.bias"):
            out[key] = dup_rows(value)
        elif key in ("dodge2.weight", "ammo2.weight"):
            z = torch.zeros_like(value)
            out[key] = torch.cat((value, z), dim=1)
        else:
            out[key] = value.clone()
    return out


def build_wide(checkpoint: Path, device="cpu") -> WideActorCritic:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = WideActorCritic()
    model.load_state_dict(expand_state(payload.get("model", payload)), strict=False)
    return model.to(device)
