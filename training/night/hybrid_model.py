from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
DERIVED_DIM = 27


class HybridActorCritic(nn.Module):
    def __init__(self, gated_dodge: bool = True, gated_ammo: bool = True):
        super().__init__()
        self.gated_dodge = bool(gated_dodge)
        self.gated_ammo = bool(gated_ammo)
        self.map0 = nn.Conv2d(MAP_C, 16, 3, padding=1)
        self.map2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.map5 = nn.Linear(32 * 6 * 5, 128)
        self.bullets0 = nn.Linear(BULLET_DIM, 32)
        self.bullets2 = nn.Linear(32, 32)
        self.scalars0 = nn.Linear(88, 128)
        self.trunk0 = nn.Linear(320, 256)
        self.actor = nn.Linear(256, ACTIONS)
        self.critic = nn.Linear(256, 1)
        self.dodge0 = nn.Linear(256, 64)
        self.dodge2 = nn.Linear(64, 1)
        self.ammo0 = nn.Linear(256, 64)
        self.ammo2 = nn.Linear(64, 4)
        self.opp_head = nn.Linear(256, ACTIONS)
        self.derived_up = nn.Linear(DERIVED_DIM, 256)
        # Zero-initialised so every pre-existing checkpoint evaluates identically
        # through the derived-feature path (u0 identity fusion).
        nn.init.zeros_(self.derived_up.weight)
        nn.init.zeros_(self.derived_up.bias)
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

    def load_from_assets(self, manifest_path: str | Path, weights_path: str | Path) -> None:
        manifest = json.loads(Path(manifest_path).read_text())
        raw = np.fromfile(weights_path, dtype="<f4")
        tensors = manifest["tensors"]

        def take(name: str) -> torch.Tensor:
            spec = tensors[name]
            start = int(spec["offset"])
            length = int(spec["length"])
            return torch.from_numpy(raw[start:start + length].copy())

        def linear(layer: nn.Linear, name: str) -> None:
            w = take(name + ".weight").reshape(layer.weight.shape)
            b = take(name + ".bias").reshape(layer.bias.shape)
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)

        def conv(layer: nn.Conv2d, name: str) -> None:
            w = take(name + ".weight").reshape(layer.weight.shape)
            b = take(name + ".bias").reshape(layer.bias.shape)
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)

        conv(self.map0, "map.0")
        conv(self.map2, "map.2")
        linear(self.map5, "map.5")
        linear(self.bullets0, "bullets.0")
        linear(self.bullets2, "bullets.2")
        linear(self.scalars0, "scalars.0")
        linear(self.trunk0, "trunk.0")
        linear(self.actor, "actor")
        self.gated_dodge = bool(manifest.get("gated", {}).get("dodge", False))
        self.gated_ammo = bool(manifest.get("gated", {}).get("ammo", False))
        if self.gated_dodge:
            linear(self.dodge0, "dodge_delta.0")
            linear(self.dodge2, "dodge_delta.2")
            self.dodge_alpha_old.data.copy_(take("dodge_alpha_old").reshape(()))
        else:
            self.dodge_scale_flat.data.copy_(take("dodge_scale").reshape(()))
        if self.gated_ammo:
            linear(self.ammo0, "ammo_delta.0")
            linear(self.ammo2, "ammo_delta.2")
            self.ammo_alpha_old.data.copy_(take("ammo_alpha_old").reshape(4))
        else:
            self.ammo_scale_flat.data.copy_(take("ammo_scale").reshape(()))
            self.shot_quality_scale.data.copy_(take("shot_quality_scale").reshape(()))
            self.ammo_lock_scale.data.copy_(take("ammo_lock_scale").reshape(()))
            self.suicide_scale.data.copy_(take("suicide_scale").reshape(()))
        self.idle_logit_penalty.data.copy_(take("idle_logit_penalty").reshape(()))

    def derived(self, flat: torch.Tensor) -> torch.Tensor:
        rows = flat[:, BULLET_OFFSET:BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM]
        rows = rows.reshape(-1, BULLET_SLOTS, BULLET_DIM)
        eta = rows[:, :, 9]
        reach_me = rows[:, :, 7]
        mine = rows[:, :, 4] > 0.5
        bounced = rows[:, :, 5] > 0.5
        # My own un-bounced rounds cannot hurt me (engine rule); bounced ones can.
        live = (reach_me > 0.5) & (~mine | bounced)
        inv_eta = torch.where(eta < 0.999, 1.0 - eta, torch.zeros_like(eta))
        threat_slot = reach_me * inv_eta
        threat_in = torch.tanh(6.0 * (threat_slot * live).sum(dim=1))
        threat_all = torch.tanh(6.0 * threat_slot.sum(dim=1))
        eta_s = (flat[:, 893] * 3.0).clamp(0.0, 1.0)
        my_win = flat[:, 890] * (1.0 - eta_s)
        my_waste = (1.0 - eta_s) * flat[:, 892]
        opp_eta = (flat[:, 898] * 3.0).clamp(0.0, 1.0)
        their_win = flat[:, 895] * (1.0 - opp_eta)
        opp_ready = flat[:, 876]
        their_threat = their_win * opp_ready
        ammo = flat[:, 863]
        clock = flat[:, 1006]
        clock_s = torch.sin(clock * 6.28318530718)
        clock_c = torch.cos(clock * 6.28318530718)
        tcnt = flat[:, 1010]
        d = flat[:, DODGE_OFFSET:DODGE_OFFSET + DODGE_DIM]
        d_min = d.min(dim=1).values
        d_argmin = d.argmin(dim=1)
        d_max = d.max(dim=1).values
        d_mean = d.mean(dim=1)
        d_std = d.std(dim=1, correction=0)
        safe_onehot = torch.nn.functional.one_hot(d_argmin, DODGE_DIM).to(flat.dtype)
        parts = (
            threat_in[:, None], threat_all[:, None], my_win[:, None], my_waste[:, None],
            their_win[:, None], their_threat[:, None], eta_s[:, None], opp_eta[:, None],
            ammo[:, None], clock_s[:, None], clock_c[:, None], tcnt[:, None],
            d_min[:, None], d_max[:, None], d_mean[:, None], d_std[:, None],
            safe_onehot, flat[:, 890][:, None], flat[:, 895][:, None],
        )
        return torch.cat(parts, dim=1)

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
        f = torch.tanh(self.trunk0(joined))
        f = f + self.derived_up(self.derived(flat))
        return f.reshape(*original, 256)

    def logits_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        dodge: torch.Tensor | None = None,
        want_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.features(obs, mask)
        flat_obs = obs.reshape(-1, OBS_DIM)
        if dodge is None:
            dodge = flat_obs[:, DODGE_OFFSET:DODGE_OFFSET + DODGE_DIM]
        else:
            dodge = dodge.reshape(-1, DODGE_DIM)
        logits = self.actor(f)
        if self.gated_dodge:
            scale = self.dodge_alpha_old + self.dodge2(torch.tanh(self.dodge0(f))).squeeze(-1)
        else:
            scale = self.dodge_scale_flat.expand(logits.shape[0])
        logits = logits + scale.unsqueeze(1) * dodge.repeat_interleave(2, dim=1)
        if self.gated_ammo:
            scales = self.ammo_alpha_old.unsqueeze(0) + self.ammo2(torch.tanh(self.ammo0(f)))
        else:
            scales = torch.stack((
                self.ammo_scale_flat,
                self.shot_quality_scale,
                self.ammo_lock_scale,
                self.suicide_scale,
            )).unsqueeze(0).expand(logits.shape[0], 4)
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
        out = (logits.reshape(*obs.shape[:-1], ACTIONS), value.reshape(*obs.shape[:-1]))
        if want_features:
            return out + (f,)
        return out

    def forward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        dodge: torch.Tensor | None = None,
        want_features: bool = False,
    ):
        return self.logits_and_value(obs, mask, dodge, want_features)


def load_default(device: torch.device | str = "cpu") -> HybridActorCritic:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "viewer" / "assets" / "hybrid.json"
    weights = root / "viewer" / "assets" / "hybrid.bin"
    model = HybridActorCritic()
    model.load_from_assets(manifest, weights)
    return model.to(device)
