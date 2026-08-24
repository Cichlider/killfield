"""PPO models and the frozen semantic-observation schema."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


OBS_SCHEMA_VERSION = 6
OBS_DIM = 1289
MAP_DIM = 1080
SELF_OFFSET = 1081
OPPONENT_OFFSET = 1090
BULLET_OFFSET = 1096
BULLET_SLOTS = 10
BULLET_DIM = 6
PHASE_OFFSET = 1156
ACTION_OFFSET = 1159
ACTION_COUNT = 130
FIRE_ACTION = 128
STOP_ACTION = 129
SCALAR_DIM = (BULLET_OFFSET - MAP_DIM) + (OBS_DIM - PHASE_OFFSET)


class FrameEncoder(nn.Module):
    """Encode one frame; bullet pooling is invariant to transport order."""

    def __init__(self) -> None:
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 6 * 5, 128), nn.Tanh(),
        )
        self.bullet_encoder = nn.Sequential(
            nn.Linear(BULLET_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(SCALAR_DIM, 64), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(128 + 64 + 64, 128), nn.Tanh())

    def forward(self, obs: torch.Tensor, bullet_mask: torch.Tensor) -> torch.Tensor:
        original = obs.shape[:-1]
        obs = obs.reshape(-1, OBS_DIM)
        bullet_mask = bullet_mask.reshape(-1, BULLET_SLOTS)
        maps = obs[:, :MAP_DIM].reshape(-1, 12, 10, 9).permute(0, 3, 1, 2)
        map_features = self.map_encoder(maps)
        bullets = obs[:, BULLET_OFFSET:PHASE_OFFSET].reshape(
            -1, BULLET_SLOTS, BULLET_DIM
        )
        bullet_features = self.bullet_encoder(bullets)
        mask = bullet_mask.unsqueeze(-1)
        count = mask.sum(1).clamp(min=1)
        bullet_mean = (bullet_features * mask).sum(1) / count
        bullet_max = torch.amax(
            bullet_features.masked_fill(~mask, -torch.inf), dim=1
        )
        bullet_max = torch.where(
            (~bullet_mask.any(1))[:, None], 0.0, bullet_max
        )
        scalar = torch.cat(
            (obs[:, MAP_DIM:BULLET_OFFSET], obs[:, PHASE_OFFSET:]), 1
        )
        scalar_features = self.scalar_encoder(scalar)
        encoded = self.fusion(
            torch.cat((map_features, bullet_mean, bullet_max, scalar_features), 1)
        )
        return encoded.reshape(*original, 128)


class ActorCritic(nn.Module):
    def __init__(self, recurrent: bool):
        super().__init__()
        self.recurrent = recurrent
        self.encoder = FrameEncoder()
        if recurrent:
            self.memory = nn.GRU(128, 256, batch_first=True)
        else:
            self.memory = nn.Sequential(nn.Linear(128, 256), nn.Tanh())
        self.actor = nn.Linear(256, ACTION_COUNT)
        self.critic = nn.Linear(256, 1)
        self._initialise()

    def _initialise(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.GRU):
                for name, parameter in layer.named_parameters():
                    if "weight_ih" in name:
                        nn.init.orthogonal_(parameter, gain=math.sqrt(2))
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter, gain=1.0)
                    else:
                        nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def initial_hidden(self, batch: int, device):
        if not self.recurrent:
            return None
        return torch.zeros(1, batch, 256, device=device)

    def step(self, obs, bullet_mask, hidden=None, episode_start=None):
        encoded = self.encoder(obs[:, None], bullet_mask[:, None])[:, 0]
        if self.recurrent:
            if hidden is None:
                hidden = self.initial_hidden(len(obs), obs.device)
            if episode_start is not None:
                hidden = hidden * (~episode_start).view(1, -1, 1)
            features, hidden = self.memory(encoded[:, None], hidden)
            features = features[:, 0]
        else:
            features = self.memory(encoded)
        logits = self.actor(features)
        return logits, self.critic(features).squeeze(-1), hidden

    def sequence(self, obs, bullet_mask, hidden=None, episode_start=None):
        """Unroll B×T contiguous sequences, resetting memory at episode starts."""
        batch, steps = obs.shape[:2]
        encoded = self.encoder(obs, bullet_mask)
        if not self.recurrent:
            features = self.memory(encoded)
        else:
            if hidden is None:
                hidden = self.initial_hidden(batch, obs.device)
            if episode_start is None:
                features, hidden = self.memory(encoded, hidden)
            else:
                hidden = hidden * (~episode_start[:, 0]).view(1, batch, 1)
                dirty = episode_start[:, 1:].any(1)
                clean_indices = (~dirty).nonzero().flatten()
                dirty_indices = dirty.nonzero().flatten()
                features = encoded.new_zeros(batch, steps, 256)
                final_hidden = hidden.new_zeros(hidden.shape)
                if len(clean_indices):
                    clean_output, clean_hidden = self.memory(
                        encoded.index_select(0, clean_indices),
                        hidden.index_select(1, clean_indices),
                    )
                    features = features.index_copy(0, clean_indices, clean_output)
                    final_hidden = final_hidden.index_copy(1, clean_indices, clean_hidden)
                if len(dirty_indices):
                    dirty_hidden = hidden.index_select(1, dirty_indices)
                    dirty_encoded = encoded.index_select(0, dirty_indices)
                    dirty_starts = episode_start.index_select(0, dirty_indices)
                    rows = []
                    for t in range(steps):
                        if t:
                            dirty_hidden = dirty_hidden * (~dirty_starts[:, t]).view(
                                1, len(dirty_indices), 1
                            )
                        output, dirty_hidden = self.memory(
                            dirty_encoded[:, t:t + 1], dirty_hidden
                        )
                        rows.append(output)
                    dirty_output = torch.cat(rows, dim=1)
                    features = features.index_copy(0, dirty_indices, dirty_output)
                    final_hidden = final_hidden.index_copy(1, dirty_indices, dirty_hidden)
                hidden = final_hidden
        logits = self.actor(features)
        return logits, self.critic(features).squeeze(-1), hidden


def make_actor_critic(kind: str):
    if kind not in ("nomem", "gru"):
        raise ValueError(f"PPO supports nomem/gru, got {kind}")
    return ActorCritic(recurrent=kind == "gru")
