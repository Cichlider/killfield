"""Serve the WASM behavior viewer and completed PPO checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from ppo_models import (
    BULLET_SLOTS, OBS_DIM, STOP_ACTION, STOP_FIRE_ACTION, make_actor_critic,
    make_legacy_schema7_actor_critic, make_legacy_schema8_actor_critic,
)


# Completed runs remain selectable so behavior review never silently replaces an
# earlier handoff, so retired protocols are reconstructed by read-only adapters
# rather than dropped.
#
# Schema 10 pairs each of 128 headings with a trigger bit, giving a 258-wide
# action one-hot and a 1182-wide payload. A schema-8 checkpoint reads a 1054
# payload and a Discrete(130) head, so `schema10_to_schema8` collapses the pair
# back down and `LEGACY_130_TO_SCHEMA10` re-expands the chosen action. The old
# 130 protocol's direction buckets carry the same world angles, and every one of
# those runs came from a curriculum that failed the episode on reverse or on
# firing, so nothing is lost in the round trip.
#
# The schema-9 runs are the exception: their 722-wide head came from a protocol
# where reverse was an explicit gear, so collapsing it would silently invent a
# forward action the policy never chose. They stay listed but unservable.
SUPPORTED_SCHEMAS = (7, 8, 10)

SOURCES = {
    "hunt-v3-s11": {
        "display": "ppo-hunt-v3-mixed-selfclosed-bfs-joystick258-nomem-s11",
        "architecture": "nomem", "schema": 10, "history": 1,
        "checkpoint": "outputs/ppo_hunt_v3_mixed_selfclosed_joystick258/nomem/s11/final.pt",
    },
    "hunt-v1-s11": {
        "display": "ppo-hunt-v1-real-maze-exp-bfs-joystick258-nomem-s11",
        "architecture": "nomem", "schema": 10, "history": 1,
        "checkpoint": "outputs/ppo_hunt_v1_real_maze_exp_bfs_joystick258/nomem/s11/final.pt",
    },
    "pursuit-v10-s22": {
        "display": "ppo-pursuit-v10-room-exp-bfs-joystick722-nomem-s22",
        "architecture": "nomem", "schema": 9, "history": 1,
        "checkpoint": "outputs/ppo_pursuit_v10_room_exp_bfs_joystick722/nomem/s22/final.pt",
    },
    "pursuit-v9-s22": {
        "display": "ppo-pursuit-v9-two-cell-exp-bfs-joystick722-nomem-s22",
        "architecture": "nomem", "schema": 9, "history": 1,
        "checkpoint": "outputs/ppo_pursuit_v9_two_cell_exp_bfs_joystick722/nomem/s22/final.pt",
    },
    "pursuit-v8-s11": {
        "display": "ppo-pursuit-v8-two-cell-exp-bfs-joystick130-nomem-s11",
        "architecture": "nomem", "schema": 8, "history": 1,
        "checkpoint": "outputs/ppo_pursuit_v8_two_cell_exp_bfs_joystick130/nomem/s11/final.pt",
    },
    "walking-v6-s11": {
        "display": "ppo-walking-v6-transition-context-serpentine-joystick130-nomem-s11",
        "architecture": "nomem", "schema": 8, "history": 1,
        "checkpoint": "outputs/ppo_walking_v6_transition_context_joystick130/nomem/s11/final.pt",
    },
    "walking-v5-s11": {
        "display": "ppo-walking-v5-waypoint-direction-serpentine-joystick130-nomem-s11",
        "architecture": "nomem", "schema": 8, "history": 1,
        "checkpoint": "outputs/ppo_walking_v5_waypoint_direction_joystick130/nomem/s11/final.pt",
    },
    "walking-v4-s11": {
        "display": "ppo-walking-v4-next-direction-serpentine-joystick130-nomem-s11",
        "architecture": "nomem", "schema": 8, "history": 1,
        "checkpoint": "outputs/ppo_walking_v4_next_direction_joystick130/nomem/s11/final.pt",
    },
    "walking-v2-s11": {
        "display": "ppo-walking-v2-no-stop-serpentine-joystick130-nomem-s11",
        "architecture": "nomem",
        "schema": 7,
        "history": 1,
        "checkpoint": "outputs/ppo_walking_v2_no_stop_joystick130/nomem/s11/final.pt",
    },
    "walking-v1-s11": {
        "display": "ppo-walking-v1-serpentine-joystick130-nomem-s11",
        "architecture": "nomem",
        "schema": 7,
        "history": 1,
        "checkpoint": "outputs/ppo_walking_v1_joystick130/nomem/s11/final.pt",
    },
    "static-kill-v1-s11": {
        "display": "ppo-static-target-fixed-v1-joystick130-nomem-s11",
        "architecture": "nomem",
        "schema": 7,
        "history": 1,
        "checkpoint": "outputs/ppo_static_target_fixed_v1_joystick130/nomem/s11/final.pt",
    },
}


def schema10_to_schema8(obs):
    """Collapse the movement/trigger pair back into the retired Discrete(130)."""
    old = list(obs[:924]) + [0.0] * 130
    active = next((i for i, value in enumerate(obs[924:1182]) if value > 0.5), None)
    if active is not None:
        movement, fire = divmod(active, 2)
        if movement < 128:
            # The old protocol could not drive and shoot at once, so a firing
            # move is reported to the legacy head as its heading.
            old_action = movement
        else:
            old_action = 128 if fire else 129
        old[924 + old_action] = 1.0
    return old


def legacy_130_to_schema10(action):
    """Re-express a Discrete(130) choice in the current protocol."""
    if action < 128:
        return action * 2
    return STOP_FIRE_ACTION if action == 128 else STOP_ACTION


def schema8_to_schema7(obs):
    """Reconstruct the retired path mask so completed schema-7 runs stay reviewable."""
    old = [0.0] * 1170
    cells = [[[0.0] * 8 for _ in range(10)] for _ in range(12)]
    start = goal = None
    for x in range(12):
        for y in range(10):
            source = (x * 10 + y) * 7
            cells[x][y][:7] = obs[source:source + 7]
            if cells[x][y][5] > 0.5:
                start = (x, y)
            if cells[x][y][6] > 0.5:
                goal = (x, y)

    def neighbours(cell):
        x, y = cell
        flags = cells[x][y]
        for nx, ny, wall in ((x, y - 1, 1), (x + 1, y, 2),
                             (x, y + 1, 3), (x - 1, y, 4)):
            if 0 <= nx < 12 and 0 <= ny < 10 and flags[wall] < 0.5 \
                    and cells[nx][ny][0] > 0.5:
                yield nx, ny

    def distances(origin):
        result = {origin: 0}
        queue = deque([origin])
        while queue:
            cell = queue.popleft()
            for nxt in neighbours(cell):
                if nxt not in result:
                    result[nxt] = result[cell] + 1
                    queue.append(nxt)
        return result

    if start is not None and goal is not None:
        from_start, from_goal = distances(start), distances(goal)
        length = from_start.get(goal)
        if length is not None:
            for cell, first in from_start.items():
                if first + from_goal.get(cell, 10**9) == length:
                    cells[cell[0]][cell[1]][7] = 1.0
    for x in range(12):
        for y in range(10):
            target = (x * 10 + y) * 8
            old[target:target + 8] = cells[x][y]
    old[960] = obs[844]
    old[961:970] = obs[845:854]
    old[970:976] = obs[854:860]
    old[976:1036] = obs[860:920]
    old[1036:1039] = obs[920:923]
    old[1039] = obs[923]
    old[1040:1170] = obs[924:1054]
    return old


class Models:
    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device
        self.loaded = {}

    def available(self):
        return [
            token for token, source in SOURCES.items()
            if source["schema"] in SUPPORTED_SCHEMAS
            and (self.root / source["checkpoint"]).exists()
        ]

    def get(self, token):
        if token not in SOURCES:
            raise KeyError(token)
        source = SOURCES[token]
        checkpoint_path = self.root / source["checkpoint"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'{source["display"]} 尚未训练完成')
        stamp = checkpoint_path.stat().st_mtime_ns
        cached = self.loaded.get(token)
        if cached is None or cached[0] != stamp:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            factory = (
                make_legacy_schema7_actor_critic if source["schema"] == 7 else
                make_legacy_schema8_actor_critic if source["schema"] == 8 else
                make_actor_critic
            )
            model = factory(source["architecture"]).to(self.device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.loaded[token] = (stamp, model)
        return self.loaded[token][1], source

    @torch.inference_mode()
    def act(self, token, history):
        model, source = self.get(token)
        history = history[-source["history"]:]
        if not history:
            raise ValueError("empty history")
        obs_rows = [item["obs"] for item in history]
        if source["schema"] < 10:
            obs_rows = [schema10_to_schema8(row) for row in obs_rows]
        if source["schema"] == 7:
            obs_rows = [schema8_to_schema7(row) for row in obs_rows]
        obs = torch.tensor(
            obs_rows,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask = torch.tensor(
            [item["mask"] for item in history],
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        if source["architecture"] == "gru":
            logits, _values, _hidden = model.sequence(obs, mask)
            probabilities = logits[0, -1].softmax(-1)
        else:
            logits, _values, _hidden = model.step(obs[:, -1], mask[:, -1])
            probabilities = logits[0].softmax(-1)
        raw_action = int(probabilities.argmax())
        action = raw_action if source["schema"] == 10 else legacy_130_to_schema10(raw_action)
        return {
            "action": action,
            "confidence": float(probabilities[raw_action]),
            "probabilities": probabilities.cpu().tolist(),
            "history": len(history),
            "model": source["display"],
            "frame_skip": 1,
            "selection": "argmax",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--models", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    models = Models(args.models, device)
    viewer = Path(__file__).resolve().parents[1] / "viewer"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(viewer), **kwargs)

        def json_response(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/models":
                self.json_response(200, {
                    "models": models.available(),
                    "display": {
                        token: source["display"] for token, source in SOURCES.items()
                    },
                    "device": str(device),
                })
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/act":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 8_000_000:
                    raise ValueError("request too large")
                request = json.loads(self.rfile.read(length))
                history = request["history"]
                if any(
                    len(item["obs"]) != OBS_DIM
                    or len(item["mask"]) != BULLET_SLOTS
                    for item in history
                ):
                    raise ValueError("observation shape mismatch")
                self.json_response(200, models.act(request["model"], history))
            except FileNotFoundError as error:
                self.json_response(503, {"error": str(error)})
            except Exception as error:
                self.json_response(400, {"error": str(error)})

        def log_message(self, format, *values):
            if not self.path.startswith("/api/act"):
                super().log_message(format, *values)

    print(f"PPO viewer: http://127.0.0.1:{args.port}")
    print(f"models: {models.available() or 'training in progress'}; inference: {device}")
    retired = [
        token for token, source in SOURCES.items()
        if source["schema"] not in SUPPORTED_SCHEMAS
    ]
    if retired:
        print(f"schema-9 双轮盘 checkpoint 不兼容 Discrete(130)，已从列表隐藏：{retired}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
