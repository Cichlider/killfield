"""Serve the wasm viewer and the current PPO paint-v1 checkpoint."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from ppo_models import BULLET_SLOTS, OBS_DIM, make_actor_critic


# Older checkpoints use retired action/observation contracts and are not
# exposed as schema-7 static-target joystick130 policies. This entry becomes available
# automatically once the compatible checkpoint exists under --models.
SOURCES = {
    "nomem-s11": {
        "display": "ppo-static-target-fixed-v1-joystick130-nomem-s11",
        "architecture": "nomem",
        "history": 1,
        "checkpoint": "nomem/s11/final.pt",
    },
}


class Models:
    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device
        self.loaded = {}

    def available(self):
        return [
            token for token, source in SOURCES.items()
            if (self.root / source["checkpoint"]).exists()
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
            model = make_actor_critic(source["architecture"]).to(self.device)
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
        obs = torch.tensor(
            [item["obs"] for item in history],
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
        action = int(probabilities.argmax())
        return {
            "action": action,
            "confidence": float(probabilities[action]),
            "probabilities": probabilities.cpu().tolist(),
            "history": len(history),
            "model": source["display"],
            "frame_skip": 1,
            "selection": "argmax",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--models", type=Path, default=Path("outputs/ppo_static_target_fixed_v1_joystick130"))
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
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
