"""Serve the viewer and the newest published checkpoint from one origin.

Two jobs, deliberately in one process so the page's relative `/api/...` fetches
work without CORS: static files out of `viewer/`, and inference against
whatever the trainer last published.

The design constraint that shapes everything here is a bug this project already
shipped once: a checkpoint exported at 1980 dimensions was served against a
2011-dimension front end, the reader used the manifest's column count on the
wider observation, and the page silently rendered a model that had never been
trained on what it was being fed. Nothing errored. So:

* the trainer stamps `schema_version`, `obs_dim` and `action_count` into the
  manifest next to the weights;
* the page reads the same three numbers out of the wasm engine it is actually
  running;
* every `/api/act` carries the page's three numbers, and any disagreement is a
  409 with the mismatch spelled out.

A wrong model is never served quietly. It is refused loudly, or it is right.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

import duel_ppo
import range_ppo

# Manifest `arch` -> the network that checkpoint's weights belong to. A run
# publishes its own architecture name, so an old checkpoint never gets loaded
# into a newer class that happens to be the default, and a curriculum that has
# been retired keeps replaying correctly.
ARCHITECTURES = {
    "duel_cnn_v1": duel_ppo.ActorCritic,
    "range_mlp_v1": range_ppo.ActorCritic,
}

MAX_BODY_BYTES = 1 << 20


class Publication:
    """The newest checkpoint in a run directory, reloaded when it changes."""

    def __init__(self, run: Path):
        self.run = run
        self._cached: tuple[int, torch.nn.Module] | None = None

    # -- manifest ---------------------------------------------------------
    def _paths(self) -> tuple[Path, Path, str] | None:
        """Prefer what training is publishing right now; fall back to a
        finished run so a completed checkpoint can be watched between
        sessions. Both require the manifest that describes them — weights
        whose observation layout is unknown are exactly what must never be
        served, so an unaccompanied checkpoint is simply not offered."""
        manifest = self.run / "live.json"
        if not manifest.exists():
            return None
        for name in ("live.pt", "final.pt"):
            weights = self.run / name
            if weights.exists():
                return weights, manifest, name
        return None

    def manifest(self) -> dict | None:
        """Re-read from disk every call. The file is tiny and this is what
        makes a browser refresh pick up a new model with no server restart."""
        found = self._paths()
        if found is None:
            return None
        weights, manifest_path, source = found
        try:
            data = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            # A torn read is impossible (the trainer writes via os.replace),
            # so this only happens if someone hand-edits the file.
            return None
        data["source"] = source
        data["mtime"] = weights.stat().st_mtime
        return data

    # -- weights ----------------------------------------------------------
    def policy(self, manifest: dict, device: torch.device) -> torch.nn.Module:
        weights, _, _ = self._paths()
        stamp = weights.stat().st_mtime_ns
        if self._cached is not None and self._cached[0] == stamp:
            return self._cached[1]

        arch = manifest.get("arch")
        if arch not in ARCHITECTURES:
            raise KeyError(f"unknown architecture {arch!r} in {weights}")
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        model = ARCHITECTURES[arch]().to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self._cached = (stamp, model)
        steps = manifest.get("steps")
        print(f"loaded {weights.name} · arch={arch} · steps={steps}", flush=True)
        return model


def schema_mismatches(manifest: dict, claim: dict) -> list[str]:
    """Every way the page's engine disagrees with the checkpoint, in words."""
    problems = []
    for key in ("schema_version", "obs_dim", "action_count"):
        want, got = manifest.get(key), claim.get(key)
        if want is None or got is None:
            problems.append(f"{key}: manifest={want} page={got} (missing)")
        elif int(want) != int(got):
            problems.append(f"{key}: checkpoint={want} page={got}")
    return problems


class Handler(SimpleHTTPRequestHandler):
    publication: Publication
    device: torch.device
    frozen: "Frozen"

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.path.split("?")[0] == "/api/model":
            manifest = self.publication.manifest()
            if manifest is None:
                self._json(404, {"error": "no checkpoint published yet",
                                 "run": str(self.publication.run)})
            else:
                manifest["frozen"] = self.frozen.describe()
                self._json(200, manifest)
            return
        super().do_GET()

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/api/act":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"error": f"bad Content-Length {length}"})
            return
        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"malformed JSON: {exc}"})
            return

        manifest = self.publication.manifest()
        if manifest is None:
            self._json(404, {"error": "no checkpoint published yet"})
            return

        problems = schema_mismatches(manifest, request)
        if problems:
            # 409, not a truncation. See the module docstring.
            self._json(409, {"error": "schema mismatch", "problems": problems})
            return

        obs_dim, slots = int(manifest["obs_dim"]), int(manifest["bullet_slots"])
        flat = request.get("obs")
        if not isinstance(flat, list) or len(flat) != obs_dim + slots:
            self._json(400, {
                "error": f"expected {obs_dim + slots} floats "
                         f"(obs {obs_dim} + mask {slots}), got "
                         f"{len(flat) if isinstance(flat, list) else type(flat).__name__}"
            })
            return

        try:
            model = self.publication.policy(manifest, self.device)
        except (KeyError, RuntimeError) as exc:
            self._json(500, {"error": str(exc)})
            return

        def seat(flat_obs, net):
            obs = torch.tensor([flat_obs[:obs_dim]], dtype=torch.float32,
                               device=self.device)
            mask = torch.tensor([[v > 0.5 for v in flat_obs[obs_dim:]]],
                                dtype=torch.bool, device=self.device)
            with torch.inference_mode():
                logits, _ = net(obs, mask)
                probabilities = torch.softmax(logits, dim=-1)[0]
            index = int(torch.argmax(probabilities))
            return index, float(probabilities[index])

        action, confidence = seat(flat, model)
        reply = {
            "action": action,
            "confidence": confidence,
            "steps": manifest.get("steps"),
            "source": manifest.get("source"),
        }

        # Both seats in one round trip. Two requests a frame would double the
        # latency for no reason, and would let the seats drift a frame apart.
        opponent = request.get("opponent_obs")
        if opponent is not None and self.frozen.model is not None:
            if not isinstance(opponent, list) or len(opponent) != obs_dim + slots:
                self._json(400, {"error": "opponent_obs has the wrong length"})
                return
            reply["opponent_action"], _ = seat(opponent, self.frozen.model)
        self._json(200, reply)

    def log_message(self, fmt, *args):
        # One line per engine frame at 25 Hz would bury the training output.
        pass


class Frozen:
    """The pool's frozen checkpoint, loaded once and never reloaded.

    Its whole value is being a fixed rung: if it drifted, "the live model is
    beating the frozen one" would stop meaning the live model improved.
    """

    def __init__(self, path: Path | None, device: torch.device):
        self.path = path
        self.manifest = None
        self.model = None
        if path is None:
            return
        manifest_path = path.with_suffix(".json")
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
        arch = (self.manifest or {}).get("arch", "duel_cnn_v1")
        payload = torch.load(path, map_location=device, weights_only=False)
        self.model = ARCHITECTURES[arch]().to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        steps = (self.manifest or {}).get("steps")
        print(f"frozen opponent: {path.name} · arch={arch} · steps={steps}", flush=True)

    def describe(self) -> dict | None:
        if self.model is None:
            return None
        return {"name": self.path.stem, "steps": (self.manifest or {}).get("steps")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path,
                        default=Path("outputs/ppo_duel_v1/s11"),
                        help="directory the trainer publishes live.pt into")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--viewer", type=Path, default=Path("viewer"))
    parser.add_argument("--frozen", type=Path, default=None,
                        help="checkpoint the page can watch the live model "
                             "play against; enables the frozen-self opponent")
    args = parser.parse_args()

    device = torch.device("cpu")  # batch of one; dispatch latency beats FLOPs
    Handler.publication = Publication(args.run.resolve())
    Handler.device = device
    Handler.frozen = Frozen(args.frozen.resolve() if args.frozen else None, device)

    handler = partial(Handler, directory=str(args.viewer.resolve()))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    manifest = Handler.publication.manifest()
    where = f"http://127.0.0.1:{args.port}/"
    if manifest is None:
        print(f"serving {where} · no checkpoint in {args.run} yet "
              f"(start training; the page picks it up on refresh)", flush=True)
    else:
        print(f"serving {where} · {manifest['source']} "
              f"schema {manifest['schema_version']} "
              f"obs {manifest['obs_dim']} steps {manifest.get('steps')}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
