#!/usr/bin/env bash
# Build the engine for the browser and serve the viewer. Ctrl-C to stop.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "错误：PORT 必须是 1-65535 的整数，当前为：$PORT" >&2
  exit 2
fi

if command -v lsof >/dev/null 2>&1; then
  occupant="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$occupant" ]]; then
    echo "错误：端口 $PORT 已被占用：" >&2
    echo "$occupant" >&2
    echo >&2
    echo "请停止旧服务，或换一个端口启动：" >&2
    echo "  PORT=8001 bash viewer/serve.sh" >&2
    exit 1
  fi
fi

bash build.sh
echo
echo "  http://127.0.0.1:$PORT"
echo

# One process serves both the page and the inference API, so the page's
# relative /api/... fetches need no CORS. RUN points at the directory the
# trainer publishes live.pt into; it re-reads the manifest on every request, so
# training can start after the server and the page still picks it up.
ROOT="$(cd .. && pwd)"
RUN="${RUN:-outputs/ppo_duel_v7/s11}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 $PYTHON；先建虚拟环境：" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r training/requirements.txt" >&2
  exit 1
fi
# FROZEN is the pool checkpoint the page can watch the live model play against.
# Skipped silently when it does not exist, so a fresh clone still serves.
FROZEN="${FROZEN:-outputs/pool/duel_gen2.pt}"
frozen_args=()
[[ -f "$ROOT/$FROZEN" ]] && frozen_args=(--frozen "$FROZEN")

cd "$ROOT"
exec "$PYTHON" training/serve_live.py --run "$RUN" --port "$PORT" --viewer viewer \
  "${frozen_args[@]}"
