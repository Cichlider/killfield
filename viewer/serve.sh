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
if [[ -x ../.venv/bin/python ]]; then
  cd ..
  exec .venv/bin/python training/serve_ppo.py --port "$PORT"
else
  echo "警告：未找到 .venv，页面可以打开，但 PPO checkpoint 推理不可用。" >&2
  python3 -m http.server "$PORT" --bind 127.0.0.1
fi
