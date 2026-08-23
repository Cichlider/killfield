#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."

mode="${1:-train}"
cargo build --release --manifest-path engine/Cargo.toml --lib

case "$mode" in
  smoke)
    .venv/bin/python -u training/train_ppo_paint_v1.py \
      --model nomem --seed 11 --smoke --output /tmp/kf-ppo-paint-v1-smoke
    ;;
  train)
    if command -v caffeinate >/dev/null 2>&1; then
      echo "caffeinate -dimsu 已启用；训练期间阻止常规空闲/系统/磁盘休眠。"
      caffeinate -dimsu .venv/bin/python -u training/train_ppo_paint_v1.py \
        --model nomem --seed 11 --output outputs/ppo_paint_v1_directional16
    else
      echo "警告：系统没有 caffeinate，训练将不带防休眠保护。" >&2
      .venv/bin/python -u training/train_ppo_paint_v1.py \
        --model nomem --seed 11 --output outputs/ppo_paint_v1_directional16
    fi
    ;;
  *)
    echo "usage: $0 [smoke|train]" >&2
    exit 2
    ;;
esac
