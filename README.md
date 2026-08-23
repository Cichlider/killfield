# Tank Trouble PPO

用 PPO 训练 Tank Trouble 智能体。Rust 是唯一仿真引擎；Python 只负责在线采样、模型和
PPO 更新；浏览器查看器加载同一份 wasm 与 PPO checkpoint。

当前只维护一条路线：Observation schema 4 + 16 世界方向/开火双 head + Paint-v1 reward + PPO。

## 浏览器观看

在仓库根目录运行：

```sh
bash viewer/serve.sh
```

脚本会先编译 wasm，再启动 PPO 推理服务。看到启动信息后打开：

<http://127.0.0.1:8000>

网页可选择 Killfield vs Laika、人工游玩和 MPC 自对弈。人工游玩提供十六边形轮盘与独立
FIRE 按钮；方向键同样选择世界方向，空格开火。`P` 暂停，暂停时按 `.` 单帧前进。

如果 8000 端口已被其他程序占用，可以指定另一个端口：

```sh
PORT=8001 bash viewer/serve.sh
```

然后打开 <http://127.0.0.1:8001>。启动脚本会直接显示占用端口的进程。

旧 `ppo-paint-v1-nomem-s11` 使用 schema 3 的联合 18 动作，与当前双 head 不兼容，网页已
停止列出。新的 directional checkpoint 训练完成后再加入模型选项。

如果只想重新编译 wasm、不启动服务：

```sh
bash viewer/build.sh
```

## 当前设计

- **Observation**：1178 个语义数值，包含完整迷宫、我方染色格、双方坦克、全部子弹、
  回合阶段，以及上一次 Movement/Fire 两个独热动作；不给未来模拟或对手内部状态。
- **Action**：两个独立 head：Movement 17 类（16 个世界方向 + STOP），Fire 2 类；每
  `40ms` 推理一次（25 Hz）。底层自动转向，朝向误差不超过 10° 时向前，永不后退。
- **Reward Paint-v1**：进入格子切换染色，`n` 个染色格价值 `2^n-1`；胜 `+20`、
  双亡/超时 `0`、负 `-20`，其余通道关闭。
- **PPO**：新双 head 版本尚未开始正式训练；无评测 gate，训练后固定对 Laika 100 局并上网页。

完整定义见 [docs/DESIGN.md](docs/DESIGN.md)，最小训练步骤见
[docs/TRAINING.md](docs/TRAINING.md)。

## 快速开始

```sh
# 1. 安装 Python 依赖（已有 .venv 时跳过）
python3 -m venv .venv
.venv/bin/pip install -r training/requirements.txt

# 2. 两次 PPO update 的冒烟测试，结果写到 /tmp
zsh training/run_ppo_paint_v1.sh smoke

# 3. 正式训练：无记忆模型、seed 11、500 万步；自动启用 caffeinate
zsh training/run_ppo_paint_v1.sh train

# 4. 浏览器观战
bash viewer/serve.sh
```

## 目录

```text
docs/       O/A/R、训练方案、历史经验与开发事实
engine/     Rust 游戏引擎、语义观测、reward 和训练 FFI
training/   仅 PPO 模型、训练、检查和网页推理服务
viewer/     wasm 查看器和 reward 诊断面板
data/       当前为空；PPO 在线采样，不依赖离线数据
outputs/    本地 PPO checkpoint（Git 忽略）
tools/      Rust 移植与 JS 参考实现的对拍工具
```

## 验证

```sh
cargo test --manifest-path engine/Cargo.toml
.venv/bin/python -m py_compile training/*.py
bash tools/difftest/verify.sh
```

网页当前不列模型 checkpoint，因为唯一旧 checkpoint 的动作 schema 已失效。Killfield vs
Laika、人工游玩和 MPC 自对弈是环境诊断模式，不是训练模型路线。
