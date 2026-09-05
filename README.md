# Killfield · `rl` 分支

在 Tank Trouble 风格的迷宫坦克对战里，用 PPO 训练一个 actor-critic 策略 **Hybrid**：
它直接从游戏局面学习移动、瞄准和开火，同时用几条零初始化的捷径接入少量可精确计算的
局部物理特征（比如某个移动方向接下来是否安全），在不改变模型初始行为的前提下帮它更快
学会危险回避。

引擎、物理、弹道用 Rust 写；训练器和网页 Viewer 链接**同一份 crate**，网页上看到的就是
训练时跑的那套物理，没有第二套 JS 实现。`main` 分支是一个零训练的实时 MPC 规划器
**Killfield**，对 Laika 的 500 局 512-ray 基准是 **90.7%**——它既是 Hybrid 的对手，
也是这条 RL 线的性能参照。

当前 checkpoint 是 **v16**（run 步 `29,982,720`，血统累计 **`477,528,064`**）：
1000 局确定性 argmax 评估中，对 Laika **94.1%**，对 512-ray Killfield **81.8%**；
并接受过该游戏资深玩家的真人长局测试。完整训练过程、逐版 loss / actor 改动和交互式
可视化见 **[项目报告](https://cichlider.github.io/killfield/paper/)**。

```sh
python3 -m venv .venv && .venv/bin/pip install -r training/requirements.txt
bash viewer/serve.sh          # http://127.0.0.1:8000/
```

接手请先读 [`docs/HANDOFF.md`](docs/HANDOFF.md)。

---

## O / A / R

当前课程是 **duel-v1**：真实对战，每局随机迷宫，双方都能开火，30 秒未分胜负判平。

| | |
|---|---|
| Observation | schema 24，1028 维（+10 位 bullet mask）。原则是只给「关于世界的事实」，不给「关于决策的答案」——地图、双方状态、子弹弹道、导航、9 维 `dodge_safety` 生存预演；不给种子、对手内部目标、对手按键、炮口角度扫描 |
| Action | `Discrete(18)` = 3 档油门 × 3 档转向 × 开火，与 MPC 同一套候选集，25 Hz 每帧一决策，无动作承诺 |
| Reward | 只在终局结算一次：赢（10 秒内满分，10–30 秒对数衰减到一半）/ 双亡 `-0.1` / 输或 30 秒平局 `-1.0`，叠加一个动作平滑分 |
| 对手池 | Laika 10% / MPC 10% / 冻结自我联赛 80%（7 档，越新越常抽，防止只克制父辈、忘掉祖辈） |

完整定义、每个常数的来历（比如平滑分满分线 13% 是从 Laika/MPC 实测的动作变更率反推
出来的）见 [`docs/DESIGN.md`](docs/DESIGN.md)。为什么放弃逐帧奖励塑形、之前四轮课程
分别被什么 credit assignment 问题推翻，见 [`docs/LESSONS.md`](docs/LESSONS.md)。

## 训练与结果

16 个训练版本的迭代——每一版做了什么、loss 公式怎么改的、actor 加了什么捷径、
对手池怎么变——完整记在 [`docs/TRAINING.md`](docs/TRAINING.md)；当前状态、正在跑
什么、已知的坑记在 [`docs/HANDOFF.md`](docs/HANDOFF.md)。同一段历史的可视化版本
（逐帧观测、动作、16 版血统矩阵）见 [项目报告](https://cichlider.github.io/killfield/paper/)。

## 验证

```sh
cargo test --manifest-path engine/Cargo.toml            # 65 个测试
./engine/target/release/probe_game 40 20260814          # 游戏与 main 的等价性指纹
./engine/target/release/probe_actions 200 512           # Laika / MPC 的动作变更率
./engine/target/release/bench_mpc 500 512 20260814      # MPC 强度
node --check viewer/viewer.js && bash viewer/build.sh
```

`probe_game` 逐帧哈希 40 局 Laika 对 Laika 的事件流、双方位姿和全部子弹。它在 `rl`
和 `main` 上都输出 `557555f86228e0ff`——这是「训练时跑的物理和 main 一样」的证据，
不是断言。

## 相关文档

- [docs/HANDOFF.md](docs/HANDOFF.md) —— **接手先读这一份**：当前状态、跑着什么、模型在哪
- [docs/DESIGN.md](docs/DESIGN.md) —— O/A/R 的完整定义
- [docs/TRAINING.md](docs/TRAINING.md) —— 训练配置与逐版结果
- [docs/LESSONS.md](docs/LESSONS.md) —— 前作 41 个实验阶段的已证伪结论
- [docs/WEB-RUNTIME.md](docs/WEB-RUNTIME.md) —— Rust/WASM 网页运行时取舍
- [项目报告](https://cichlider.github.io/killfield/paper/) —— 交互式：观测/动作实时可视化、loss 与 actor 逐版演化、16 版训练血统

## License

见 [LICENSE](LICENSE)。
