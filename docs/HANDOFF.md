# 接手须知

> 更新于 2026-09-03。这一份讲**现在跑着什么、模型在哪、什么已经查过**；
> O/A/R 的完整定义在 [`DESIGN.md`](DESIGN.md)，配置与逐版结果在
> [`TRAINING.md`](TRAINING.md)，三份已同步。`LESSONS.md` 是历史，
> 里面提到的课程全部退役，保留的是它们证伪的结论。

## 一句话

`rl` 分支现在训练的是 **duel-v1**：真实 Killfield 对战，每局随机迷宫，
奖励只有胜负（外加时间折扣和一个平滑项），对手池是 Laika / MPC / 自己的冻结档。

## O / A / R（当前，schema 20）

**O — 1010 维**，定义在 `engine/src/duel_obs.rs`，偏移量在文件头的常量里。

| 组 | 偏移 | 维度 |
|---|---:|---:|
| 迷宫栅格（补齐 12×10，每格 7 通道） | 0 | 840 |
| 墙射线（16 向，上限 4 格） | 840 | 16 |
| 自身（位姿、速度、角速度、弹槽、开火、存活、撞墙、贴墙） | 856 | 12 |
| 对手（同上，全部转到我车坐标系，含相对朝向） | 868 | 12 |
| 导航（BFS 格数、梯度 one-hot、直线距离、方位、双方死胡同惩罚） | 880 | 10 |
| 我方瞄准辅助 | 890 | 5 |
| 对手瞄准辅助 | 895 | 5 |
| 子弹 10×10（相对位姿速度、归属、已反弹、剩余寿命、弹道预演） | 900 | 100 |
| 威胁（双方 incoming risk、最近通过距离） | 1000 | 3 |
| 回合阶段 + 时钟 | 1003 | 4 |
| 上一动作 | 1007 | 3 |

外加 10 位 bullet mask（批处理元数据，不算语义维度）。全通道有界在 `[-1, 1]`。

**不给**（有回归测试 `the_opponents_internal_state_never_reaches_the_observation`
守着，它会把下面前三项全部打乱并断言观测逐元素不变）：

- 地图 seed 与 RNG 状态
- Laika 的内部目标栈
- 对手当前按键（用速度/角速度代替，由环境按上一帧位姿差分）
- 炮口角度扫描（只给当前角度，找射位必须靠转身学）
- kill field / MPC 对候选的评分

**A — `Discrete(18)`**，`score::CANDIDATES` 的 `[throttle, turn, fire]`，
用引擎自身转速，每帧一决策，无动作承诺、无帧跳。

**R** — 只在终局结算一次（`engine/src/duel.rs`）：

| 结果 | 值 |
|---|---|
| 赢 | ≤10 秒 `+1.0`，10→30 秒**对数**衰减到 `+0.5` |
| 双亡 | `0.0` |
| 输 | `-1.0` |
| 30 秒平局 | `-1.0` |
| **平滑分**（叠加在以上任意结果上） | 动作变更率 ≤13% 得满分 `+0.25`，对数衰减到每帧都换时归零 |

三条数值约束有测试守着，改常数前先看 `the_style_bonus_cannot_outrank_the_result`
和 `the_scale_is_ordered_win_trade_loss_stall`：

- `WIN_FLOOR > 双亡 + STYLE_MAX`（最差的胜必须赢过最好看的双亡）
- `REWARD_WIN > WIN_FLOOR + STYLE_MAX`（快而糙的胜必须赢过慢而美的胜）
- 换命的盈亏平衡点 `(双亡 − 输)/(赢 − 输)` 必须落在 45%–90%

**终局判定只认 `Event::RoundEnd`**。它在首次击杀的 75 帧结算窗口**之后**才发出，
期间在途子弹仍可能把胜利改成双亡。不要在 `Event::Destroy` 或 `alive_count <= 1`
上提前终止。

## 对手池

每局每槽独立掷骰，权重由 `--mix LAIKA MPC FROZEN` 给（当前 `0.4 0.4 0.2`）。

- **Laika**：引擎内部驱动，零接线。
- **MPC**：`KillFieldAgent::new(1, seed)`，**必须设 `OppModel::L1`**——默认的 L2
  会在前视里推演真正的 Laika 脚本，只有对手真是 Laika 时才成立。
- **Frozen**：冻结的策略 checkpoint。它的权重在 Python 里，引擎跑不了神经网络，
  所以环境额外发布 tank 1 视角的观测（`kf_duel_opponent_obs`），训练器过一遍冻结
  网络再把动作从 `kf_duel_step` 的第二个参数喂回去。用**采样**不用 argmax。

## 现在跑着什么

```sh
# 训练（每约 22.9 万步原子发布一次 live.pt + live.json）
.venv/bin/python training/duel_ppo.py --seed 11 --envs 256 --mix 0.4 0.4 0.2 \
  --init-from outputs/pool/duel_gen2.pt --frozen-from outputs/pool/duel_gen2.pt \
  --save-every 200000 --output outputs/ppo_duel_v6

# 网页 + 推理服务，一个进程同源提供
bash viewer/serve.sh          # http://127.0.0.1:8000/
```

`viewer/serve.sh` 的 `RUN` 和 `FROZEN` 是变量，默认指向 v6 和 gen2。

两个进程都用 `start_new_session=True` 起（等价于 setsid，macOS 没有这个命令），
父进程是 1，**关掉终端不影响它们**。日志在 `logs/train.log` 和 `logs/serve.log`。
重起的话：

```python
import subprocess
subprocess.Popen(argv, start_new_session=True,
                 stdout=open("logs/train.log", "w"),
                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
```

被杀掉会丢什么：上次发布之后不到 22.9 万步的进度、**Adam 动量**（`save_live`
只存 `model.state_dict()`，从不存 optimizer）、以及学习率调度的位置。所以每次
重启都是权重热启动而不是真正续训，lr 会从头开始线性衰减——反复重启会让它一直
停在高位。

## 模型在哪

| 文件 | 步数 | 对 Laika（argmax 实测） |
|---|---:|---:|
| `outputs/pool/duel_gen0.pt` | 34.9M | 50.0% |
| `outputs/pool/duel_gen1.pt` | 11.7M | 52.0% |
| `outputs/pool/duel_gen2.pt` | 8.5M | **57.9%** |
| `outputs/ppo_duel_v6/s11/live.pt` | 跑着 | 未测 |

步数在代际间下降是因为每代都热启动重新计数，不是能力下降。
`outputs/*` 在 `.gitignore` 里，不进版本库。

## 各 run 改了什么

每次改奖励都开新 run，好让数字可归因。

| run | 相对上一版的改动 | 冻结档 |
|---|---|---|
| v1 | duel 首版：赢 +1 / 输 −1 / 双亡 **−1** / 平 **−0.3** | 无 |
| v2 | 双亡 **+0.3**、平 **−1**（我第一版把这两个写反了） | gen0 |
| v3 | 冻结档升到 gen1 | gen1 |
| v4 | 赢改为**时间折扣**（10s 满分，对数衰减到 30s 的 0.5） | gen1 |
| v5 | 双亡 **+0.2 → 0** | gen1 |
| v6 | 加**平滑分** `STYLE_MAX = 0.25`，满分线 13% | gen2 |

## 命令速查

```sh
cargo test --manifest-path engine/Cargo.toml         # 60 个测试
./engine/target/release/probe_game 40 20260814       # 游戏与 main 的等价性指纹
./engine/target/release/probe_actions 200 512        # Laika/MPC 的动作变更率
./engine/target/release/bench_mpc 500 512 20260814   # MPC 强度（当前 90.2%）
.venv/bin/python training/bench_duel.py              # 环境吞吐
.venv/bin/python training/eval_duel.py --run outputs/ppo_duel_v6/s11 \
  --frozen outputs/pool/duel_gen2.pt --mix 0.4 0.4 0.2 --episodes 300
```

`eval_duel.py` 跑的是 **argmax**（网页看到的那个），并同时报动作切换率、
转向反向率。训练日志里的 `chg=` 是**采样**策略的变更率，天然偏高，
判断平滑分有没有起效要看 argmax 那个数。

## 已经核查过、不要重复查的事

- **游戏本体与 `main` 逐位相同**。`probe_game` 40 局 Laika 对 Laika 的指纹
  两边都是 `557555f86228e0ff`。`game.rs` 相对 main 的差异全是增量。
- **MPC 强度 90.2%**（500 局 512 rays，种子段 20260814）。main 的三个规划器
  修复已经合进来了。
- **吞吐**：Laika 侧 357k steps/s，50/50 混合 33k，端到端训练约 10–15k。
  引擎单帧 0.003 ms，MPC 决策 0.79 ms（wasm）/ 0.23 ms（原生）。
- **设备**：这个网络（22.6 万参数）在 MPS 上比 CPU 快 36 倍，训练器启动时实测选择。
  但 batch=1 的推理服务用 CPU 更快，已写死。

## 待办

1. 动作抖动：argmax 变更率 41%，Laika 是 13%。平滑分刚上，效果待测。
   如果不够，下一步是**动作承诺**（k 帧内不换）或 CAPS 时间正则，
   两者都是结构层干预，优先于继续加奖励项。
2. 自我对战里双亡占 62.5%，是所有对手里最高的。
3. 训练器不保存 optimizer 状态，续训只能靠权重热启动。要长跑就该存。
4. `AGENTS.md` 要求「恰好 100 局对固定 Laika」的胜率报告，
   `eval_duel.py` 目前是按对手池比例分配局数，需要一个纯 Laika 的 100 局模式。
