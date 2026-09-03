# PPO 训练与交付

> 更新于 2026-09-03。当前课程是 **duel-v1**；O/A/R 的完整定义见
> [`docs/DESIGN.md`](DESIGN.md)，接手先读 [`docs/HANDOFF.md`](HANDOFF.md)。
> 之前的 static-target / walking / pursuit / hunt / shooting-range 课程全部退役，
> 它们证伪的结论保留在 [`docs/LESSONS.md`](LESSONS.md)，配置不再保留。

## 契约

| 项 | 值 |
|---|---|
| Observation | schema 20，`1010` 维（+ 10 位 bullet mask） |
| Action | `Discrete(18)`：`[throttle, turn, fire]`，引擎自身转速 |
| 决策频率 | 25 Hz，每引擎帧一次，无动作承诺、无帧跳 |
| 地图 | 每局 `setup_battle` 随机生成 4-12 × 4-10 |
| 终局 | `Event::RoundEnd`；750 帧（30 秒）未分胜负判平 |
| 对手池 | Laika 40% / MPC 40% / 冻结自我 20%，每局每槽独立掷骰 |

## Reward

只在终局结算一次：

| 结果 | 值 |
|---|---|
| 赢 | ≤10 秒 `+1.0`，10→30 秒对数衰减到 `+0.5` |
| 双亡 | `0.0` |
| 输 | `-1.0` |
| 30 秒平局 | `-1.0` |
| 平滑分（叠加在任意结果上） | 变更率 ≤13% 得 `+0.25`，对数衰减到每帧都换时归零 |

无逐帧塑形、无 BFS 距离项、无 LOS / 瞄准答案 / MPC 未来信息。

## 训练配置

| 项 | 值 | 依据 |
|---|---|---|
| 网络 | 迷宫 CNN + 子弹共享编码器（mean/max 掩码池化）+ 标量 MLP → trunk 256 | 226,227 参数 |
| device | 启动时实测 CPU vs MPS 取快的 | 这个网络 MPS 快 36 倍；但 batch=1 的推理服务 CPU 更快，已分别写死 |
| 并行环境 | 256 | |
| rollout | 每环境 128 步（batch 32,768） | |
| epochs / minibatch | 4 / 8 | |
| gamma | `0.999` | 奖励只在终局，回合最长 875 帧；`0.999^875 = 0.42` |
| GAE lambda | `0.95` | |
| clip | `0.2` | |
| **lr** | `3e-4` **线性衰减到 0** | P16：恒定 lr 续训约 2M 步后性能崩塌且永不恢复 |
| **entropy** | `0.01`，随 lr 同步衰减 | 稀疏奖励需要更久的探索 |
| **critic 预热** | 前 20 次 update 冻结 policy 只训 critic | 稀疏终局下 critic 是慢的一半 |
| seed | 11 | |

引擎侧用 `std::thread::scope` 按槽分块并行（不引入 rayon，保持 crate 零依赖）。

## 吞吐实测（M5，10 核 / 4 性能核）

| 配置 | steps/s |
|---|---:|
| 纯环境 100% Laika | 356,892 |
| 纯环境 50/50 Laika/MPC | 33,021 |
| 纯环境 100% MPC | 19,511 |
| **端到端训练**（含策略前向 + PPO 更新） | **10,000–15,000** |

单帧成本：引擎 0.003 ms，观测编码 0.024 ms，MPC 决策 0.79 ms（wasm）/ 0.23 ms
（原生），策略前向 0.285 ms（CPU batch 1）/ 4.8 µs（MPS batch 256）。

## 各 run 的改动与结果

每改一次奖励就开一个新 run，好让数字可归因。步数在代际间下降是因为每代都热启动
重新计数，不是能力下降。

| run | 相对上一版 | 冻结档 | 对 Laika（argmax） | 双亡 |
|---|---|---|---:|---:|
| v1 | duel 首版：双亡 `-1`、平 `-0.3` | 无 | 50.0% | 32.5% |
| v2 | 双亡 `+0.3`、平 `-1`（v1 把这两个写反了） | gen0 | 52.0% | 32.8% |
| v3 | 冻结档升到 gen1 | gen1 | — | — |
| v4 | 赢改为时间折扣 | gen1 | — | — |
| v5 | 双亡 `+0.2 → 0` | gen1 | **57.9%** | 26.2% |
| v6 | 加平滑分 `STYLE_MAX = 0.25` | gen2 | 跑着 | — |

v5 的完整评估（300 局 argmax，对手池 0.4/0.4/0.2）：

| 对手 | 胜 | 负 | 双亡 | 平 |
|---|---:|---:|---:|---:|
| Laika | 57.9% | 15.9% | 26.2% | 0.0% |
| MPC | 35.5% | 33.6% | 29.1% | 1.8% |
| 冻结档 gen1 | 25.0% | 9.4% | 62.5% | 3.1% |

动作切换率 41.2%、转向反向率 12.1%（Laika 分别是 12.9% 和 1.0%）。

参照线：MPC 规划器对 Laika **90.2%**（500 局 512 rays）；
`LESSONS.md` 记的纯 PPO 历史天花板 **36.4%**，本分支近期 checkpoint 是 9% / 7% /
5% / 1%。

## 命令

```sh
# 训练（每约 22.9 万步原子发布 live.pt + live.json）
.venv/bin/python training/duel_ppo.py --seed 11 --envs 256 --mix 0.4 0.4 0.2 \
  --init-from outputs/pool/duel_gen2.pt --frozen-from outputs/pool/duel_gen2.pt \
  --save-every 200000 --output outputs/ppo_duel_v6

# 网页 + 推理服务（同一进程同源提供，RUN / FROZEN 是变量）
bash viewer/serve.sh                     # http://127.0.0.1:8000/

# 确定性评估：argmax，并报动作切换率与转向反向率
.venv/bin/python training/eval_duel.py --run outputs/ppo_duel_v6/s11 \
  --frozen outputs/pool/duel_gen2.pt --mix 0.4 0.4 0.2 --episodes 300

# 工程 smoke，两次 update，不作为行为结论
.venv/bin/python training/duel_ppo.py --steps 8192 --envs 32 --output /tmp/smoke
```

训练日志里的 `chg=` 是**采样**策略的变更率，天然高于 argmax。判断平滑分是否起效
要看 `eval_duel.py` 报的那个数。

## 强制交付方式

见 [`AGENTS.md`](../AGENTS.md)。要点：不设评估门槛、不因指标阻断部署；每个完成的
run 必须在网页 viewer 里可直接观看；胜率报告用恰好 100 局对固定 Laika。

> 当前缺口：`eval_duel.py` 按对手池比例分配局数，还没有「纯 Laika 恰好 100 局」
> 的模式。这是 `AGENTS.md` 的硬要求，需要补。
