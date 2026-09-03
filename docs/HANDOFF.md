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

两个进程都用 `start_new_session=True` 起（等价于 setsid，macOS 没有这个命令），
父进程是 1，**关掉终端不影响它们**。日志在 `logs/train.log` 和 `logs/serve.log`。

```sh
tail -f logs/train.log                  # 训练进度
curl -s localhost:8000/api/model | jq   # 服务端当前在服务哪个 checkpoint
pgrep -f "duel_ppo.py|serve_live.py"    # 还活着吗
```

**睡眠。** 息屏不影响训练，系统睡眠会把它冻住。这台机器 AC 下 `sleep 0`（永不睡）、
电池下 `sleep 1`（闲置一分钟就睡），而让 AC 显示「prevented」的那条断言是 powerd 的
"Prevent sleep while display is on"——息屏那一刻就没了。所以起训练之后挂一个跟着它
生命周期走的 caffeinate：

```python
subprocess.Popen(["caffeinate", "-ims", "-w", str(train_pid)],
                 start_new_session=True)      # 训练退出时它自己退出
```

`caffeinate` 挡不住**合盖**——那是强制睡眠，不是闲置睡眠。要长跑就别合盖。

起进程的配方（`&` 起的会随工具调用的 shell 一起死，必须走这个）：

```python
import subprocess
subprocess.Popen(argv, start_new_session=True,
                 stdout=open("logs/train.log", "w"),
                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
```

---

## 怎么继续训练

### 情况一：什么都不改，接着跑

```sh
.venv/bin/python training/duel_ppo.py --seed 11 --envs 256 --mix 0.4 0.4 0.2 \
  --resume outputs/ppo_duel_v7/s11/resume.pt \
  --frozen-from outputs/pool/duel_gen2.pt \
  --save-every 200000 --output outputs/ppo_duel_v7
```

`--resume` 恢复权重、Adam 状态和调度位置，且**不重跑 critic 预热**。
输出目录可以复用，`metrics.jsonl` 会被truncate，想留历史就换目录。

### 情况二：改了奖励，开新 run

**必须手给 `--trained-steps`，否则学习率会跳回 3e-4。** 这个默认值坑过一次
（见「已知的坑」）。

```sh
# 1. 先记下血统累计步数
python3 -c "
import json,pathlib
t=sum(json.loads(pathlib.Path(f'outputs/ppo_duel_v{v}/s11/metrics.jsonl')
      .read_text().splitlines()[-1])['steps']
      for v in range(1,8) if pathlib.Path(f'outputs/ppo_duel_v{v}/s11/metrics.jsonl').exists())
print(t)"

# 2. 快照当前权重（新 run 会覆盖旧目录的 live.pt）
cp outputs/ppo_duel_v7/s11/live.pt outputs/pool/v7_final.pt

# 3. 开新 run
.venv/bin/python training/duel_ppo.py --seed 11 --envs 256 --mix 0.4 0.4 0.2 \
  --init-from outputs/pool/v7_final.pt --frozen-from outputs/pool/duel_gen2.pt \
  --trained-steps <上面那个数> --schedule-steps 200000000 \
  --save-every 200000 --output outputs/ppo_duel_v8

# 4. 把网页指过去
python3 - <<'EOF'
import pathlib, re
p = pathlib.Path("viewer/serve.sh"); s = p.read_text()
m = re.search(r'RUN="\$\{RUN:-([^}]+)\}"', s); assert m, "RUN default not found"
p.write_text(s[:m.start(1)] + "outputs/ppo_duel_v8/s11" + s[m.end(1):])
EOF
# 重起服务（先杀掉占 8000 的旧进程）
```

改奖励用 `--init-from` 而不是 `--resume` 是对的：奖励变了，旧的 Adam 动量和
value 头本来就该重来，critic 预热也该跑。要保留的只有调度位置。

### 情况三：自博弈升代

冻结档是一根**固定的横杆**，训练期间绝不更新——它要是也在漂，「打赢了 frozen」
就不再意味着变强了。升代 = 把当前权重快照成新一代：

```sh
cp outputs/ppo_duel_v7/s11/live.pt   outputs/pool/duel_gen3.pt
cp outputs/ppo_duel_v7/s11/live.json outputs/pool/duel_gen3.json
# 然后新 run 传 --frozen-from outputs/pool/duel_gen3.pt
# 网页想看新一代对战，还要把 serve.sh 的 FROZEN 也改掉
```

升代前先评估一次，把这一代的实力记进下面的表——不然代际比较就断了。

---

## 怎么评估

**`AGENTS.md` 的硬要求是「恰好 100 局对固定 Laika」。** 这是唯一可以对外报的
胜率口径，不要用其他样本量替代，也不要拿混合池里 Laika 那一栏顶替：

```sh
.venv/bin/python training/eval_duel.py --run outputs/ppo_duel_v7/s11 \
  --mix 1 0 0 --episodes 100 --envs 25
```

看整体画像（含 MPC 和冻结档）时才用混合池：

```sh
.venv/bin/python training/eval_duel.py --run outputs/ppo_duel_v7/s11 \
  --frozen outputs/pool/duel_gen2.pt --mix 0.4 0.4 0.2 --episodes 300
```

三条评估纪律：

1. **跑 argmax，不跑采样。** `eval_duel.py` 默认就是 argmax，和网页一致。
   训练日志里的 `chg=` 是采样策略的变更率，天然高 30 个点，不能拿来判断平滑分。
2. **单次 100 局的标准误约 5%。** 想下结论就换种子多跑几次。实测 v7 的三次分别是
   57.0 / 59.0 / 55.0；而同一模型在混合池里的 Laika 栏是 44.1%（n=127），
   这个差距是抽样噪声，不是 bug。
3. **不要跨 run 比较 argmax 行为**（切换率、局长这类），除非两个 checkpoint 在
   调度上的位置相当。v1–v6 每次重启都把策略重新搅动过。

---

## 模型在哪

`outputs/*` 在 `.gitignore` 里，不进版本库。**只有这台机器上有。**

| 文件 | 对 Laika（100 局 argmax） | 说明 |
|---|---:|---|
| `outputs/pool/duel_gen0.pt` | 50.0% | 第一代冻结档 |
| `outputs/pool/duel_gen1.pt` | 52.0% | 第二代 |
| `outputs/pool/duel_gen2.pt` | 57.9% | **当前对手池用的这个** |
| `outputs/pool/v6_final.pt` | — | v7 的热启动来源 |
| `outputs/ppo_duel_v7/s11/live.pt` | **57.0%** | 跑着，网页服务的就是它 |
| `outputs/ppo_duel_v7/s11/resume.pt` | — | 带 Adam 状态，`--resume` 用 |

各 run 目录里的 `metrics.jsonl` 是完整历史，逐 update 一行。

## 各 run 改了什么

| run | 相对上一版的改动 | 冻结档 | 对 Laika |
|---|---|---|---:|
| v1 | duel 首版：赢 +1 / 输 −1 / 双亡 **−1** / 平 **−0.3** | 无 | 50.0% |
| v2 | 双亡 **+0.3**、平 **−1**（v1 把这两个写反了） | gen0 | 52.0% |
| v3 | 冻结档升到 gen1 | gen1 | — |
| v4 | 赢改为**时间折扣**（10s 满分，对数衰减到 30s 的 0.5） | gen1 | — |
| v5 | 双亡 **+0.2 → 0** | gen1 | **57.9%** |
| v6 | 加**平滑分** `STYLE_MAX = 0.25`，满分线 13% | gen2 | 52.0% |
| v7 | 奖励未变。修调度：血统 79.4M 接在 200M 的 39.7%，lr 从 1.81e-4 起 | gen2 | **57.0%** |

步数在代际间下降是因为每代都热启动重新计数，不是能力下降。

v7 在 340 万步（血统 8240 万）的完整画像，混合池 300 局：

| 对手 | 胜 | 负 | 双亡 | 平 |
|---|---:|---:|---:|---:|
| Laika | 44.1%(n=127) | 26.0% | 29.9% | 0.0% |
| MPC | 41.3% | 33.0% | 24.8% | 0.9% |
| 冻结档 gen2 | 21.9% | 9.4% | **65.6%** | 3.1% |

动作切换率 49.8%、转向反向率 22.6%（Laika 是 12.9% 和 1.0%）。

参照线：MPC 规划器对 Laika **90.2%**；`LESSONS.md` 记的纯 PPO 历史天花板 **36.4%**，
本分支更早的 checkpoint 是 9% / 7% / 5% / 1%。

## 已知的坑

1. **`--init-from` 不给 `--trained-steps` 会把学习率跳回 3e-4。** v1–v6 六次都
   踩了，等于用近似恒定 lr 跑了 7800 万步——而 `LESSONS.md` 记着「恒定 lr 长训会崩
   且不可逆」。没崩是运气。连带后果：熵系数从未离开 0.01，采样变更率一直钉在 83%；
   v6 之前所有跨 run 的 argmax 行为比较都不可靠。
2. **改 `serve.sh` 的 `RUN`/`FROZEN` 用正则匹配，不要字符串替换。** 用
   `.replace()` 替换一个不存在的字符串会静默跳过，结果是网页服务了一个落后两个 run
   的 checkpoint 却把它标成 live。这个坑真的发生过。
3. **新 run 会覆盖旧目录的 `live.pt`**，换目录前先 `cp` 一份到 `outputs/pool/`。
4. **`&` 起的后台进程会随工具调用的 shell 一起死**，必须 `start_new_session=True`。

## 命令速查

```sh
cargo test --manifest-path engine/Cargo.toml         # 60 个测试
./engine/target/release/probe_game 40 20260814       # 游戏与 main 的等价性指纹
./engine/target/release/probe_actions 200 512        # Laika/MPC 的动作变更率
./engine/target/release/bench_mpc 500 512 20260814   # MPC 强度
.venv/bin/python training/bench_duel.py              # 环境吞吐
bash viewer/build.sh                                 # 改了引擎或 viewer.js 之后必跑
```

改了 `engine/` 要 `cargo build --release`（训练用的 dylib）**和** `viewer/build.sh`
（网页用的 wasm）两个都跑，否则两边跑的不是同一份代码。

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

1. **动作抖动。** argmax 变更率 ~50%，Laika 是 13%；转向反向 23% 对 1%。
   平滑分（v6 起）目前没看到效果，但每一次测量都落在一次重启之后，读数不干净。
   **v7 是第一个真正在退火的 run**，等它跑到 1500–2000 万步（血统约 1 亿）再测，
   那才是第一次可信的读数。
   如果那时候还压不下来，下一步不该继续加奖励项——平滑分是**终局奖励**，一局约
   200 帧，从 50% 降到 13% 的全部收益 0.165 摊到每个动作只有约 0.0008，而胜负项
   是 ±1，信噪比差三个数量级。正确的形式是**动作承诺**（k 帧内不换，MPC 就是这么
   做的，`COMMIT_MOVE_FRAMES = 4`）或 **CAPS 时间正则**（加在策略损失上的
   `D(π(·|s_t), π(·|s_{t+1}))`，完全不经过 credit assignment）。按仓库的三级纪律
   （补观测 → 改结构 → 加奖励项），这两个都优先于再加一项。
2. **自我对战 65.6% 是双亡**，所有对手里最高。两份同源权重、同样的冲脸倾向。
   如果要治，先看它是不是集中在长局——若是，说明时间折扣把「拖到后面不如换命」
   变划算了（回合末换命的盈亏平衡点是 67%，回合初是 50%），该让双亡走同一条
   衰减曲线，而不是把赢的曲线压平。
3. **`V(s)` 到底可不可学。** `LESSONS.md` 第 0 节记着六个受控消融里 R² 全在 0 附近、
   「局面几乎不决定胜负」，但训练中 explained variance 稳定在 +0.90 以上。
   这两个数字需要一个解释——很可能「对某个固定策略的结局可预测」和「局面决定胜负」
   不是一回事，但没人验证过。
4. **对手池的权重从没做过消融。** 40/40/20 是拍的。
