# Killfield

Killfield 是一个用于 Tank Trouble 风格迷宫坦克对战的实时 MPC（Model Predictive
Control）智能体。游戏引擎、物理、弹道和规划器使用 Rust 编写；网页加载同一份
Rust/WASM 引擎，用来观看 Killfield 对战、亲自与它交手，以及观察 AI 自对弈。

**当前基准：** 512 rays 对固定 Laika 运行 2000 局（种子 `20260814–20262813`），
取得 **1771 胜 / 125 负 / 104 双杀 / 0 超时，胜率 88.5%**。手机摇杆在 0%–10%
半径平滑增加转速，在 25%–33% 半径平滑增加移速，随后保持满转速与满移速。

> 一句话概括：Killfield 通常每 4 帧重规划一次，在当前局面的沙盒副本中推演 18 个动作，
> 计算每条未来轨迹的 score；终局结果立即返回，最后执行 score 最高的动作。

## 在线体验

<https://cichlider.github.io/killfield/>

Page 当前提供三种模式：

- **Watch it play**：观看 Killfield 对战固定脚本对手 Laika；
- **Play against it**：使用键盘或手机触控轮盘亲自对战 Killfield；
- **AI vs AI**：观看两个 Killfield MPC 智能体自对弈。

页面还提供 512/256 射线精度、对手预测模型、可保存的摇杆前向对齐范围、玩家轮盘瞬间
转向开关、实时 MPC telemetry 和参数面板。手机端支持横屏、全屏、左侧移动轮盘和右侧
开火。游戏物理以 25 Hz（每帧 40 ms）运行，页面按显示器
刷新率绘制并在相邻物理状态之间插值。

子弹会反弹，并在十秒内持续致命，包括威胁开火者自己。坦克被击毁后世界仍会继续运行三秒，
因此已经飞出的子弹仍可能把一次胜利变成平局。

## Killfield 原理

### 1. 反向弹道密度场

规划器从敌方所在格向外发射一组确定性的反向射线，让射线在迷宫墙壁上反弹。射线经过某个
格子的次数越多，从该位置向敌人开火成功的机会通常越高。一次计算会得到：

- 每个格子的射击密度与相对质量；
- 通往优质射位的 guidance；
- 每个射位较好的炮口方向；
- 子弹到达目标的大致时间。

这张地图只负责告诉 MPC“哪里值得去、朝哪里瞄”。最终是否击中仍由真实游戏弹道模拟决定。

### 2. 推演 18 个动作

动作空间为：

$$
\mathcal A=
\{\text{后退},\text{停止},\text{前进}\}
\times
\{\text{左转},\text{不转},\text{右转}\}
\times
\{\text{不开火},\text{开火}\},
\qquad |\mathcal A|=18.
$$

每次规划会克隆当前游戏状态，在隔离的沙盒中向前推演每个候选。默认前视范围为 36 帧；
普通移动最多保持 4 帧后重规划，纯转向保持 2 帧。遇到可开火窗口、撞墙、命中或终局等事件
会提前重新规划。开火候选会额外比较九种下一帧移动方式，寻找“现在开火以后怎样走”最安全的
continuation。

### 3. Score

没有在推演窗口内出现终局时，动作 $a$ 的分数可以概括为：

$$
\begin{aligned}
S(a)={}&
w_{\Delta F}\Delta F
+w_{F_{\max}}\bigl(F_{\max}-F_0\bigr)_+
+w_G\Delta G
+w_C\Delta C \\
&+w_A\,Q_0K_A\Delta A
+w_M\frac{\lVert\Delta \mathbf x\rVert}{s}
+S_{\mathrm{fire}}
+S_{\mathrm{ammo}} \\
&-w_R\rho
-P_{\mathrm{stuck}}.
\end{aligned}
$$

其中：

- $\Delta F$：沿轨迹向更高弹道密度格移动的增量；
- $F_{\max}-F_0$：推演中到达过的最佳射位相对起点的提升；
- $\Delta G$：沿 guidance 接近优质射位的进展；
- $\Delta C$：连续向更好射位移动形成的 hunt chain；
- $Q_0K_A\Delta A$：当前射位质量、射线集中度和炮口对准改善的乘积；
- $\lVert\Delta\mathbf x\rVert/s$：以格子尺寸 $s$ 归一化的有效净位移；
- $\rho$：推演结束时的来袭子弹风险；
- $S_{\mathrm{fire}}$：预计命中、空枪或自杀枪的得分；
- $S_{\mathrm{ammo}}$：保留弹药的价值；
- $P_{\mathrm{stuck}}$：重复执行无效撞墙动作的惩罚。

主要默认参数如下：

| 项 | 默认值 | 含义 |
|---|---:|---|
| $w_{\Delta F}$ | `34` | Killfield 密度上升 |
| $w_{F_{\max}}$ | `6` | 推演中触及更好射位 |
| $w_G$ | `120` | guidance 进展 |
| $w_C$ | `12` | hunt chain 增益 |
| $w_A$ | `190` | 炮口朝向改善 |
| $w_M$ | `60` | 有效净位移 |
| 预计命中 | `+1800` | 尚未在窗口内实际击杀，但弹道检查预计命中 |
| 空枪 / 自杀枪 | `-260 / -2500` | 浪费弹药或预计击中自己 |
| 弹药储备 | `450` | 子弹越少，继续消耗的代价越高 |
| $w_R$ | `320` | 来袭火力风险 |
| $P_{\mathrm{stuck}}$ | `600` | 重复无效动作 |

弹药项使用对数压力：

$$
S_{\mathrm{ammo}}
=-450\ln\left(\frac{B+1}{b+1}\right),
$$

其中 <var>B</var> 是弹匣容量，<var>b</var> 是剩余可发射子弹数。预计命中的奖励还会根据飞行时间和
弹药稀缺程度衰减，使快速、可靠且节省弹药的射击更有价值。

### 4. 终局优先

如果某条推演轨迹在 36 帧内出现终局，规划器立即停止计算普通 shaping，直接返回终局分数：

$$
S_{\mathrm{terminal}}(a)=
\begin{cases}
12000-8t+S_{\mathrm{ammo}}, & \text{主动击杀敌人},\\
1500-2t+S_{\mathrm{ammo}}, & \text{敌人自行死亡},\\
-12000+t, & \text{自己死亡},
\end{cases}
$$

其中 $t$ 是终局发生的推演帧。主动且更快的击杀会压过普通位置收益，导致自己死亡的动作
则会被强烈排除。最终动作是：

$$
a^*=\arg\max_{a\in\mathcal A} S(a).
$$

击杀后规划器仍会继续推演在途子弹，并选择存活概率更高、与子弹保持更大间距的动作，避免
已经赢下交火后又撞上反弹回来的子弹。

## 本地运行

需要 Rust 工具链以及 `wasm32-unknown-unknown` target：

```sh
rustup target add wasm32-unknown-unknown
bash viewer/serve.sh
```

启动后打开 <http://127.0.0.1:8000>。端口被占用时可以指定其他端口：

```sh
PORT=8001 bash viewer/serve.sh
```

只构建浏览器 WASM：

```sh
bash viewer/build.sh
```

## 项目结构

```text
engine/     Rust 游戏引擎、弹道密度场、MPC sandbox 与 score
viewer/     GitHub Page、本地网页、手机控制和实时参数面板
docs/       设计、开发记录与历史实验结论
training/   暂存的 PPO 基础设施；不是当前 main 分支的开发重点
data/       当前无需离线数据
outputs/    本地实验输出说明；checkpoint 默认不提交 Git
```

## 验证

```sh
cargo test --manifest-path engine/Cargo.toml
node --check viewer/viewer.js
bash viewer/build.sh
```

## PPO 状态

当前 `main` 到此结束 PPO 方向的探索，不在本 README 展开。后续 PPO 工作会从独立 branch
重新开始，避免把训练实验与当前可运行、可解释的 Killfield MPC 混在一起。

## License

MIT，见 [LICENSE](LICENSE)。
