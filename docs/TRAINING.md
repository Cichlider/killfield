# PPO 训练与交付

## pursuit-v10 双 360° 轮盘、右上房间巡游 Laika 课程

- 地图：固定 seed `20260827`，`6×3` 蛇形道路，并把右上角开放为 `2×2` 房间；Laika
  无武器，沿房内 8 个 waypoint 做包含水平、垂直与斜线段的不规则巡游；靠近或追上
  Laika 不结束回合；
- Observation schema 9：地图 840、当前动态 waypoint 方向和 BFS 剩余格数 5、自身 9、
  对手 6、子弹 60、阶段 3、时间 1、上一动作 722，共 1646 维；
- Action 为 `Discrete(722)`：`0..359` 是完整 360° 前进轮盘，`360..719` 是完整 360°
  倒车轮盘，`720` FIRE，`721` STOP；两套轮盘都瞬时对齐，无转速限制；
- STOP、FIRE、撞墙/侧滑、无位移立即 `-10` 并终止；前进/倒车均为合法显式挡位；
  300 帧 horizon 是 truncation，没有追上奖励、到达终止、逐帧成本或超时扣分；
- 每 4 帧精确结算 `-exp(BFS当前 - BFS初始)`；它等价于给 `e^BFS当前` 乘固定系数
  `e^-BFS初始`，避免十几格初始距离造成数值爆炸；
- 从 walking-v6 schema-8 checkpoint 迁移地图/导航编码和旧动作头；用精确 300 帧动态
  路线以 lr `1e-4` 做 200 epochs 方向预训练，随后重建 Adam，以 lr `1e-9`、entropy
  `0` 运行 16,384 PPO 步；无 paint、gate 或 MPC 信息；
- 固定动态图恰好 100 局：100 局均严格完成 300 帧，0 失败，平均 BFS `1.67`、最小
  BFS 均值 `0`、最终 BFS 均值 `1.00`；固定 Laika 旁路恰好 100 局：9 胜、82 负、
  9 双亡、0 超时，胜率 9%。

## walking-v6 transition-context 课程

- 地图：固定 seed `20260825`，`6×3` 单通道蛇形道路，终点为不行动、不射击的 Laika；
- Observation schema 8：地图不再携带整条 path mask；导航只提供当前 waypoint 四方向
  one-hot 和剩余格子数；
- 动作仍为冻结的 `Discrete(130)`；方向动作同一帧瞬间对准并前进，没有原地转向动作；
- 撞墙/侧滑、无位移、偏离当前 waypoint 方向、位移与车头不一致、开火均 `-10` 并终止；
- 最佳连续路径进度累计最多 `+5`，合法帧 `-0.002`，到达终点 `+10`，300 帧超时 `-10`；
- v4 从头 1,015,808 步，v5/v6 各继续 507,904 步，不设 gate；
- v6 同图恰好 100 局：100 到达、0 失败、0 超时，平均 208 帧；固定 Laika 旁路报告
  恰好 100 局：5 胜、92 负、3 双亡，胜率 5%。
- 地图 v2 是未参与训练的固定 `7×4` 单通道道路，seed `20260826`，路线依次为
  右→上→左→上→右（22 格、5 次转弯）。v6 零样本恰好 100 局：100 到达、0 失败、
  0 超时、0 开火，平均 257 帧。Viewer 当前默认在地图 v2 上验收。

## 固定配置

| 项目 | 配置 |
|---|---|
| 模型名 | `ppo-static-target-fixed-v1-joystick130-nomem-s11` |
| 训练对手 | 固定 seed `20260824`，静止且不能开火 |
| 最终评估 | 固定 Laika，恰好 100 局 |
| 网络 | schema-7 frame encoder + MLP-256 Actor-Critic，无记忆；单一 `Discrete(130)` head |
| Action | `0..127` 轮盘方向、`128` 原地开火、`129` 停止；方向瞬转，支持轮盘倒车分区 |
| 动作频率 | 25 Hz，每引擎帧一次 |
| 并行环境 | 64 |
| rollout | 每环境 256 步 |
| 总步数 | 5,000,000 |
| seed | 11 |
| PPO | lr `3e-4`、gamma `0.9975`、GAE `0.95`、clip `0.2` |

## Reward

- 新最短路径纪录：累计最多 `+0.5`；
- 实际发弹：即时 `+0.1`，每局最多 `+0.5`，失败时全部扣回；
- 干净击杀：`10 + 2 × (1 - t_kill/750)`；
- 自杀、双亡、超时：最终严格 `-10`；
- 保留击杀后 75 帧残弹结算，击杀后自杀仍为失败；
- 无 paint、LOS、瞄准答案或 MPC 未来信息。

## 命令

```sh
# 两次 update 的工程 smoke，不作为行为结论
zsh training/run_ppo_paint_v1.sh smoke

# 正式 500 万步；macOS 自动使用 caffeinate -dimsu
zsh training/run_ppo_paint_v1.sh train
```

## 强制交付方式

不设置评测 gate，不以指标阻止模型上网页。正式训练结束后，只对固定 Laika 运行恰好
100 局，记录胜率，然后将 checkpoint 直接接入网页 review 行为。

每次交付必须明确写出模型名、Reward、Observation、Action、Laika 100 局胜率以及网页
启动命令和 URL。根目录 `AGENTS.md` 对所有 agent 重复声明了这条规则。

## 当前训练结果

- schema-9 `ppo-pursuit-v10-room-exp-bfs-joystick722-nomem-s22`：16,384 PPO 步；
  右上房间动态图恰好 100 局均完成 300 帧，平均 BFS `1.67`，固定 Laika恰好 100 局
  胜率 9%；
- schema-9 `ppo-pursuit-v9-two-cell-exp-bfs-joystick722-nomem-s22`：上一版两格往返对照；
  动态图恰好 100 局均完成 300 帧，平均 BFS `5.60`，固定 Laika 恰好 100 局胜率 5%；
- schema-8 `ppo-pursuit-v8-two-cell-exp-bfs-joystick130-nomem-s11`：旧 128 方向协议的
  已完成对照，动态图 100 局均在第 211 帧路线翻向失败；仍保留在 Viewer 供行为对照；
- schema-8 `ppo-walking-v6-transition-context-serpentine-joystick130-nomem-s11`：固定图
  恰好 100 局全部到达，平均 208 帧；未训练地图 v2 零样本恰好 100 局全部到达，
  平均 257 帧；
- schema-8 `ppo-walking-v5-waypoint-direction-serpentine-joystick130-nomem-s11`：固定图
  100 局均在第 138 帧 waypoint 方向错误；
- schema-8 `ppo-walking-v4-next-direction-serpentine-joystick130-nomem-s11`：固定图
  100 局全部超时；
- schema-7 `ppo-walking-v2-no-stop-serpentine-joystick130-nomem-s11`：507,904 步；固定
  曲折道路恰好 100 局全部在第 33 帧撞墙；STOP/无位移已是立即失败；已直接接入 Viewer；
- schema-7 `ppo-walking-v1-serpentine-joystick130-nomem-s11`：507,904 步；固定曲折道路
  恰好 100 局全部超时，模型学会不撞墙、不倒车、不射击，但利用了 STOP；已直接接入 Viewer；
- schema-7 `ppo-static-target-fixed-v1-joystick130-nomem-s11`：5,013,504 步；固定 Laika
  恰好 100 局为 7 胜、85 负、8 双亡、0 超时，胜率 7%。训练完成后直接接入 Viewer，
  本结果不作为部署 gate；
- 历史 schema-5 `ppo-paint-v1-directional128-nomem-s11`：5,013,504 步；固定 Laika 恰好
  100 局为 1 胜、90 负、9 双亡、0 超时，胜率 1%。因旧双 head 动作协议已取消，不再
  出现在 Viewer 模型列表；
- 历史 schema-3 `ppo-paint-v1-nomem-s11`：5,013,504 步；固定 Laika 100 局为
  0 胜、95 负、5 双亡；它使用已取消的联合 18 动作，不再出现在网页模型选项中。
