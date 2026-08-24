# PPO 训练与交付

## walking-v2 no-stop 短课程

- 地图：固定 seed `20260825`，`6×3` 单通道蛇形道路，终点为不行动、不射击的 Laika；
- 动作仍为冻结的 `Discrete(130)`；方向动作同一帧瞬间对准并前进，没有原地转向动作；
- 撞墙/侧滑、无位移、位移与车头不一致、开火均 `-10` 并立即终止；
- 最佳连续路径进度累计最多 `+5`，合法帧 `-0.002`，到达终点 `+10`，300 帧超时 `-10`；
- 训练 507,904 步（约为上一版十分之一），不设 gate；
- 同图恰好 100 局：0 到达、100 撞墙、0 超时，平均 33 帧；固定 Laika 旁路报告恰好
  100 局：5 胜、87 负、8 双亡，胜率 5%。

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
