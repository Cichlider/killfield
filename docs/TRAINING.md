# Paint-v1 PPO 训练与交付

## 固定配置

| 项目 | 配置 |
|---|---|
| 下一模型名 | `ppo-paint-v1-joystick130-nomem-s11` |
| 对手 | 固定 Laika |
| 网络 | schema-6 frame encoder + MLP-256 Actor-Critic，无记忆；单一 `Discrete(130)` head |
| Action | `0..127` 轮盘方向、`128` 原地开火、`129` 停止；方向瞬转，支持轮盘倒车分区 |
| 动作频率 | 25 Hz，每引擎帧一次 |
| 并行环境 | 64 |
| rollout | 每环境 256 步 |
| 总步数 | 5,000,000 |
| seed | 11 |
| PPO | lr `3e-4`、gamma `0.9975`、GAE `0.95`、clip `0.2` |

## Reward

进入不同格子时切换染色。`n` 个当前染色格的状态分数为 `S(n)=2^n-1`，每次奖励
`S(n_next)-S(n)`。胜利 `+20`，双亡和超时 `0`，失败 `-20`。没有其他 reward。

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

- schema-6 `ppo-paint-v1-joystick130-nomem-s11`：动作协议和训练链路已实现并通过集成测试，
  尚未开始正式训练；完成后仍将对固定 Laika 只评估恰好 100 局并直接接入 Viewer；
- 历史 schema-5 `ppo-paint-v1-directional128-nomem-s11`：5,013,504 步；固定 Laika 恰好
  100 局为 1 胜、90 负、9 双亡、0 超时，胜率 1%。因旧双 head 动作协议已取消，不再
  出现在 Viewer 模型列表；
- 历史 schema-3 `ppo-paint-v1-nomem-s11`：5,013,504 步；固定 Laika 100 局为
  0 胜、95 负、5 双亡；它使用已取消的联合 18 动作，不再出现在网页模型选项中。
