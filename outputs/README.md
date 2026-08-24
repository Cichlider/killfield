# outputs

当前 Viewer 暴露六个 joystick130 PPO；前三个是本轮 schema-8 walking 课程：

- `ppo_walking_v6_transition_context_joystick130/nomem/s11/final.pt`：
  `ppo-walking-v6-transition-context-serpentine-joystick130-nomem-s11`，固定曲折道路恰好
  100 局全部到达，平均 208 帧；
- `ppo_walking_v5_waypoint_direction_joystick130/nomem/s11/final.pt`：
  `ppo-walking-v5-waypoint-direction-serpentine-joystick130-nomem-s11`，固定曲折道路恰好
  100 局均在第 138 帧因 waypoint 方向错误失败；
- `ppo_walking_v4_next_direction_joystick130/nomem/s11/final.pt`：
  `ppo-walking-v4-next-direction-serpentine-joystick130-nomem-s11`，固定曲折道路恰好
  100 局全部在 300 帧超时；

以下三个是为行为回看保留的 schema-7 checkpoint；服务端会将 schema-8 Viewer 观测转换为
旧模型需要的输入：

- `ppo_walking_v2_no_stop_joystick130/nomem/s11/final.pt`：
  `ppo-walking-v2-no-stop-serpentine-joystick130-nomem-s11`，固定曲折道路 100 局均在
  第 33 帧撞墙；
- `ppo_walking_v1_joystick130/nomem/s11/final.pt`：
  `ppo-walking-v1-serpentine-joystick130-nomem-s11`，固定曲折道路 100 局全部超时；

- `ppo_static_target_fixed_v1_joystick130/nomem/s11/final.pt`：
  `ppo-static-target-fixed-v1-joystick130-nomem-s11` checkpoint；
- 同目录 `config.json`：训练配置；
- 同目录 `complete.json`：固定 Laika 恰好 100 局的最终摘要；
- `last.pt` 与 `metrics.jsonl`：训练过程文件，完成后删除。

walking-v6 的旁路固定 Laika 恰好 100 局为 5 胜、92 负、3 双亡，胜率 5%；walking-v5
为 4 胜、89 负、7 双亡，胜率 4%；walking-v4 为 2 胜、88 负、10 双亡，胜率 2%。
walking-v2 为 5 胜、87 负、8 双亡，胜率 5%；walking-v1 为 9 胜、88 负、3 双亡，胜率 9%。static-target-v1
为 7 胜、85 负、8 双亡，胜率 7%。均不设评测 gate，checkpoint 直接供 Viewer review。
