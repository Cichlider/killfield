# outputs

当前 Viewer 暴露三个 schema-7 joystick130 PPO：

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

walking-v2 的旁路固定 Laika 100 局为 5 胜、87 负、8 双亡，胜率 5%；walking-v1 为
9 胜、88 负、3 双亡，胜率 9%。static-target-v1
为 7 胜、85 负、8 双亡，胜率 7%。均不设评测 gate，checkpoint 直接供 Viewer review。
