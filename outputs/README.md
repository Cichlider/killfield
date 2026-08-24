# outputs

当前训练格式是 schema-7 static-target fixed-v1 joystick130 PPO：

- `ppo_static_target_fixed_v1_joystick130/nomem/s11/final.pt`：
  `ppo-static-target-fixed-v1-joystick130-nomem-s11` checkpoint；
- 同目录 `config.json`：训练配置；
- 同目录 `complete.json`：固定 Laika 恰好 100 局的最终摘要；
- `last.pt` 与 `metrics.jsonl`：训练过程文件，完成后删除。

当前模型固定 Laika 恰好 100 局为 7 胜、85 负、8 双亡、0 超时，胜率 7%。不设评测 gate，
checkpoint 已直接供 `rl` Viewer review。历史 schema-5 checkpoint 保留但不再暴露。
