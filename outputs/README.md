# outputs

当前训练格式是 schema-6 joystick130 PPO。新 checkpoint 将写入：

- `ppo_paint_v1_joystick130/nomem/s11/final.pt`：
  `ppo-paint-v1-joystick130-nomem-s11` checkpoint；
- 同目录 `config.json`：训练配置；
- 同目录 `complete.json`：固定 Laika 恰好 100 局的最终摘要；
- `last.pt` 与 `metrics.jsonl`：训练过程文件，完成后删除。

历史 schema-5 directional128 checkpoint 保留在
`ppo_paint_v1_directional128/nomem/s11/`，但动作协议不兼容，不再暴露到 Viewer。新模型
尚未正式训练；完成后不设评测 gate，固定 Laika 恰好 100 局结果无论高低都直接网页 review。
