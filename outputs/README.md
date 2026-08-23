# outputs

当前训练格式是 schema-5 directional128 PPO。仓库只保留当前 review 所需的模型、配置和
恰好 100 局的结果；断点与逐 update 日志不提交：

- `ppo_paint_v1_directional128/nomem/s11/final.pt`：当前
  `ppo-paint-v1-directional128-nomem-s11` checkpoint；
- 同目录 `config.json`：训练配置；
- 同目录 `complete.json`：固定 Laika 恰好 100 局的最终摘要；
- `last.pt` 与 `metrics.jsonl`：训练过程文件，完成后删除。

当前模型对固定 Laika 运行 100 局为 1 胜、90 负、9 双亡、0 超时，胜率 1%。不设评测
gate，行为由 `rl` 分支网页直接 review。
