# outputs

当前训练格式是 schema-5 directional128 PPO。训练过程文件不提交 Git：

- `ppo_paint_v1/nomem/s11/final.pt`：历史 schema-3 checkpoint，与当前动作不兼容；
- `last.pt`：训练未结束时的断点，完成后可删除；
- `complete.json`：最终评估摘要；
- `metrics.jsonl`：训练曲线。

当前没有可发布的 schema-5 checkpoint。正式结果只在训练完成后对固定 Laika 运行 100 局，
不设评测 gate，行为由网页直接 review。
