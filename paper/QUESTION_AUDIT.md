# 「一些问题」查证与实验记录

更新日期：2026-09-08。本文档记录交互式论文防御性问题清单所依据的代码核查、补充实验和与旧正文冲突之处。所有胜率均按胜局 / 完成局数计算；`pp` 表示百分点。

## 1. 与旧正文不符的事实

1. 固定 Laika / Killfield 基准不是「原版、无 30 秒超时」规则。`training/eval_duel.py` 使用 `DuelVec`，Rust `duel.rs` 在 30 秒时把对局置为 `Outcome::Draw` 并给学习方 −1.0。主结果中 Killfield 的 43 场非胜局含 3 场这种超时终局，因此旧页面的规则标签与结果明细矛盾，现已统一为「训练规则」。
2. 真人对局使用原版规则，没有 30 秒超时。L_Shy_P 记录中的 12 场「平」是双亡，奖励 −0.1，不是超时，也不重分类为负局。
3. 训练对手并非一个独立的「只判断能否活下来」简化算法。训练池是 Laika / 同一个 512-ray Killfield / 冻结祖先 = 10% / 10% / 80%。24 帧生存 rollout 是 Hybrid 的 `dodge_safety` 输入特征，不是训练对手。
4. 旧页面写的 15,000 帧安全上限不属于产生 97.5% / 92.7% 的 `training/eval_duel.py` 协议，已从主结果元数据移除。

## 2. Q1：评测协议复核

Checkpoint：`outputs/pool/v17b_gs_league_u352.pt`。每个补充格 500 局，argmax，除采样对照外均与主结果使用同一协议。

| 对手 | seed 95300 | seed 96300 | 差值 |
|---|---:|---:|---:|
| Laika | 97.4% | 96.8% | 0.6 pp |
| Killfield | 91.4% | 92.6% | 1.2 pp |

| 对手 | argmax | sampling | argmax − sampling |
|---|---:|---:|---:|
| Laika（seed 95300） | 97.4% | 96.4% | +1.0 pp |
| Killfield（seed 96300） | 92.6% | 93.4% | −0.8 pp |

两个种子基差异均低于预设的 3 pp 警戒线；argmax 在两个对手上的差值方向相反，本样本不支持「argmax 一致高估部署强度」。

## 3. Q2：两个实际计算的定义

### 512-ray Killfield MPC

- 512 条反射射线，最多 2 次反弹，最大飞行 75 帧（3 秒），72 个瞄准桶。
- 规划 horizon 36 帧，动作保持 8 帧；18 个计划（9 个不开火、9 个原地开火后续计划，移动中开火被 mask），取 argmax。
- 移动 / 转向 commit 分别为 4 / 2 帧；出现开火机会时可重规划。
- 分段终局分数：自身死亡 `−12000+t`；主动击杀 `12000−8t+R_a`；对手自杀 `1500−2t+R_a`。
- 非终局分数含峰值风险改善、引导进度、chain gain、开火机会×集中度×对齐增益、位移、贴墙、incoming risk、开火项和弹药保留项。弹药保留 `R_a=−450 ln((capacity+1)/(slots+1))`；命中开火项 `1800−30(1+1.5 scarcity)×flight_frames`，自击 −2500，miss `−260(1+start_relative_density)`。

### Hybrid 的 dodge_safety

- 9 个不开火移动候选，horizon 24 帧；L1 模型固定当前对手按钮；不使用密度场或 512-ray bank。
- 无子弹且对手不开火时全部为 1；存活到 horizon 时为 `clip(min_clearance/8,0,1)`；在第 `e` 帧死亡时为 `−1+0.5e/24`。
- 它输出 9 个 observation / shortcut 特征，最终动作仍由学习策略选择。

## 4. Q3 / Q4：dodge 训练消融与 v17b 推理期消融

### 4.1 已有匹配续训实验

所有 arms 从同一成熟 v11 祖先开始，在同一奖励、对手池和 600M 调度下追加 4,194,304 步；`trained_steps` 从 404,881,408 到 409,075,712；seeds 21 / 22 / 23。它是受控续训，不是从零训练。

| Route | Laika seeds 21 / 22 / 23 | Killfield seeds 21 / 22 / 23 |
|---|---|---|
| Base | 66.4 / 68.0 / 64.5 | 53.0 / 48.9 / 51.9 |
| Read | 78.5 / 75.3 / 73.7 | 60.6 / 56.2 / 58.9 |
| Act | 66.5 / 67.5 / 67.1 | 52.6 / 54.1 / 53.8 |
| Read+Act | 78.0 / 77.1 / 76.1 | 60.5 / 60.8 / 61.4 |
| Shuffled path | 78.6 / 74.4 / 78.5 | 60.3 / 61.1 / 59.7 |

配对差值：Read−Base 为 +9.53 / +7.30 pp（Laika / Killfield）；Act−Base 为 +0.73 / +2.23 pp；Read+Act−Read 为 +1.23 / +2.33 pp；Read+Act−Shuffled 为 −0.10 / +0.53 pp。后两项三种子区间重叠，按不确定处理。训练期 AUC 使用 sampled policy 指标，不能当成严格 fixed-argmax 样本效率曲线。

Shuffled path 使用训练全程固定的零索引置换 `(4,7,1,8,3,0,2,5,6)`。置换只作用于直接 logit 项；共享 observation 中的 9 个 dodge 值保持正确顺序。因此它检验的是正确 action-logit 对应关系相对于一个可被网络重学的固定错位映射是否有增益，不是每步随机噪声对照。

证据结论：Read−Base 的三种子差值区间均不含 0，支持 dodge 信息本身有效；Read+Act−Shuffled 为 −0.10 / +0.53 pp、与零相容，不支持正确动作对齐路由带来可测端点增益。推理期拔掉 v17b dodge 路径的大幅下降只说明当前 checkpoint 已依赖其训练时选择的路径，不把该路径提升为其他训练策略的必经之路。

### 4.2 v17b 推理期干预

同一 checkpoint、同一种子、同一环境；只把指定 shortcut 的最终 logit 贡献置零。每行 500 局 Laika（seed 95300）+ 500 局 Killfield（seed 96300）。

| 置零通路 | Laika | Δ | Killfield | Δ |
|---|---:|---:|---:|---:|
| 无（基线） | 97.4% | — | 92.6% | — |
| dodge | 86.2% | −11.2 | 75.4% | −17.2 |
| ammo | 97.4% | 0.0 | 92.0% | −0.6 |
| shot_quality | 95.2% | −2.2 | 92.4% | −0.2 |
| ammo_lock | 97.4% | 0.0 | 92.4% | −0.2 |
| suicide | 97.4% | 0.0 | 92.6% | 0.0 |
| idle | 97.4% | 0.0 | 92.4% | −0.2 |
| 全部 | 83.2% | −14.2 | 77.2% | −15.4 |

这证明当前 checkpoint 对 dodge logit 通路有明显依赖；不单独回答一个从零训练的 no-dodge 策略最终能达到什么能力。训练阶段的因果证据来自上一节匹配续训实验。

### 4.3 logit 尺度与行为代理

8,192 个真实状态（两类对手各 128 帧 × 32 环境）：

| 项 | mean | std | q01 | q50 | q99 | abs q90 |
|---|---:|---:|---:|---:|---:|---:|
| trunk logits | −1.287 | 9.284 | −22.623 | −2.944 | 22.726 | 15.680 |
| dodge | −0.052 | 3.194 | −14.009 | 0.521 | 4.377 | 4.226 |
| ammo | −0.540 | 0.189 | — | −0.569 | — | 0.750 |
| shot_quality | 0.254 | 0.645 | 0 | 0 | 2.726 | 1.272 |
| ammo_lock | −0.049 | 0.146 | — | 0 | — | 0.114 |
| suicide | −0.272 | 0.362 | — | — | — | 0.796 |
| idle | −0.454 | 1.196 | — | — | — | 2.824 |

`shot_quality` 与 `ammo_lock` 的逐状态贡献 Pearson 相关系数为 +0.08375；虽在各自激活时符号相反，但数据不支持「多数状态互相抵消」。

匹配 ammo 续训（同样 4.19M 步、三 seed）的 Laika / Killfield / bad-fire 均值为：full 76.3 / 60.2 / 25.1%，no_ammo 77.1 / 59.5 / 25.2%，no_eta 75.9 / 60.3 / 24.6%，no_suicide 77.5 / 60.1 / 23.1%，四条全关 76.9 / 58.3 / 23.3%。未见各开火捷径在 trunk 已可读信息上的清晰独立增益。

置零 dodge 后，`(loss+double death)/completed rounds` 从 2.6%→13.8%（Laika）、7.4%→24.6%（Killfield）。在 `min(dodge_safety)<−0.3` 的威胁帧上，所选 movement 等于 argmax safety 的比例从 17.28%→11.93%、20.42%→12.71%。这些是终局受击与有效闪避代理，不是逐 projectile 标注。

原始机器可读结果见 [`question_audit_results.json`](question_audit_results.json)。

## 5. Q5：搜索派生观测维度

- 严格 ballistics/lookahead：aim assist 10 + 每弹预测 30 + threat summary / count 4 + dodge safety 9 = **53 / 1028（5.16%）**。
- 再计 wall rays 16 与 BFS path / gradient / dead-end 7：**76 / 1028（7.39%）**。
- Killfield density-field channels：**0**。

因此正文主张改为「把 action-conditioned lookahead 输出作为观测信息带来可测增益；额外的动作对齐 logit 路由没有可测增益」。同时保留更窄的系统描述：「学习策略在消费规划器式中间产物后超越完整规划器」，不再写「学习超越规划」。

## 6. Q6：平滑奖励查证

没有找到只增大终局 `[0,0.25]` 动作平滑 reward 的干净对照。历史记录称该 bonus 在八次 run 中未见效果，但被 restart 等因素混淆。正文将其标成基于梯度路径时延 / 方差的论证，而非实验结论。

## 7. Q7：PPO 边界核查

- 采样顺序是 `env.step(action)` 后把 `env.dones` 写入 `done_buf[t]`，Rust 在该 transition 后立即置 terminal。因此 `alive=1-done_buf[t]` 与本 buffer 约定一致；CleanRL 的 `dones[t+1]` 是另一种存储约定。
- buffer 不区分 termination / truncation。30 秒超时是主动设计的 −1.0 MDP terminal，而非外部 TimeLimit truncation；本任务语义正确，原样迁移到标准 Gymnasium TimeLimit 不正确。
- value loss 无 clip；内部 `0.5*MSE` 后再乘外层 0.5，实际为 0.25×MSE。
- 无 reward normalization；观测用 Rust 固定物理范围缩放，无 running mean/std。
- GAE 双循环对拍、λ=0、单状态收敛与 LunarLander/CleanRL 集成对照尚未完成，不能据此宣称 PPO 栈已经独立验证。

## 8. Q8：诊断量

历史上没有记录 `approx_kl` / `clipfrac`，entropy 只取最后一个 minibatch，不能可靠回填。`training/duel_ppo.py` 已改为每个 update 聚合 entropy、policy loss、value loss、approximate KL 和 clip fraction，供后续训练使用。当前分支代码与本机后期 gate checkpoint 的结构不一致，尝试回放时 checkpoint 出现新增 gate 参数与 scalar width 不匹配；未把失败的回放包装成历史曲线。

## 9. Q9：消融调度

dodge、ammo、distillation 已完成 arms 的起点、终点、步数、600M 调度和 seeds 21/22/23 均一致；不存在 arms 进入不同退火区间的问题。样本仍只有三个续训 seed。v7 源于一次误把 LR 重置到初值，正文主动披露。

## 10. Q10–Q12：边界核查

- 未运行 frozen-v17b best-response / exploiter；真人未发现稳定 counter 不能排除梯度可达漏洞。
- L_Shy_P 组记录为 80–24–12，12 是双亡；真人规则无 30 秒超时。更广真人记录的 104 与 258 场分胜负窗口胜率及区间为 76.9% [69%,85%]、77.5% [72%,83%]，但同会话局不独立，且属于 v16 而非 v17b。
- 正式评测零帧延迟；viewer 的 3 帧为 120ms，尚无正式对照。零延迟训练策略在三帧下运行属于分布偏移；干净方案是训练时 delay domain randomisation 或把 delay 写入 observation。
