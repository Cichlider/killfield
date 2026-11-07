# Overnight session 2026-10-20 → 10-21 06:30

## USER DIRECTIVE (verbatim, must survive every compaction)

> 这是一个长程任务，我不在电脑旁，不用问问题，我完全同意你的想法，结束前先看时间，如果没到就开多agent找创新点和增强或者改进的地方，也可以上网搜索有关论文（clash已开，自己找端口），允许大幅度重置+允许大幅度使用CPU+GPU，直到10/21 6:30，中途不要以为到了6:30就停下，也不要靠等待消磨时间；到达时间前，压缩对话必须必须必须包含这段提示词！到达时间后不再保留

## Session rules
- Do NOT idle while waiting. Check time before ending: keep going until 2026-10-21 06:30 local.
- When this conversation is compacted BEFORE 10/21 06:30, the summary MUST include the
  user directive above verbatim. After 06:30, discard it.
- Heavy CPU/GPU use and big resets are explicitly allowed.
- Never modify deployed assets: viewer/assets/hybrid.{json,bin}, viewer/kf_engine.wasm frozen.
- AGENTS.md handoff rule: report model name / reward / observation / action / exact-100
  Laika win rate / browser review command for anything produced. No promotion gates.

## Environment facts
- Started 2026-10-20 23:37. GPU RTX 4060 8GB CUDA (torch 2.14.0+cu126 on C:\Python314\python.exe).
- FlClash TUN mode, NO local http proxy port; GitHub + export.arxiv.org reachable;
  api.semanticscholar.org + huggingface.co BLOCKED; websearch tool works (server-side).
- Engine DLL: engine/target/release/kf_engine.dll. Training python: C:\Python314\python.exe.
- Baselines (2000-game scale unless noted):
  - v1 reference Hybrid: Laika 96.7% / MPC 89.3% / mirror 43.3%
  - **v2 night_final (deployed, `?model=night`)**: Laika 96.4% / MPC 95.6% / mirror 57.55%;
    exact-100 Laika handoff 99/100; per-seat vs v1 59.9%/55.2%
  - v15 seats 54.55% (200-game 63.5% was noise); v3-v15 all failed vs v2.
- Throughput: solo 5-6k sps (512-1024 envs CUDA); ~2.5k with MPC group.
- night_run1 recipe (u790, 25.9M steps): 256 envs rollout 128, lr 6e-5 decay, kl .02,
  entropy .001, league laika .375/frozen .3125/self .3125, engine default reward
  (loss -1, double -0.1, draw -1, style ≤ 0.25). v2 = soup of run1 checkpoints.

## Working log
- 23:37 started. Journal created.
- 23:50 audit done (sub-agent): fixed dagger_collect.py (2 bugs); added --mpc-share league;
  --group-advnorm; train_seats --value-clip/--kl-coef/--target-kl; teacher_search
  --depth/--reward-mode/--threads; NEW measure_actions.py.
- 00:05 v1 behavior histogram (argmax, 200 mirror games): top action back+right+fire
  12.1%; suicide-length mean 0.32 vs enemy-hits 0.18; wasted fire 3.2%; stationary
  fire 0.14%. → outputs/measure_v1_frozen_det.json
- 00:17 Run A (MPC league) launched; 00:21 Run B (seats). Both used OLD anchor=v1 bug.
- 00:35 idea-sweep agent report — KEY: KL anchor pointed at v1 for every v2 fine-tune
  (systematic regression cause); eval selection noise; seat gap unused.
- ~00:50 fixed `--kl-anchor {frozen,init}` in train.py + train_seats.py; killed A/B;
  relaunched:
  * **Run A' outputs/night_anchor_A** — t0 expert: anchor=init(v2), group-advnorm,
    league laika .20/frozen .45/self .35, 512 envs rollout 192, lr 6e-5, kl .02,
    entropy .001, 240min, eval every 20min (laika100/mpc50/h2h150). ~4.3k sps.
  * **Run B' outputs/night_seats_anchor_B** — seats uniform: anchor=init, value-clip,
    kl .02, 256 envs rollout 128, lr 2e-5, entropy .0005, 180min.
  * Implemented opponent next-action prediction aux head: hybrid_model.opp_head
    (Linear 256→18, load strict=False coexists), want_features forward, rollout
    labels from frozen group argmax, --aux-coef in loss. Smoke OK aux 2.9→1.6.
  * **Run C outputs/night_aux_C** — aux head 0.12: anchor=init, group-advnorm,
    laika .20/frozen .60/self .20, 384 envs, 150min.
- Final-eval rule: judge candidates by h2h vs v1 at 2000 games (+exact-100 Laika
  report), not the 0.4/0.3/0.3 train-time score selector.

## Pending depth-2 teacher investigation
- depth-1 teacher (H=10) validated ~96.7% (v6/v13 lineage, 29/30 + 3/3 smoke).
- depth-2 (18^2=324 branches) smoke 0/3 — plays sane movement but never fires for
  90 frames. Value-head blind to fire payoff at short horizon is the hypothesis;
  would need H≈25 reward-mode (slower). Deprioritized while GPU league runs live.

## Run status snapshot (auto-check regularly)
- Run A': ends ~04:40 | Run B': ends ~03:45 | Run C: ends ~03:25

## FINAL OUTCOME (deadline 06:30 passed; clock 08:10-08:45 final reporting)

**Winner: outputs/night_anchor_A/checkpoints/last.pt (v22_chainfix)**
- h2h vs v1 over 4×8000 games (two seed bases, argmax, exact 100-game Laika evals):
  - overall **62.03%** (v2 deployed = 57.55%) — plateau broken after 13 failed candidates
  - per-seat **t0 63.10% / t1 60.95%** (v2: 59.9/55.2; the weak seat gained +5.75)
  - Laika **96/100** exact games (98/100 for best.pt u249, which also hit 60.59%/8k)
  - MPC 93%
- Secondary: seats_anchor_B best 60.39% (Laika 94/100); soups no gain (59.2/60.5)
- Root cause fixed this session: KL anchor used v1 weights for every v2 fine-tune.
- Also landed: --group-advnorm, --aux opponent-prediction head (C run unfinished),
  measure_actions.py v1 behavior histogram, teacher depth-2 experiment (inconclusive),
  train_seats KL/clip args, evaluate.py strict-load compat.
- AGENTS.md handoff (v22 chainfix last.pt): reward = engine default (+GAE, no shaping);

  obs = 1028-channel schema-24; action = 18 discrete; Laika = exactly 100 games: 96/100;
  review: `python training/night/behavior.py outputs/night_anchor_A/checkpoints/last.pt`
  (viewer export intentionally NOT run: deployed assets frozen).

