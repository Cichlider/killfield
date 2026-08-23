# Project rules for every agent

## Mandatory training handoff

For the current PPO route:

1. Do not add an evaluation gate or block deployment on a metric threshold.
2. Every completed training run must be exposed in the web viewer for direct behavior review.
3. Evaluate exactly 100 games against the fixed Laika opponent for the reported win rate. Do not
   silently replace this with a larger sample or a gate.
4. Every training handoff must explicitly report:
   - the model/checkpoint name;
   - the exact reward;
   - the exact observation;
   - the exact action space;
   - the win rate over exactly 100 games against Laika;
   - the browser command/URL used to review behavior.

These are reporting and review requirements, not promotion criteria. The user decides behavior
quality by watching the model.
