import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel_env import ACTIONS, DuelVec, OPPONENT_FROZEN
from teacher_search import choose_search, load, logits

CAND = ["back-L", "back-Lf", "back-0", "back-0f", "back-R", "back-Rf",
        "0-L", "0-Lf", "STOP", "STOPf", "0-R", "0-Rf",
        "fwd-L", "fwd-Lf", "fwd-0", "fwd-0f", "fwd-R", "fwd-Rf"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = load(Path(sys.argv[1]) if len(sys.argv) > 1 else
              Path("outputs/night_run1/checkpoints/hybrid_night_final.pt"), device)
opponent = load(Path("outputs/night_run1/checkpoints/initial.pt"), device)

env = DuelVec(1, 20261101, OPPONENT_FROZEN, 1)
HERO = []
for frame in range(8):
    search = env.make_search(0, 0, ACTIONS ** 2, 8)
    action, opp_a, best, src_v, scores = choose_search(
        env, search, policy, opponent, device, 12, 0.99, 0, 2, False,
    )
    root, second = divmod(action, ACTIONS)
    top = np.argsort(-scores)[:5]
    interesting = [f"{CAND[i // ACTIONS]}/{CAND[i % ACTIONS]}={scores[i]:.4f}" for i in top]
    print(f"frame {frame}: choose {CAND[root]}->{CAND[second]} best={best:.4f} "
          f"src_v={src_v:.4f} top: " + " | ".join(interesting), flush=True)
    HERO.append((root, second))
    search.close()
    obs = env.obs.copy()
    print(f"   learner_hits={obs[0][890]:.2f} suicide={obs[0][891]:.2f} "
          f"ready={obs[0][864]:.2f} bullets={int(env.masks.sum())} "
          f"opp_hits={obs[0][895]:.2f}", flush=True)
    env.step(np.array([action], np.uint16), np.array([opp_a], np.uint16))
    done = bool(env.terminals[0])
    if not done:
        env.reset_done()
    if done:
        print("terminal", int(env.outcomes[0]), flush=True)
        break
env.close()
