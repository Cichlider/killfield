from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel_env import ACTIONS, DuelVec, OPPONENT_FROZEN
from hybrid_model import HybridActorCritic, load_default

FEATURE_IDS = {
    "ammo_self": 863, "ready_self": 864, "alive_self": 869,
    "hits_enemy": 890, "suicide": 891, "eta": 893,
    "aim_opp_hits": 895, "ready_opp": 876, "idle": 1027,
}


def load(path: Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = HybridActorCritic(gated_dodge=True, gated_ammo=True)
    model.load_state_dict(payload.get("model", payload), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def run(games, envs, seed, device, model, opp_model, deterministic):
    env = DuelVec(envs, seed, OPPONENT_FROZEN, min(16, envs))
    hist = np.zeros(ACTIONS, np.int64)
    feature_sum = {k: 0.0 for k in FEATURE_IDS}
    dodge_mean = np.zeros(9, np.float64)
    move_fire = 0
    stationary_fire = 0
    fire_frames = 0
    wasted_fire = 0
    fire_when_enemy_aimed = 0
    fire_when_safekill = 0
    dodge_frames = 0
    feature_frames = 0
    rows = []
    episode_fire = np.zeros(envs, np.int64)
    episode_move_fire = np.zeros(envs, np.int64)
    while len(rows) < games:
        a_obs = torch.as_tensor(env.obs.copy(), dtype=torch.float32, device=device)
        a_mask = torch.as_tensor(env.masks.astype(bool, copy=True), device=device)
        o_obs = torch.as_tensor(env.obs_opponent.copy(), dtype=torch.float32, device=device)
        o_mask = torch.as_tensor(env.masks_opponent.astype(bool, copy=True), device=device)
        logits, _ = model(a_obs, a_mask)
        opp_logits, _ = opp_model(o_obs, o_mask)
        if deterministic:
            a = logits.argmax(-1).cpu().numpy().astype(int)
            o = opp_logits.argmax(-1).cpu().numpy().astype(int)
        else:
            a = Categorical(logits=logits).sample().cpu().numpy().astype(int)
            o = Categorical(logits=opp_logits).sample().cpu().numpy().astype(int)
        pre = env.obs.copy()
        hist += np.bincount(a, minlength=ACTIONS)
        fire_bit = a % 2 == 1
        move_id = a // 2  # 0 back, 1 neutral, 2 fwd
        fire_frames += int(fire_bit.sum())
        move_fire += int((fire_bit & (move_id != 1)).sum())
        stationary_fire += int((fire_bit & (move_id == 1)).sum())
        wasted_fire += int((fire_bit & (pre[:, 864] < 0.5)).sum())
        fire_when_enemy_aimed += int((fire_bit & (pre[:, 895] > 0.5)).sum())
        # safe-kill: own shot hits enemy now (obs 890) but not self (891)
        fire_when_safekill += int((fire_bit & (pre[:, 890] > 0.5) & (pre[:, 891] < 0.5)).sum())
        dodge_mean += pre[:, 1018:1027].sum(axis=0)
        dodge_frames += len(pre)
        for k, fid in FEATURE_IDS.items():
            feature_sum[k] += float(pre[:, fid].sum())
        feature_frames += len(pre)
        episode_fire += fire_bit.astype(np.int64)
        episode_move_fire += (fire_bit & (move_id != 1)).astype(np.int64)
        env.step(np.asarray(a, np.uint16), np.asarray(o, np.uint16))
        done = env.dones.astype(bool)
        for i in range(envs):
            if done[i] and env.terminals[i]:
                rows.append({
                    "outcome": int(env.outcomes[i]),
                    "frames": int(env.episode_frames[i]),
                    "fire": int(episode_fire[i]),
                    "move_fire": int(episode_move_fire[i]),
                })
        episode_fire[done] = 0
        episode_move_fire[done] = 0
        env.reset_done()
    env.close()
    out = np.asarray([r["outcome"] for r in rows], int)
    total = int(hist.sum())
    dist = (hist / max(total, 1)).round(4).tolist()
    per_ep_fire = np.asarray([r["fire"] for r in rows], float)
    return {
        "games": len(rows),
        "wins": int((out == 1).sum()),
        "losses": int((out == 2).sum()),
        "doubles": int((out == 3).sum()),
        "win_rate": float((out == 1).mean()),
        "frames_mean": float(np.mean([r["frames"] for r in rows])),
        "fire_per_episode_mean": float(per_ep_fire.mean()),
        "action_histogram": dist,
        "move_fire_rate": float(move_fire / max(feature_frames, 1)),
        "stationary_fire_rate": float(stationary_fire / max(feature_frames, 1)),
        "fire_rate": float(fire_frames / max(feature_frames, 1)),
        "wasted_fire_rate": float(wasted_fire / max(feature_frames, 1)),
        "fire_while_enemy_aimed": float(fire_when_enemy_aimed / max(feature_frames, 1)),
        "fire_while_safekill": float(fire_when_safekill / max(feature_frames, 1)),
        "dodge_mean": (dodge_mean / max(dodge_frames, 1)).round(3).tolist(),
        "mean_features": {k: round(v / max(feature_frames, 1), 4) for k, v in feature_sum.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--opponent-model", type=Path, default=None)
    parser.add_argument("--opponent-default-asset", action="store_true",
                        help="opponent is the frozen v1 asset hybrid.bin")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--envs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--sample", action="store_true", help="sample both policies")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load(args.model, device)
    if args.opponent_default_asset or args.opponent_model is None:
        opp = load_default(device).eval()
    else:
        opp = load(args.opponent_model, device)
    result = run(
        args.games, args.envs, args.seed, device, model, opp,
        deterministic=not args.sample,
    )
    result["model"] = str(args.model)
    output = args.output or args.model.with_suffix(".measure.json")
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
