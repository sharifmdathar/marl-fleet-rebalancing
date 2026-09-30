"""Diagnostic: where does a frozen policy send its relocation actions?

Compares argmax vs sampled action selection on the same checkpoint, seeds and
env. If argmax concentrates destinations while sampling spreads them, the
round-based action structure is the cause of the argmax/sampling SR gap.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def run(env, policy, seed, stochastic):
    env.reset(seed=seed)
    dest = Counter()
    per_turn = Counter()
    max_into_one = 0
    while env.agents:
        agent = env.agent_selection
        obs = obs_to_vector(env.observe(agent))
        mask = env.action_mask(agent)
        with torch.no_grad():
            act, _, _ = policy.act(obs, mask, deterministic=not stochastic)
        a = int(act)
        if a > env._max_pending:
            d = a - env._max_pending - 1
            dest[d] += 1
            per_turn[d] += 1
            max_into_one = max(max_into_one, per_turn[d])
        env.step(a)
        if getattr(env, "_turn_complete", False):
            per_turn.clear()
    return dest, max_into_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--env", default="nyc")
    args = ap.parse_args()

    policy, _manifest, env = load_trained_policy(Path(args.checkpoint))

    for label, stoch in (("argmax", False), ("sampled", True)):
        agg = Counter()
        worst = 0
        for ep in range(args.episodes):
            d, m = run(env, policy, 1000 + ep, stoch)
            agg += d
            worst = max(worst, m)
        total = sum(agg.values())
        top = agg.most_common(8)
        share = sum(c for _, c in top[:5]) / max(total, 1)
        print(f"\n{label}: {total} relocations over {args.episodes} eps")
        print(f"  distinct destinations: {len(agg)}")
        print(f"  top-5 destination share: {share:.1%}")
        print(f"  max relocations into one zone in a single turn: {worst}")
        print(f"  top 8: {[(z, c, f'{c/max(total,1):.1%}') for z, c in top]}")


if __name__ == "__main__":
    main()
