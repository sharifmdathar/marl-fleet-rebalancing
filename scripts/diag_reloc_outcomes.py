"""Do relocated cars serve the zone they were sent to, and how concentrated are destinations?

Runs a frozen policy on NYC val episodes (argmax actions) and counts, per episode:
executed relocations, relocated cars later dispatched, and dispatches in the zone
the car was sent to. Also reports destination concentration (top-1 share, HHI).

    uv run python scripts/diag_reloc_outcomes.py --mappo runs/<run>/checkpoint.pt --episodes 3
"""
from __future__ import annotations

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mappo", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=8000)
    args = ap.parse_args()

    learner, _m, _ = load_trained_policy(Path(args.mappo))
    cfg = NYCConfig()
    cfg.nyc_split = "val"
    tot = Counter()
    dests: Counter = Counter()
    for ep in range(args.episodes):
        env = RideGymAEC(cfg)
        env.reset(seed=args.base_seed + ep)
        while env.agents:
            ag = env.agent_selection
            m = env.action_mask(ag)
            with torch.no_grad():
                a, _, _ = learner.act(obs_to_vector(env.observe(ag))[None, :], m[None, :], deterministic=True)
            a = int(a[0])
            if a > env._max_pending:
                dests[a - env._max_pending - 1] += 1
            env.step(a)
        for k, v in env._reloc_out.items():
            tot[k] += v
    n = max(sum(dests.values()), 1)
    shares = np.array(list(dests.values()), dtype=float) / n
    print(f"episodes {args.episodes}")
    print(f"relocations executed : {tot['executed']}")
    print(f"later dispatched     : {tot['dispatched']} ({100*tot['dispatched']/max(tot['executed'],1):.1f}%)")
    print(f"dispatched in target : {tot['in_target']} ({100*tot['in_target']/max(tot['dispatched'],1):.1f}% of dispatched)")
    print(f"destination top-1 share : {shares.max()*100:.1f}%")
    print(f"destination HHI          : {float((shares**2).sum()):.3f}  (floor 1/{env._num_zones} = {1/env._num_zones:.3f})")


if __name__ == "__main__":
    main()
