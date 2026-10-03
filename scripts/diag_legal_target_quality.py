"""How many legal relocation targets actually have unmet demand?

If a large share of legal targets are demand-free, then the 87% wasted-move
rate is partly the action mask permitting pointless moves rather than the
policy choosing badly. That distinction decides whether the fix belongs in the
mask or in the policy.
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
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    policy = None
    if args.checkpoint:
        policy, _m, _ = load_trained_policy(Path(args.checkpoint))

    for label in (["rl"] if policy else ["noop"]):
        n_legal = 0
        n_demand_free = 0
        best_is_free = 0
        chosen = Counter()
        chosen_free = 0
        relocs = 0
        for i in range(args.episodes):
            env.reset(seed=args.base_seed + i)
            while env.agents:
                agent = env.agent_selection
                zone = env._zone_int_id(agent)
                obs = obs_to_vector(env.observe(agent))
                mask = env.action_mask(agent)
                pending = env._pending_per_zone * env._num_drivers
                legal = env._legal_relocate_targets(zone)
                if legal:
                    n_legal += len(legal)
                    free = [j for j in legal if pending[j] <= 0]
                    n_demand_free += len(free)
                    best = max(legal, key=lambda j: pending[j])
                    if pending[best] <= 0:
                        best_is_free += 1
                if policy is not None:
                    with torch.no_grad():
                        act, _, _ = policy.act(
                            obs[None, :], mask[None, :], deterministic=False,
                        )
                    a = int(act[0])
                else:
                    a = int(np.flatnonzero(mask)[0])
                if not mask[a]:
                    a = int(np.flatnonzero(mask)[0])
                if a > env._max_pending:
                    relocs += 1
                    t = a - env._max_pending - 1
                    chosen[t] += 1
                    if pending[t] <= 0:
                        chosen_free += 1
                env.step(a)

        print("policy: %s over %d episodes" % (label, args.episodes))
        print("  legal relocation targets offered : %d" % n_legal)
        print("  of those, demand-free (pending<=0): %d  (%.1f%%)"
              % (n_demand_free, 100 * n_demand_free / max(n_legal, 1)))
        print("  turns where even the BEST legal target is demand-free: %d" % best_is_free)
        print("  relocations taken                 : %d" % relocs)
        if relocs:
            print("  of those, to a demand-free zone   : %d  (%.1f%%)"
                  % (chosen_free, 100 * chosen_free / relocs))
            top = ", ".join("z%d:%d" % (z, c) for z, c in chosen.most_common(5))
            print("  most-chosen destinations          : %s" % top)
        print()
        print("  => if 'legal targets offered' are mostly demand-free, the mask")
        print("     is offering pointless moves and the fix belongs there.")


if __name__ == "__main__":
    main()