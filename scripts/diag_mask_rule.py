"""What would a stricter relocation-legality rule offer?

The current rule admits any neighbour whose idle count is at most its pending
count, which lets a zone with no waiting rider become a target. This compares
candidate rules on how many targets they admit and how often a genuinely
demanded zone is reachable at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    n_zones = env._num_zones

    stats = {
        "current": dict(offered=0, free=0, best_free=0, no_target=0, turns=0),
        "pending_ge1": dict(offered=0, free=0, best_free=0, no_target=0, turns=0),
        "deficit": dict(offered=0, free=0, best_free=0, no_target=0, turns=0),
    }
    deficit_reachable = 0
    pending_reachable = 0
    total_turns = 0

    for ep in range(args.episodes):
        env.reset(seed=args.base_seed + ep)
        while env.agents:
            zone = env._zone_int_id(env.agent_selection)
            nbrs = [n for n in env.env.region_neighbours[zone] if n != zone]
            pending = env._pending_per_zone * env._num_drivers
            idle = env._zone_idle_counts
            total_turns += 1

            def record(key, legal):
                s = stats[key]
                s["turns"] += 1
                if not legal:
                    s["no_target"] += 1
                    return
                s["offered"] += len(legal)
                free = [j for j in legal if pending[j] <= 0]
                s["free"] += len(free)
                best = max(legal, key=lambda j: pending[j])
                if pending[best] <= 0:
                    s["best_free"] += 1

            # current rule, as implemented in _legal_relocate_targets
            record("current", [j for j in nbrs if idle[j] <= max(int(round(pending[j])), 1)])
            # candidate A: destination must have at least one waiting rider
            record("pending_ge1", [j for j in nbrs if pending[j] >= 1 and idle[j] <= max(int(round(pending[j])), 1)])
            # candidate B: destination must have a real deficit (idle < pending)
            record("deficit", [j for j in nbrs if idle[j] < pending[j]])

            if any(pending[j] >= 1 for j in nbrs):
                pending_reachable += 1
            if any(idle[j] < pending[j] for j in nbrs):
                deficit_reachable += 1

            m = env.action_mask(env.agent_selection)
            env.step(int(np.flatnonzero(m)[0]) if m.any() else 0)

    print("turns sampled: %d over %d episodes\n" % (total_turns, args.episodes))
    print("  %-14s %10s %12s %14s %12s" % ("rule", "targets/ep", "demand-free", "best-is-free", "no-target/ep"))
    for key, label in (("current", "current"), ("pending_ge1", "pending>=1"), ("deficit", "deficit")):
        s = stats[key]
        t = max(s["turns"], 1)
        print("  %-14s %10.1f %11.1f%% %13.1f%% %12.1f%%"
              % (label, s["offered"] / t, 100 * s["free"] / max(s["offered"], 1),
                 100 * s["best_free"] / t, 100 * s["no_target"] / t))

    print()
    print("turns where a neighbour had pending demand : %.1f%%" % (100 * pending_reachable / total_turns))
    print("turns where a neighbour had a real deficit  : %.1f%%" % (100 * deficit_reachable / total_turns))
    print()
    print("'best-is-free' is the share of turns where even the highest-demand")
    print("legal target has no waiting rider: that is the ceiling on precision.")


if __name__ == "__main__":
    main()