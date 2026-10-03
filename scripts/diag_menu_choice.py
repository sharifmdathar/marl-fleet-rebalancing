"""How much choice does the agent actually have, and is any of it learnable?

Measured finding that motivates this: with require_destination_demand=True the
mask removes destinations with no waiting rider BEFORE the agent sees them.
99.9% of legal relocate edges pointed at a zone that already had a rider, the
menu averaged 2.20 options, and 68.5% of turns offered options with IDENTICAL
pending demand. So relocate-first-rand matched relocate-first-best: the mask had
already made the options equivalent.

This measures, per candidate configuration, whether the action menu leaves
anything to discriminate:

  menu size        mean legal targets per turn
  spread           mean (max-min) pending demand ACROSS legal targets
  null turns       % of turns where all legal targets are equivalent
                   (spread == 0 -> "best" and "random" are the same action)
  info gap         mean best-minus-worst pending across legal targets

A configuration with high spread and low null turns has a decision the agent
could in principle learn. A configuration with spread ~0 does not.

    uv run python scripts/diag_menu_choice.py --episodes 2
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC

CONFIGS = {
    # current study config
    "current": {},
    # route 1: let demand-free zones back into the menu
    "allow_demand_free": {"require_destination_demand": False},
    # route 2: citywide relocation instead of neighbour-only
    "citywide": {"neighbor_relocation_only": False},
    # both
    "citywide_demand_free": {
        "neighbor_relocation_only": False,
        "require_destination_demand": False,
    },
    # no claim rule, so more destinations stay open
    "no_claim": {"claim_relocation_targets": False},
}


def measure(env, episodes, base_seed):
    st = Counter()
    sizes: list[int] = []
    spreads: list[int] = []
    for ep in range(episodes):
        env.reset(seed=base_seed + ep)
        while env.agents:
            z = env._zone_int_id(env.agent_selection)
            legal = env._legal_relocate_targets(z)
            if not legal:
                st["no_targets"] += 1
            else:
                st["has_targets"] += 1
                p = [int(round(float(env._pending_per_zone[j] * env._num_drivers)))
                     for j in legal]
                sizes.append(len(legal))
                spreads.append(max(p) - min(p))
                st["edges"] += len(legal)
                # Per-EDGE, not per-turn. An earlier version incremented this
                # once per turn while dividing by the edge total, mixing two
                # denominators and reporting a meaningless percentage.
                st["edges_with_rider"] += int(max(p) >= 1) * len(legal)
            # Keep the run alive with a legal non-degenerate action.
            m = env.action_mask(env.agent_selection)
            d = np.flatnonzero(m[1 : 1 + env._max_pending])
            r = np.flatnonzero(m[1 + env._max_pending:])
            if len(d):
                act = int(d[0]) + 1
            elif len(r):
                act = int(r[0]) + 1 + env._max_pending
            else:
                act = 0
            env.step(act)
    turns = max(st["has_targets"] + st["no_targets"], 1)
    return {
        "turns": turns,
        "menu": float(np.mean(sizes)) if sizes else 0.0,
        "spread": float(np.mean(spreads)) if spreads else 0.0,
        "null_turns_pct": 100.0 * float(np.mean([s == 0 for s in spreads]))
                          if spreads else 100.0,
        "no_target_pct": 100.0 * st["no_targets"] / turns,
        "edge_rider_pct": 100.0 * st["edges_with_rider"] / max(st["edges"], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    print("order_timeout now defaults to 5.0 min (env 3.0)\n")
    print("%-24s %7s %9s %11s %12s %12s"
          % ("config", "menu", "spread", "null turns", "no-target", "edge+rider"))
    print("-" * 82)
    for name, overrides in CONFIGS.items():
        cfg = NYCConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        m = measure(RideGymAEC(cfg), args.episodes, args.base_seed)
        print("%-24s %7.2f %9.2f %10.1f%% %11.1f%% %11.1f%%"
              % (name, m["menu"], m["spread"], m["null_turns_pct"],
                 m["no_target_pct"], m["edge_rider_pct"]))

    print("\nnull turns  = all legal targets have identical pending demand,")
    print("              so 'pick the best' and 'pick at random' are the")
    print("              same action. High values mean targeting is unlearnable.")


if __name__ == "__main__":
    main()