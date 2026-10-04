"""Does the policy aim at the zone that needs a car? Measured at DECISION time.

Why this script exists. Every earlier targeting number compared the zone a policy
AIMED at with the zone its car ENDED UP in. That comparison is invalid:

    aimed zone 10 -> ... dispatched en route to a rider in zone 8

`allow_relocating_dispatch` (D14) hands a relocating car to a rider before it
arrives, and 7 of 7 traced legs were cut short that way. So the landing zone
reflects where a rider happened to be, not where the policy aimed. Precision
figures built on it (including the 4.1% chance baseline in D44) are confounded.

Here the comparison is made at the instant the action is chosen, from the
demand distribution the agent could see at that moment:

    hits   : chose the legal target with the most unmet demand
    chance : 1 / (number of legal targets available right now)

Also reports two harder bars, because "best legal target" is a weak standard:

    global_best  : chose the zone with the most unmet demand citywide, whether
                   or not it was a legal target
    expressible  : of the moves that missed global_best, how many aimed at a
                   zone that was not offered at all

That last split is the one that matters. A miss is only a policy failure if the
right zone was on the menu.

    uv run python scripts/diag_targeting_at_decision.py --episodes 3 \
        --ippo runs/d51-ippo-s0/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def pending_counts(env) -> np.ndarray:
    """Unmet demand per zone, in orders."""
    scale = max(env._num_drivers, 1)
    return np.asarray(env._pending_per_zone, dtype=float) * scale


def run_arm(policy, cfg, args, label: str) -> dict:
    env = RideGymAEC(cfg)
    mp = env._max_pending

    n_moves = 0
    hits_best_legal = 0
    hits_global = 0
    chance_sum = 0.0
    n_targets_sum = 0
    miss_because_unoffered = 0
    miss_because_other_legal = 0
    n_targets_hist: dict[int, int] = {}
    demand_at_pick: list[float] = []
    demand_best_legal: list[float] = []
    target_was_global_best_but_unoffered = 0

    for ep in range(args.episodes):
        if hasattr(policy, "reset") and not hasattr(policy, "act"):
            policy.reset(seed=args.base_seed + ep)
        env.reset(seed=args.base_seed + ep)

        while env.agents:
            for _ in range(env._num_zones):
                if not env.agents:
                    break
                ag = env.agent_selection
                zi = env._zone_int_id(ag)
                m = env.action_mask(ag)

                pend = pending_counts(env)
                legal = env._legal_relocate_targets(zi)

                obs = obs_to_vector(env.observe(ag))
                if hasattr(policy, "act"):
                    with torch.no_grad():
                        act, _, _ = policy.act(obs[None, :], m[None, :],
                                               deterministic=False)
                    a_val = int(act[0])
                else:
                    a_val = int(policy.select_action(ag, env, env.observe(ag)))
                if not m[a_val]:
                    a_val = int(np.flatnonzero(m)[0])

                decoded = env._decode_action(a_val, zi)
                if isinstance(decoded, dict) and "relocate" in decoded and legal:
                    pick = int(decoded["relocate"])
                    n_moves += 1

                    n_legal = len(legal)
                    n_targets_hist[n_legal] = n_targets_hist.get(n_legal, 0) + 1
                    n_targets_sum += n_legal
                    chance_sum += 1.0 / max(n_legal, 1)

                    lg = np.asarray(legal, dtype=int)
                    best_legal = int(lg[int(np.argmax(pend[lg]))])
                    demand_at_pick.append(float(pend[pick]))
                    demand_best_legal.append(float(pend[best_legal]))

                    if pick == best_legal:
                        hits_best_legal += 1

                    gbest = int(np.argmax(pend))
                    if pick == gbest:
                        hits_global += 1
                    elif gbest not in legal:
                        # The genuinely neediest zone was not on the menu at all.
                        miss_because_unoffered += 1
                        target_was_global_best_but_unoffered += 1
                    else:
                        miss_because_other_legal += 1

                env.step(a_val)

    return {
        "label": label,
        "moves": n_moves,
        "best_legal": 100.0 * hits_best_legal / max(n_moves, 1),
        "chance": 100.0 * chance_sum / max(n_moves, 1),
        "global": 100.0 * hits_global / max(n_moves, 1),
        "miss_unoffered": miss_because_unoffered,
        "miss_other_legal": miss_because_other_legal,
        "mean_targets": n_targets_sum / max(n_moves, 1),
        "hist": n_targets_hist,
        "d_pick": float(np.mean(demand_at_pick)) if demand_at_pick else 0.0,
        "d_best": float(np.mean(demand_best_legal)) if demand_best_legal else 0.0,
    }


def report(r: dict) -> None:
    m = max(r["moves"], 1)
    print(f"\n--- {r['label']}")
    print(f"  relocation moves                : {r['moves']}")
    print(f"  mean legal targets per move     : {r['mean_targets']:.2f}")
    print(f"  chance (1/legal targets)        : {r['chance']:.2f}%")
    print(f"  chose BEST LEGAL target         : {r['best_legal']:.2f}%")
    print(f"  chose GLOBAL best target        : {r['global']:.2f}%")
    print()
    print(f"  misses where the neediest zone was NOT offered : "
          f"{r['miss_unoffered']} ({100*r['miss_unoffered']/m:.1f}%)")
    print(f"  misses where it WAS offered but not chosen     : "
          f"{r['miss_other_legal']} ({100*r['miss_other_legal']/m:.1f}%)")
    print()
    print(f"  mean unmet demand at chosen target : {r['d_pick']:.2f}")
    print(f"  mean unmet demand at best target   : {r['d_best']:.2f}")
    ratio = r["d_pick"] / r["d_best"] if r["d_best"] > 0 else float("nan")
    print(f"  ratio                              : {ratio:.2f}"
          f"   (1.0 = always as needy as the best available)")
    print(f"  legal-target count distribution   : "
          f"{dict(sorted(r['hist'].items()))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"

    arms: list[tuple[str, object]] = []
    if args.ippo:
        learner, _m, _ = load_trained_policy(Path(args.ippo))
        arms.append(("IPPO", learner))
    if args.mappo:
        learner, _m, _ = load_trained_policy(Path(args.mappo))
        arms.append(("MAPPO", learner))
    if not arms:
        print("pass --ippo and/or --mappo")
        return

    print("=" * 72)
    print("Targeting measured at the moment of the decision")
    print("=" * 72)
    for label, policy in arms:
        report(run_arm(policy, cfg, args, label))

    print()
    print("Reading: 'best legal' vs 'chance' is the fair test -- it asks whether")
    print("the policy picks the neediest zone ON THE MENU. 'global best' is the")
    print("harder standard and is bounded by the mask, so the split of misses")
    print("says how much of the failure is policy and how much is expressibility.")


if __name__ == "__main__":
    main()
