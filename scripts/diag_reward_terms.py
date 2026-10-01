"""Full reward decomposition per episode.

The policy relocates ~220 times an episode and that looks heavily net-negative
under the configured weights. If it still does it, either the empty-km cost is
not actually being charged, or something else is paying. This totals each term.
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


def decompose(env, policy, seed, stochastic=False):
    cfg = env._cfg
    from ride_gym.enums import OrderStatus

    env.reset(seed=seed)
    totals = dict(served=0.0, empty_km_cost=0.0, empty_km_raw=0.0,
                  wait=0.0, cancel=0.0, n_reloc=0, n_dispatch=0,
                  n_completion=0, n_credit_fires=0)

    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        # Consume the completion delta ONCE, score it, then restore the state so
        # the real call sees the same delta. Restoring afterwards would make the
        # real call observe nothing new and silently zero the served credit.
        seen = set(env._seen_completed_ids)
        reloc_before = dict(env._driver_relocator)

        for did, ev in events.items():
            if ev.get("is_empty_move", False):
                totals["empty_km_raw"] += float(ev.get("distance_moved", 0.0)) / 1000.0

        for oid in env._newly_completed_ids():
            o = env.env.orders[oid]
            did = getattr(o, "assigned_driver", None)
            disp = env._order_dispatcher.get(oid)
            totals["n_completion"] += 1
            if did is not None and did in reloc_before and disp is not None \
                    and reloc_before[did] != disp:
                totals["n_credit_fires"] += 1
            if disp is not None:
                totals["served"] += cfg.gamma

        env._seen_completed_ids = seen
        return orig(events, dispatched_this_turn)

    env._compute_zone_rewards = traced

    while env.agents:
        agent = env.agent_selection
        obs = obs_to_vector(env.observe(agent))
        mask = env.action_mask(agent)
        with torch.no_grad():
            act, _, _ = policy.act(
                obs[None, :], mask[None, :], deterministic=stochastic,
            )
        a = int(act[0])
        if not mask[a]:
            a = int(np.flatnonzero(mask)[0])
        if a > env._max_pending:
            totals["n_reloc"] += 1
        elif a > 0:
            totals["n_dispatch"] += 1
        env.step(a)

    totals["empty_km_cost"] = cfg.beta * totals["empty_km_raw"]
    env._compute_zone_rewards = orig
    return totals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))

    agg = {}
    for i in range(args.episodes):
        t = decompose(env, policy, args.base_seed + i)
        for k, v in t.items():
            agg[k] = agg.get(k, 0.0) + v
        print("ep %d: served-credit=%.1f  empty=%.0fkm cost=%.1f  reloc=%d  "
              "credit-fires=%d" % (args.base_seed + i, t["served"], t["empty_km_raw"],
                                   t["empty_km_cost"], t["n_reloc"], t["n_credit_fires"]))
    n = args.episodes
    print("\nper-episode averages:")
    print("  completions served      : %.1f  -> reward +%.1f" % (agg["n_completion"] / n, agg["served"] / n))
    print("  empty km travelled      : %.1f  -> cost    -%.1f" % (agg["empty_km_raw"] / n, agg["empty_km_cost"] / n))
    print("  credit fires (relocated): %.1f  -> share   +%.1f" % (agg["n_credit_fires"] / n, cfg.relocator_share * agg["n_credit_fires"] * cfg.gamma / n))
    print("  relocation actions      : %.1f" % (agg["n_reloc"] / n))
    km_per_reloc = agg["empty_km_raw"] / max(agg["n_reloc"], 1)
    print("\n  empty km per relocation : %.2f" % km_per_reloc)
    print("  cost per relocation     : %.3f" % (cfg.beta * km_per_reloc))
    print("  credit per relocation   : %.3f" % (cfg.relocator_share * agg["n_credit_fires"] * cfg.gamma / max(agg["n_reloc"], 1)))
    print("  NET per relocation      : %+.3f"
          % (cfg.relocator_share * agg["n_credit_fires"] * cfg.gamma / max(agg["n_reloc"], 1)
             - cfg.beta * km_per_reloc))


if __name__ == "__main__":
    main()