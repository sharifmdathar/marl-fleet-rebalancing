"""Relocation precision and distance efficiency for a frozen policy.

Answers two questions that service rate alone cannot:

1. Precision: when the policy relocates a car from zone i to zone j, was j the
   best target available at that moment? Chance is ~1/8 given 8 neighbours.
2. Efficiency: how much service rate is bought per empty km, compared to the
   baselines? Service rate alone is purchasable with deadhead distance, so a
   policy that wins SR while burning 4x the distance may not have won anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def episode(env, policy, seed, stochastic):
    """One episode, recording the pending demand at each legal relocate target."""
    env.reset(seed=seed)
    reloc = total = 0
    best_chosen = n_considered = 0
    empty_at_arrival = 0
    is_rl = hasattr(policy, "act")
    while env.agents:
        agent = env.agent_selection
        zone = env._zone_int_id(agent)
        obs = obs_to_vector(env.observe(agent))
        mask = env.action_mask(agent)
        # Snapshot of unmet demand per zone at this instant, used only for
        # scoring the choice after the fact. Never fed to the policy.
        pending = (env._pending_per_zone * env._num_drivers).copy()

        if is_rl:
            with torch.no_grad():
                act, _, _ = policy.act(
                    obs[None, :], mask[None, :], deterministic=not stochastic,
                )
            action = int(act[0])
        else:
            action = int(policy.select_action(agent, env, obs))
        if not mask[action]:
            action = int(np.flatnonzero(mask)[0])

        total += 1
        if action > env._max_pending:
            reloc += 1
            target = action - env._max_pending - 1
            legal = [
                j for j in env._legal_relocate_targets(zone)
            ]
            if legal:
                n_considered += 1
                best = max(legal, key=lambda j: (pending[j], -j))
                if target == best:
                    best_chosen += 1
            if pending[target] <= 0:
                empty_at_arrival += 1
        env.step(action)

    k = env.kpi_summary()
    served, cancelled = k["served"], k["cancelled"]
    return {
        "sr": 100.0 * served / max(served + cancelled, 1.0),
        "empty": k["empty_km"],
        "reloc": reloc,
        "total": total,
        "precision": best_chosen / max(n_considered, 1),
        "considered": n_considered,
        "wasted_share": empty_at_arrival / max(reloc, 1),
        "served": served,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--orders", type=int, default=130)
    ap.add_argument("--switch", type=float, default=25.0)
    ap.add_argument("--mode", default="base", choices=["base", "localhub"])
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_window_limit = args.orders
    if args.mode == "localhub":
        cfg.demand_shift_mode = "localhub"
        cfg.localhub_switch_min = args.switch

    env = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))
    lp = ZoneLPRebalancer()

    out = {}
    for name, pol, stoch in (
        ("lp", lp, True), ("rl-argmax", policy, False), ("rl-sampled", policy, True),
    ):
        rows = []
        for i in range(args.episodes):
            if not hasattr(pol, "act"):
                pol.reset(seed=args.base_seed + i)
            rows.append(episode(env, pol, args.base_seed + i, stoch))
        out[name] = rows

    print("mode=%s, %d episodes, %d orders/window\n" % (args.mode, args.episodes, args.orders))
    print("  %-12s %7s %10s %10s %10s %12s" % (
        "policy", "SR", "empty km", "relocs/ep", "precision", "SR per 100km"))
    for name in out:
        rs = out[name]
        sr = np.mean([r["sr"] for r in rs])
        km = np.mean([r["empty"] for r in rs])
        rel = np.mean([r["reloc"] for r in rs])
        prec = np.mean([r["precision"] for r in rs if r["considered"] > 0]) \
            if any(r["considered"] for r in rs) else float("nan")
        rate = 100.0 * sr / km if km > 0 else float("nan")
        print("  %-12s %6.1f%% %10.0f %10.0f %9.0f%% %12.2f"
              % (name, sr, km, rel, 100 * prec, rate))

    print("\nwasted share of relocations (destination had no unmet demand):")
    for name in out:
        rs = [r for r in out[name] if r["reloc"] > 0]
        if rs:
            print("  %-12s %.0f%%  (%d of %d relocations scored)"
                  % (name, 100 * np.mean([r["wasted_share"] for r in rs]),
                     sum(r["considered"] for r in rs),
                     sum(r["reloc"] for r in rs)))
        else:
            print("  %-12s no relocations" % name)

    a = np.mean([r["precision"] for r in out["rl-argmax"] if r["considered"] > 0] or [float("nan")])
    s = np.mean([r["precision"] for r in out["rl-sampled"] if r["considered"] > 0] or [float("nan")])
    print("\nprecision reference: 1/8 = 12.5%% is chance over 8 neighbours")
    print("  rl-argmax %.1f%%   rl-sampled %.1f%%" % (100 * a, 100 * s))


if __name__ == "__main__":
    main()