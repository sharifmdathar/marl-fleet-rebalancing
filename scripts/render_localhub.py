"""Render + compare the local-hub scenario: all demand at z10, then all at z12.

Before the switch every order originates at zone 10 with a destination in one
of its 8 neighbours; after it, every order originates at zone 12 with a
destination in zone 12's neighbours. Nothing else competes for service, so any
difference between policies is about redeploying cars across the map.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector
from marl_fleet_rebalancing.visualization import TrajectoryRecorder, render_animation


def episode(env, policy, seed, stochastic, frames=False):
    env.reset(seed=seed)
    rec = TrajectoryRecorder() if frames else None
    if rec is not None:
        rec.relocation_points = list(env.env.relocation_points)
    last = -1.0
    reloc = total = 0
    served1 = served2 = 0
    switch = env._cfg.localhub_switch_min
    is_rl = hasattr(policy, "act")
    while env.agents:
        agent = env.agent_selection
        obs = obs_to_vector(env.observe(agent))
        mask = env.action_mask(agent)
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
        env.step(action)
        if rec is not None and env.env.time > last:
            rec.snapshot(env.env)
            last = env.env.time
    k = env.kpi_summary()
    served, cancelled = k["served"], k["cancelled"]
    return rec, {
        "sr": 100.0 * served / max(served + cancelled, 1.0),
        "wait": k["avg_wait"], "empty": k["empty_km"],
        "reloc_pct": 100.0 * reloc / max(total, 1), "served": served,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--out-dir", default="/home/prash/college/minor/logs/videos")
    ap.add_argument("--tag", default="hub")
    ap.add_argument("--orders", type=int, default=130)
    ap.add_argument("--switch", type=float, default=25.0)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.demand_shift_mode = "localhub"
    cfg.nyc_window_limit = args.orders
    cfg.localhub_switch_min = args.switch
    z1, z2 = cfg.localhub_stage1_zone, cfg.localhub_stage2_zone

    env = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))
    greedy, lp = GreedyZoneDispatcher(), ZoneLPRebalancer()
    pols = [("greedy", greedy, True), ("lp", lp, True),
            ("rl-argmax", policy, False), ("rl-sampled", policy, True)]

    rows = {}
    for name, pol, stoch in pols:
        rows[name] = [
            episode(env, pol, args.base_seed + i, stoch)[1]
            for i in range(args.episodes)
        ]

    lp_sr = np.array([r["sr"] for r in rows["lp"]])
    print("local hub: ALL demand at z%d until min %.0f, then ALL at z%d (%d orders)"
          % (z1, args.switch, z2, args.orders))
    print()
    print("  %-12s %8s %11s %7s %10s %8s" % ("policy", "SR", "vs LP", "won", "empty km", "reloc%"))
    for name in rows:
        v = np.array([r["sr"] for r in rows[name]])
        d = v - lp_sr
        print("  %-12s %7.2f%% %+10.2fpp %5d/%-3d %10.0f %8.1f"
              % (name, v.mean(), d.mean(), int((d > 0).sum()), len(d),
                 np.mean([r["empty"] for r in rows[name]]),
                 np.mean([r["reloc_pct"] for r in rows[name]])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    for name, pol, stoch in pols:
        if not hasattr(pol, "act"):
            pol.reset(seed=args.base_seed)
        rec, kpi = episode(env, pol, args.base_seed, stoch, frames=True)
        path = out_dir / f"nyc16_{args.tag}_{name}_seed{args.base_seed}.mp4"
        render_animation(
            rec, out_path=str(path), fps=10,
            description=(f"{name} - all demand z{z1} then z{z2}, switch at "
                         f"min {args.switch:.0f} - SR {kpi['sr']:.1f}%"),
        )
        print("  ->", path)


if __name__ == "__main__":
    main()