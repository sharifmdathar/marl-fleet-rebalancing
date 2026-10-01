"""Render + compare the sequential two-stage surge (zone 10 -> zone 12).

Runs every policy on identical windows (paired, per docs/eval-noise.md) and
writes a matched video each. Zones are outlined and the active surge hub is
ringed, so it is visible whether cars actually move between the two stages.
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
    s1 = s2 = 0
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
    ap.add_argument("--tag", default="surge")
    ap.add_argument("--orders", type=int, default=130)
    ap.add_argument("--pct", type=float, default=60.0)
    ap.add_argument("--stage1-end", type=float, default=18.0)
    ap.add_argument("--stage2-start", type=float, default=32.0)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.demand_shift_mode = "surge2"
    cfg.nyc_window_limit = args.orders
    cfg.surge_pct = args.pct
    cfg.surge_stage1_end_min = args.stage1_end
    cfg.surge_stage2_start_min = args.stage2_start
    z1, z2 = cfg.surge_zone1, cfg.surge_zone2

    env = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))
    greedy, lp = GreedyZoneDispatcher(), ZoneLPRebalancer()
    pols = [("greedy", greedy, True, False), ("lp", lp, True, False),
            ("rl-argmax", policy, False, True), ("rl-sampled", policy, True, True)]

    rows = {}
    for name, pol, stoch, _rl in pols:
        out = []
        for i in range(args.episodes):
            if not hasattr(pol, "act"):
                pol.reset(seed=args.base_seed + i)
            _r, kpi = episode(env, pol, args.base_seed + i, stoch)
            out.append(kpi)
        rows[name] = out

    lp_sr = np.array([r["sr"] for r in rows["lp"]])
    print("sequential surge z%d -> z%d, %.0f%% of %d orders, handover %.0f-%.0f min\n"
          % (z1, z2, args.pct, args.orders, args.stage1_end, args.stage2_start))
    print("  %-12s %9s %10s %8s %10s %8s" % ("policy", "SR", "vs LP", "won", "empty km", "reloc%"))
    for name in rows:
        v = np.array([r["sr"] for r in rows[name]])
        d = v - lp_sr
        print("  %-12s %8.2f%% %+9.2fpp %5d/%d %10.0f %8.1f"
              % (name, v.mean(), d.mean(), int((d > 0).sum()), len(d),
                 np.mean([r["empty"] for r in rows[name]]),
                 np.mean([r["reloc_pct"] for r in rows[name]])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    for name, pol, stoch, _rl in pols:
        if not hasattr(pol, "act"):
            pol.reset(seed=args.base_seed)
        rec, kpi = episode(env, pol, args.base_seed, stoch, frames=True)
        path = out_dir / f"nyc16_{args.tag}_{name}_seed{args.base_seed}.mp4"
        render_animation(
            rec, out_path=str(path), fps=10,
            description=(f"{name} - sequential surge z{z1} -> z{z2}, "
                         f"{args.pct:.0f}% of {args.orders} orders - SR {kpi['sr']:.1f}%"),
        )
        print("  ->", path)


if __name__ == "__main__":
    main()