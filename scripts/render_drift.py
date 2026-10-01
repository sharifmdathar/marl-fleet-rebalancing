"""Render the drifting-hotspot scenario: LP rebalancer vs the trained RL policy.

Scenario: demand starts concentrated in source hub A (zone 12, which has 8
neighbours), then migrates across the map to destination hub D (zone 10,
9.05 km away and 4.26 km from A's nearest neighbour) between minutes 20 and
35. Cars staged for A are useless once demand moves, so the only way to serve
the late episode is to relocate to D and back.

Both policies run the identical window and seed, so the clips are comparable
frame by frame.
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
from marl_fleet_rebalancing.visualization import TrajectoryRecorder, render_animation


def record(env, policy, seed, stochastic):
    """One episode; returns the recorder and per-phase KPIs.

    The recorder does not pick up relocation_points on its own, so the zone
    overlay (and every zone label in the video) has to be attached here.
    """
    env.reset(seed=seed)
    rec = TrajectoryRecorder()
    rec.relocation_points = list(env.env.relocation_points)
    rec.target_zone = env._cfg.drift_source_zone
    rec.target_zone_pct = env._cfg.drift_source_pct
    last_time = -1.0
    reloc = 0
    total = 0
    early = late = 0
    is_learner = hasattr(policy, "act")
    while env.agents:
        agent = env.agent_selection
        obs = obs_to_vector(env.observe(agent))
        mask = env.action_mask(agent)
        if is_learner:
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
        if env.env.time > last_time:
            rec.snapshot(env.env)
            served = sum(
                1 for o in env.env.orders.values()
                if getattr(o.status, "value", o.status) == "completed"
            )
            if env.env.time < env._cfg.horizon / 2:
                early = served
            else:
                late = served
            last_time = env.env.time
    k = env.kpi_summary()
    return rec, {
        "service_rate": 100.0 * k["served"] / max(k["served"] + k["cancelled"], 1.0),
        "served": k["served"],
        "cancelled": k["cancelled"],
        "avg_wait": k["avg_wait"],
        "empty_km": k["empty_km"],
        "relocate_pct": 100.0 * reloc / max(total, 1),
        "served_first_half": early,
        "served_second_half": late - early,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out-dir", default="/home/prash/college/minor/logs/videos")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = NYCConfig()
    cfg.demand_shift_mode = "drift"
    env = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))
    lp = ZoneLPRebalancer()

    lp.reset(seed=args.seed)
    rec_lp, kpi_lp = record(env, lp, args.seed, stochastic=True)
    rec_rl, kpi_rl = record(env, policy, args.seed, stochastic=args.stochastic)

    mode = "sampled" if args.stochastic else "argmax"
    A, D = cfg.drift_source_zone, cfg.drift_dest_zone
    specs = [
        ("lp_drift", rec_lp, kpi_lp, f"LP rebalancer on drifting hotspot (A=zone{A} -> D=zone{D})"),
        (f"rl_drift_{mode}", rec_rl, kpi_rl, f"RL attention ({mode}) on drifting hotspot (A=zone{A} -> D=zone{D})"),
    ]
    for name, rec, kpi, desc in specs:
        path = out_dir / f"nyc16_{name}_seed{args.seed}.mp4"
        render_animation(
            rec, out_path=str(path), fps=args.fps,
            description=(
                f"{desc} - SR {kpi['service_rate']:.1f}%, "
                f"{kpi['served_first_half']:.0f} served before the drift, "
                f"{kpi['served_second_half']:.0f} after"
            ),
        )
        print(f"\n=== {desc} ===")
        print(f"  overall SR {kpi['service_rate']:.1f}%  served {kpi['served']:.0f}"
              f"  cancelled {kpi['cancelled']:.0f}")
        print(f"  served before drift {kpi['served_first_half']:.0f}"
              f"  ->  after drift {kpi['served_second_half']:.0f}")
        print(f"  avg wait {kpi['avg_wait']:.2f} min  empty {kpi['empty_km']:.0f} km"
              f"  relocate {kpi['relocate_pct']:.1f}% of turns")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()