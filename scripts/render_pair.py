"""Render matched videos: LP rebalancer vs the trained RL policy.

Both run the SAME window with the SAME seed, so the clips are directly
comparable. Records the first episode only and writes one MP4 per policy.
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


def record_episode(env, policy, seed, stochastic):
    env.reset(seed=seed)
    rec = TrajectoryRecorder()
    last_time = -1.0
    reloc = 0
    total = 0
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
            last_time = env.env.time
    k = env.kpi_summary()
    sr = 100.0 * k["served"] / max(k["served"] + k["cancelled"], 1.0)
    return rec, {
        "service_rate": sr,
        "served": k["served"],
        "cancelled": k["cancelled"],
        "avg_wait": k["avg_wait"],
        "empty_km": k["empty_km"],
        "relocate_pct": 100.0 * reloc / max(total, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out-dir", default="/home/prash/college/minor/logs/videos")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax (RL-sampled mode)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = RideGymAEC(NYCConfig())
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))
    lp = ZoneLPRebalancer()

    lp.reset(seed=args.seed)
    rec_lp, kpi_lp = record_episode(env, lp, args.seed, stochastic=True)
    rec_rl, kpi_rl = record_episode(
        env, policy, args.seed, stochastic=args.stochastic,
    )

    mode = "sampled" if args.stochastic else "argmax"
    specs = [
        ("lp_rebalancer", rec_lp, kpi_lp, "LP rebalancer (tuned-free baseline)"),
        (f"rl_attention_{mode}", rec_rl, kpi_rl, f"RL attention ({mode})"),
    ]
    for name, rec, kpi, desc in specs:
        path = out_dir / f"nyc16_{name}_seed{args.seed}.mp4"
        render_animation(
            rec,
            out_path=str(path),
            fps=args.fps,
            description=f"{desc} - SR {kpi['service_rate']:.1f}%, "
                        f"{kpi['served']:.0f} served, {kpi['empty_km']:.0f} empty km",
        )
        print(f"\n=== {desc} ===")
        print(f"  service rate {kpi['service_rate']:.1f}%  served {kpi['served']:.0f}"
              f"  cancelled {kpi['cancelled']:.0f}")
        print(f"  avg wait {kpi['avg_wait']:.2f} min  empty {kpi['empty_km']:.0f} km"
              f"  relocate {kpi['relocate_pct']:.1f}% of turns")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
