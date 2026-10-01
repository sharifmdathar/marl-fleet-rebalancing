"""Interactive scenario picker: choose demand and hubs, render both policies.

Prints the zone table, then prompts for the demand cap and the source/dest
hub zones, validates the choice against the actual zone geometry, and renders
LP vs the trained RL policy on that scenario.

    uv run python scripts/render_scenario.py --checkpoint <ckpt>
"""
from __future__ import annotations

import argparse
import json
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


def km(cents, a, b):
    lon = (cents[a][0] - cents[b][0]) * 111.0 * np.cos(np.radians(cents[a][1]))
    lat = (cents[a][1] - cents[b][1]) * 111.0
    return float(np.hypot(lon, lat))


def prompt_int(text, lo, hi, default):
    while True:
        raw = input(f"{text} [{lo}-{hi}, default {default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  enter a whole number")
            continue
        if lo <= v <= hi:
            return v
        print(f"  must be between {lo} and {hi}")


def prompt_float(text, lo, hi, default):
    while True:
        raw = input(f"{text} [{lo}-{hi}, default {default}]: ").strip()
        if not raw:
            return default
        try:
            v = float(raw)
        except ValueError:
            print("  enter a number")
            continue
        if lo <= v <= hi:
            return v
        print(f"  must be between {lo} and {hi}")


def record(env, policy, seed, stochastic):
    env.reset(seed=seed)
    rec = TrajectoryRecorder()
    rec.relocation_points = list(env.env.relocation_points)
    last_time = -1.0
    reloc = total = 0
    served_first_half = served_at_half = 0
    half = env._cfg.horizon / 2
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
            if last_time < 0:
                served_first_half = 0
            if abs(env.env.time - half) < env._cfg.dt:
                served_at_half = sum(
                    1 for o in env.env.orders.values()
                    if str(o.status).endswith("COMPLETED")
                )
            last_time = env.env.time
    k = env.kpi_summary()
    return rec, {
        "service_rate": 100.0 * k["served"] / max(k["served"] + k["cancelled"], 1.0),
        "served": k["served"], "cancelled": k["cancelled"],
        "avg_wait": k["avg_wait"], "empty_km": k["empty_km"],
        "relocate_pct": 100.0 * reloc / max(total, 1),
        "before": served_at_half, "after": k["served"] - served_at_half,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out-dir", default="/home/prash/college/minor/logs/videos")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--tag", default="custom")
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    cents = np.asarray(env.env.relocation_points, float)
    ids = json.loads(Path(cfg.nyc_splits_dir, "manifest.json").read_text())["zone_ids"]
    nbr = {z: sorted(env.env.region_neighbours[z]) for z in range(env._num_zones)}

    print("zones (env numbering; A/D must be given in this numbering)\n")
    print("  zone  taxi-id    lon       lat    neighbours")
    for z in range(env._num_zones):
        print("  %4d  %7d  %8.4f %8.4f    %s"
              % (z, ids[z], cents[z][0], cents[z][1], nbr[z]))

    print("\n--- scenario ---")
    cfg.demand_shift_mode = "drift"
    cfg.nyc_window_limit = prompt_int(
        "orders per window", 40, 400, cfg.nyc_window_limit,
    )
    src = prompt_int("source hub A (early demand)", 0, env._num_zones - 1, 12)
    dst = prompt_int("dest hub D (late demand)", 0, env._num_zones - 1, 10)
    if src == dst:
        print("  A and D must differ; using D=10")
        dst = 10 if src != 10 else 11
    cfg.drift_source_zone = src
    cfg.drift_dest_zone = dst
    cfg.drift_source_pct = prompt_float(
        "share of orders originating at A before the ramp", 0.1, 1.0, 0.6,
    )
    cfg.drift_ramp_start_min = prompt_float("drift starts (min)", 1, 55, 20.0)
    cfg.drift_ramp_end_min = prompt_float("drift ends (min)", 2, 60, 35.0)
    if cfg.drift_ramp_end_min <= cfg.drift_ramp_start_min:
        print("  ramp end must exceed start; using 35.0")
        cfg.drift_ramp_end_min = min(35.0, cfg._cfg.horizon)

    near = min((km(cents, dst, n) for n in nbr[src] if n != dst), default=0.0)
    print("\n  A=%d D=%d  A-D %.2f km, D to A's nearest neighbour %.2f km"
          % (src, dst, km(cents, src, dst), near))
    if near < 1.0:
        print("  WARNING: D is adjacent to A, so this is not a cross-map drift")

    env2 = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))
    lp = ZoneLPRebalancer()
    lp.reset(seed=args.seed)
    rec_lp, kpi_lp = record(env2, lp, args.seed, stochastic=True)
    rec_rl, kpi_rl = record(env2, policy, args.seed, stochastic=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    desc = (f"A=zone{src} -> D=zone{dst}, {cfg.nyc_window_limit} orders, "
            f"ramp {cfg.drift_ramp_start_min:.0f}-{cfg.drift_ramp_end_min:.0f} min")
    for name, kpi, label in (
        (f"lp_{args.tag}", kpi_lp, "LP rebalancer"),
        (f"rl_{args.tag}", kpi_rl, "RL attention (sampled)"),
    ):
        path = out / f"nyc16_{name}_seed{args.seed}.mp4"
        render_animation(
            rec_lp if name.startswith("lp") else rec_rl,
            out_path=str(path), fps=args.fps,
            description=f"{label} - {desc} - SR {kpi['service_rate']:.1f}%",
        )
        print(f"\n=== {label} ===")
        print("  SR %.1f%%  served %d  cancelled %d  wait %.2f min"
              % (kpi["service_rate"], kpi["served"], kpi["cancelled"], kpi["avg_wait"]))
        print("  empty %.0f km  relocate %.1f%%  served before/after drift %d/%d"
              % (kpi["empty_km"], kpi["relocate_pct"], kpi["before"], kpi["after"]))
        print("  ->", path)


if __name__ == "__main__":
    main()