"""Fleet-scale sweep: does the 130-order cap distort the picture?

Motivation. The study build is 16 zones / 26 drivers / 130 orders per hour,
holding the historical 62-zone ratios (1.61 drivers per zone, 5 orders per
driver). Two questions this answers:

  1. Is the 130-order cap binding?  Measured: the 16 busiest zones contain a
     mean of 3,624 orders per hour, so the cap discards ~96% of them.
  2. Does the operating point change the ranking of the policies?

Scaling rule. Fleet and demand move together at the historical 5 orders per
driver, so every arm has the same pressure ratio and only the absolute load
changes. num_zones / max_pending / demand_history_len are held FIXED at
16 / 16 / 4, which is what keeps the 16-zone checkpoints loadable -- the
network shape depends on those three values only.

Metrics. Service rate alone is purchasable, so every arm reports empty km and
mean wait too. Empty km as a raw sum is also hard to read, so we report the
mean distance of a single relocation leg (traced the same way as
scripts/diag_reloc_leg_distance.py: from the turn a relocation is commanded
to the turn the driver stops being `relocating`).

Caveat printed with the results: max_pending is an observation budget, not a
demand cap. Once concurrent pending approaches 16, the policy can no longer
see or address every waiting rider, and the learned arms are handicapped in a
way that has nothing to do with the cap hypothesis. The script reports peak
concurrent pending per arm so that limit is visible rather than hidden.

    uv run python scripts/sweep_fleet_scale.py --episodes 4 --video-episode 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
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

# (order cap, drivers) -- 5 orders per driver throughout
SCALES = [(130, 26), (260, 52), (520, 104)]

# The tracer reads the trace dict by IDENTITY, so run_episode must assign this
# global rather than copying keys into it -- a copy would miss the leg/status
# entries the tracer needs.
CUR: dict | None = None


def install_tracer(env) -> None:
    """Accumulate relocation legs exactly as the reward sees them.

    Events are read where the environment itself reads them -- inside
    _compute_zone_rewards -- rather than by instrumenting the state we are
    measuring afterwards, which would consume the delta before the real call.
    """
    orig = env._compute_zone_rewards
    n_drv = env._num_drivers

    def traced(events, dispatched_this_turn=None):
        t = CUR
        if t is None:
            return orig(events, dispatched_this_turn)
        for did, ev in (events or {}).items():
            if ev.get("is_empty_move", False):
                km = float(ev.get("distance_moved", 0.0)) / 1000.0
                t["empty_km"] = t.get("empty_km", 0.0) + km
                if did < n_drv and t["status"][did] == "relocating":
                    t["leg_m"][did] = t["leg_m"].get(did, 0.0) + km
        out = orig(events, dispatched_this_turn)
        for did in range(n_drv):
            cur = env._last_obs[did]["self"].get("status")
            if t["status"][did] == "relocating" and cur != "relocating":
                km = t["leg_m"].pop(did, 0.0)
                if km > 0.0:
                    t["legs"].append(km)
            t["status"][did] = cur
        return out

    env._compute_zone_rewards = traced


def run_episode(env, policy, seed, trace: dict):
    """Play one episode, accumulating KPIs plus per-leg relocation distance."""
    global CUR
    env.reset(seed=seed)
    n_drv = env._num_drivers
    trace["leg_m"] = {}
    trace["status"] = [""] * n_drv
    trace["legs"] = []
    trace["empty_km"] = 0.0
    trace.setdefault("peak_pending", 0)
    CUR = trace

    rec = TrajectoryRecorder()
    rec.relocation_points = list(env.env.relocation_points)
    last_time = -1.0
    reloc_cmds = 0
    total = 0
    is_rl = hasattr(policy, "act")

    while env.agents:
        agent = env.agent_selection
        obs = obs_to_vector(env.observe(agent))
        mask = env.action_mask(agent)
        if is_rl:
            with torch.no_grad():
                a, _, _ = policy.act(obs[None, :], mask[None, :], deterministic=True)
            action = int(a[0])
        else:
            action = int(policy.select_action(agent, env, obs))
        if not mask[action]:
            action = int(np.flatnonzero(mask)[0])
        total += 1
        if action > env._max_pending:
            reloc_cmds += 1
        env.step(action)
        peak = sum(int(env.observe(z)["pending_passengers"].sum())
                   for z in env.possible_agents)
        trace["peak_pending"] = max(trace.get("peak_pending", 0), peak)
        # Snapshot ONCE per simulated time-step, not once per agent step.
        # With 16 agents per turn, snapshotting every step records ~16x too
        # many frames and the rendered episode runs minutes long instead of
        # the 60 seconds it represents.
        if env.env.time > last_time:
            rec.snapshot(env.env)
            last_time = env.env.time

    k = env.kpi_summary()
    served, cancelled = k["served"], k["cancelled"]
    legs = trace["legs"]
    trace.update({
        "sr": 100.0 * served / max(served + cancelled, 1.0),
        "served": served,
        "cancelled": cancelled,
        "wait": k["avg_wait"],
        "empty_km": k["empty_km"],
        "mean_leg_km": float(np.mean(legs)) if legs else 0.0,
        "n_legs": len(legs),
        "reloc_cmds": reloc_cmds,
        "actions": total,
    })
    return k, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=12000)
    ap.add_argument("--mappo", default="runs/d83-chain-mappo-s1/checkpoint.pt")
    ap.add_argument("--ippo", default="runs/d83-chain-ippo-s1/checkpoint.pt")
    ap.add_argument("--video-episode", type=int, default=0,
                    help="which episode index to render (-1 disables)")
    ap.add_argument("--video-scale", type=int, default=520,
                    help="which order cap the videos are rendered at")
    ap.add_argument("--out", default="results/fleet_scale_sweep")
    ap.add_argument("--scale", default="",
                    help="single 'cap:drivers' arm, e.g. 130:30, overriding "
                         "the default ladder")
    ap.add_argument("--video-policies", default="greedy,lp,mappo",
                    help="comma-separated subset to render: greedy,lp,ippo,mappo")
    ap.add_argument("--timeout", type=float, default=None,
                    help="TEMPORARY rider patience in minutes (env default 5.0). "
                         "Raising this raises service rate mechanically -- riders "
                         "who wait instead of cancelling count as served -- so it "
                         "must be labelled alongside any service number.")
    args = ap.parse_args()

    scales = ([tuple(int(x) for x in args.scale.split(":"))]
              if args.scale else SCALES)
    video_on = set(args.video_policies.split(","))

    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for cap, drivers in scales:
        # obs_inbound must match the checkpoints, or the destination head is
        # built with a different number of per-destination columns and cannot
        # run at all. d83-chain-{ippo,mappo}-s1 are the only matched
        # IPPO+MAPPO pair on disk; both were trained obs_inbound=True, beta 0.12.
        cfg = replace(NYCConfig(), num_drivers=drivers,
                      nyc_window_limit=cap, max_pending=16,
                      obs_inbound=True, beta=0.12,
                      order_timeout=args.timeout if args.timeout
                      else NYCConfig().order_timeout)
        env = RideGymAEC(cfg)
        install_tracer(env)

        mappo, _, menv = load_trained_policy(Path(args.mappo))
        ippo, _, ienv = load_trained_policy(Path(args.ippo))
        greedy = GreedyZoneDispatcher()
        lp = ZoneLPRebalancer()

        arms = [("Greedy (no relocation)", greedy),
                ("LP rebalancer (tuned)", lp),
                ("IPPO", ippo),
                ("MAPPO", mappo)]

        print(f"\n=== cap={cap}  drivers={drivers}  "
              f"orders/driver={cap/drivers:.1f}  agents={len(env.possible_agents)} ===")
        print(f"  {'policy':24s} {'SR%':>7s} {'km':>7s} {'km/leg':>7s} "
              f"{'legs':>6s} {'wait':>6s} {'SR/100km':>9s} {'peakPend':>9s}")

        for label, pol in arms:
            per_ep = []
            for ep in range(args.episodes):
                seed = args.base_seed + ep
                if not hasattr(pol, "act"):
                    pol.reset(seed=seed)
                trace: dict = {"peak_pending": 0}
                k, rec = run_episode(env, pol, seed, trace)
                per_ep.append(trace)

                tag = label.split()[0].lower()
                if (args.video_episode >= 0 and cap == args.video_scale
                        and ep == args.video_episode
                        and tag in video_on):
                    path = out / "videos" / f"{tag}_cap{cap}_d{drivers}.mp4"
                    render_animation(
                        rec, out_path=str(path), fps=10, detailed=True,
                        description=(f"{label} | cap {cap} orders/h | "
                                     f"{drivers} drivers | seed {seed}"),
                    )
                    print(f"    rendered {path.name}")

            def ms(key):
                return float(np.mean([e[key] for e in per_ep]))

            empty = ms("empty_km")
            sr = ms("sr")
            row = {
                "order_cap": cap,
                "drivers": drivers,
                "orders_per_driver": round(cap / drivers, 2),
                "order_timeout_min": cfg.order_timeout,
                "policy": label,
                "service_rate_pct": round(sr, 2),
                "empty_km": round(empty, 1),
                "mean_reloc_leg_km": round(ms("mean_leg_km"), 3),
                "reloc_legs": round(ms("n_legs"), 1),
                "avg_wait_min": round(ms("wait"), 3),
                "sr_per_100km": round(100.0 * sr / empty, 2) if empty else None,
                "peak_pending": int(max(e["peak_pending"] for e in per_ep)),
                "episodes": args.episodes,
            }
            rows.append(row)
            print(f"  {label:24s} {row['service_rate_pct']:7.2f} "
                  f"{row['empty_km']:7.0f} {row['mean_reloc_leg_km']:7.2f} "
                  f"{row['reloc_legs']:6.0f} {row['avg_wait_min']:6.2f} "
                  f"{row['sr_per_100km']:9.2f} {row['peak_pending']:9d}")

    (out / "sweep_results.json").write_text(json.dumps(rows, indent=2))
    with (out / "sweep_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out/'sweep_results.json'}")
    print(f"wrote {out/'sweep_results.csv'}")
    print(f"videos in {out/'videos'}")
    print("\nNOTE: max_pending=16 is an observation budget, not a demand cap. "
          "Where peak_pending approaches 16 the learned arms can no longer "
          "address every waiting rider. Read peak_pending before comparing.")


if __name__ == "__main__":
    main()