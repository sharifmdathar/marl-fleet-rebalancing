#!/usr/bin/env python
"""Bandra fleet-size evaluation for D93 — evaluation-time driver sweep on frozen policies.

Runs IPPO-200-s0 and MAPPO-200-s0 with varying num_drivers (frozen policies,
directional signal only — not retrained per fleet size).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def run_episode(policy, cfg, seed: int, deterministic: bool = True) -> dict:
    env = RideGymAEC(cfg)
    env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed=seed)

    reloc = 0
    while env.agents:
        for _ in range(env._num_zones):
            if not env.agents:
                break
            ag = env.agent_selection
            m = env.action_mask(ag)
            if hasattr(policy, "act"):
                with torch.no_grad():
                    a, _, _ = policy.act(
                        obs_to_vector(env.observe(ag))[None, :], m[None, :],
                        deterministic=deterministic)
                a = int(a[0])
            else:
                a = int(policy.select_action(ag, env, env.observe(ag)))
            if not m[a]:
                a = int(np.flatnonzero(m)[0])
            if a > env._max_pending:
                reloc += 1
            env.step(a)

    k = env.kpi_summary()
    served = int(k["served"])
    cancelled = int(k["cancelled"])
    terminal = int(k["total"])
    return {
        "seed": seed,
        "served": served,
        "cancelled": cancelled,
        "service_rate": 100.0 * served / max(terminal, 1),
        "empty_km": float(k["empty_km"]),
        "avg_wait": float(k["avg_wait"]),
        "mean_trip_km": float(k.get("mean_trip_km", 0.0)),
        "reloc": reloc,
    }


def main():
    # Driver counts from D93
    driver_counts = [20, 23, 26, 29, 32, 35]
    episodes = 10
    base_seed = 3000
    seeds = [base_seed + i for i in range(episodes)]

    print(f"Bandra fleet-size evaluation: {episodes} episodes, seeds {seeds[0]}..{seeds[-1]}")
    print(f"Driver counts: {driver_counts}")

    # Load the frozen checkpoints
    ippo_path = Path("runs/ippo-bandra-200-s0/checkpoint.pt")
    mappo_path = Path("runs/mappo-bandra-200-s0/checkpoint.pt")

    ippo, _, _ = load_trained_policy(ippo_path)
    mappo, _, _ = load_trained_policy(mappo_path)

    arms = [
        ("IPPO-200-s0", ippo),
        ("MAPPO-200-s0", mappo),
    ]

    all_rows = []

    for drivers in driver_counts:
        cfg = BandraConfig()
        cfg.num_drivers = drivers

        print(f"\n=== {drivers} drivers ===")
        print(f"  {'policy':16s} {'SR%':>8s} {'empty_km':>10s} {'wait_min':>8s}")

        for name, policy in arms:
            rows = [run_episode(policy, cfg, s, deterministic=True) for s in seeds]

            sr_vals = [r["service_rate"] for r in rows]
            empty_vals = [r["empty_km"] for r in rows]
            wait_vals = [r["avg_wait"] for r in rows]

            mean_sr = np.mean(sr_vals)
            mean_empty = np.mean(empty_vals)
            mean_wait = np.mean(wait_vals)

            print(f"  {name:16s} {mean_sr:7.1f}% {mean_empty:9.1f} {mean_wait:7.2f}")

            for r in rows:
                all_rows.append({
                    "policy": name,
                    "drivers": drivers,
                    "seed": r["seed"],
                    "service_rate": r["service_rate"],
                    "empty_km": r["empty_km"],
                    "avg_wait": r["avg_wait"],
                    "served": r["served"],
                    "cancelled": r["cancelled"],
                    "reloc": r["reloc"],
                })

    # Save
    out_dir = Path("results/bandra_fleet_D93")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    with open(out_dir / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nSaved to {out_dir}/sweep_results.json")
    print(f"Saved to {out_dir}/sweep_results.csv")

    # Print summary table matching D93
    print("\n=== D93 Summary Table ===")
    for name in ["IPPO-200-s0", "MAPPO-200-s0"]:
        sub = [r for r in all_rows if r["policy"] == name]
        for drivers in driver_counts:
            dsub = [r for r in sub if r["drivers"] == drivers]
            sr = np.mean([r["service_rate"] for r in dsub])
            ek = np.mean([r["empty_km"] for r in dsub])
            wt = np.mean([r["avg_wait"] for r in dsub])
            print(f"  {name}  {drivers}d  SR={sr:.1f}%  empty={ek:.0f}  wait={wt:.2f}")


if __name__ == "__main__":
    main()