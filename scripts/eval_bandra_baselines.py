#!/usr/bin/env python
"""Bandra baseline evaluation for D95 — matched protocol to RL arms.

Evaluates Greedy and LP baselines on Bandra with:
- 12 episodes, deterministic argmax
- seeds 9000-9011 (12 seeds for 12 val windows, per D70)
- identical seeds across methods for paired comparison

The 5-seed subset (9000-9004) reproduces the D95 table exactly; the full
12-seed run is the D70 protocol and is also saved. Outputs:
- results/bandra_baselines_D95.json   — per-episode KPIs
- results/bandra_baselines_D95.csv    — per-episode CSV trace
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC
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
                import torch
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


def summarise(rows, n_seeds=None):
    """Mean +/- std over the first n_seeds episodes (default: all)."""
    if n_seeds is not None:
        rows = rows[:n_seeds]
    import statistics as st
    return {
        "service_rate": (float(np.mean([r["service_rate"] for r in rows])),
                         float(np.std([r["service_rate"] for r in rows]))),
        "empty_km": (float(np.mean([r["empty_km"] for r in rows])),
                     float(np.std([r["empty_km"] for r in rows]))),
        "avg_wait": (float(np.mean([r["avg_wait"] for r in rows])),
                     float(np.std([r["avg_wait"] for r in rows]))),
    }


def main():
    cfg = BandraConfig()
    episodes = 12
    base_seed = 9000
    seeds = [base_seed + i for i in range(episodes)]

    print(f"Bandra baseline evaluation: {episodes} episodes, seeds {seeds[0]}..{seeds[-1]}")
    print(f"Config: drivers={cfg.num_drivers}, orders={cfg.num_orders}, "
          f"zones={cfg.max_pending}, dt={cfg.dt}, horizon={cfg.horizon}")
    print(f"alpha={cfg.alpha}, beta={cfg.beta}, gamma={cfg.gamma}")

    arms = [
        ("Greedy", GreedyZoneDispatcher()),
        ("LP", ZoneLPRebalancer()),
    ]

    results = {}
    for name, policy in arms:
        rows = [run_episode(policy, cfg, s, deterministic=True) for s in seeds]
        results[name] = rows

        s5 = summarise(rows, 5)
        s12 = summarise(rows, 12)
        print(f"\n{name} (5 seeds 9000-9004):")
        print(f"  Service Rate: {s5['service_rate'][0]:.1f}% ± {s5['service_rate'][1]:.1f}%")
        print(f"  Empty km:     {s5['empty_km'][0]:.1f} ± {s5['empty_km'][1]:.1f}")
        print(f"  Wait min:     {s5['avg_wait'][0]:.2f} ± {s5['avg_wait'][1]:.2f}")
        print(f"\n{name} (12 seeds 9000-9011):")
        print(f"  Service Rate: {s12['service_rate'][0]:.1f}% ± {s12['service_rate'][1]:.1f}%")
        print(f"  Empty km:     {s12['empty_km'][0]:.1f} ± {s12['empty_km'][1]:.1f}")
        print(f"  Wait min:     {s12['avg_wait'][0]:.2f} ± {s12['avg_wait'][1]:.2f}")
        print(f"  Per-seed SR:  {', '.join(f'{r['service_rate']:.1f}%' for r in rows)}")

    # Save to JSON
    out_path = Path("results/bandra_baselines_D95.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Per-episode CSV for traceability
    csv_path = Path("results/bandra_baselines_D95.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "served", "cancelled", "service_rate",
                    "empty_km", "avg_wait", "mean_trip_km", "reloc"])
        for name, rows in results.items():
            for r in rows:
                w.writerow([name, r["seed"], r["served"], r["cancelled"],
                            r["service_rate"], r["empty_km"], r["avg_wait"],
                            r["mean_trip_km"], r["reloc"]])
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()