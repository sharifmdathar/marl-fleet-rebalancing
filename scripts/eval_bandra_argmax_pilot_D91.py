#!/usr/bin/env python
"""Bandra argmax-vs-sampled pilot for D91 — 50 episodes, same seeds.

Compares deterministic (argmax) vs stochastic (sampled) action selection on
IPPO-200-s0 and MAPPO-200-s0, on identical demand windows. Decision criterion:
argmax wins if it beats sampled by >3pp on service rate (the <3pp simplicity
rule from D91); otherwise sampled is kept.

Outputs:
- results/bandra_argmax_pilot_D91/sweep_results.json
- results/bandra_argmax_pilot_D91/sweep_results.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def run_episode(policy, cfg, seed: int, deterministic: bool) -> dict:
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
            with torch.no_grad():
                a, _, _ = policy.act(
                    obs_to_vector(env.observe(ag))[None, :], m[None, :],
                    deterministic=deterministic)
            a = int(a[0])
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
    episodes = 50
    base_seed = 2000
    seeds = [base_seed + i for i in range(episodes)]

    print(f"Bandra argmax-vs-sampled pilot: {episodes} episodes, seeds {seeds[0]}..{seeds[-1]}")

    ippo_path = Path("runs/ippo-bandra-200-s0/checkpoint.pt")
    mappo_path = Path("runs/mappo-bandra-200-s0/checkpoint.pt")

    ippo, _, _ = load_trained_policy(ippo_path)
    mappo, _, _ = load_trained_policy(mappo_path)

    arms = [
        ("IPPO-200-s0", ippo),
        ("MAPPO-200-s0", mappo),
    ]

    all_rows = []
    cfg = BandraConfig()

    for name, policy in arms:
        for deterministic in (True, False):
            label = "argmax" if deterministic else "sampled"
            rows = [run_episode(policy, cfg, s, deterministic) for s in seeds]

            sr_vals = [r["service_rate"] for r in rows]
            empty_vals = [r["empty_km"] for r in rows]
            wait_vals = [r["avg_wait"] for r in rows]

            mean_sr = np.mean(sr_vals)
            mean_empty = np.mean(empty_vals)
            mean_wait = np.mean(wait_vals)

            print(f"\n{name} {label}:")
            print(f"  Service Rate: {mean_sr:.1f}% ± {np.std(sr_vals):.1f}%")
            print(f"  Empty km:     {mean_empty:.1f} ± {np.std(empty_vals):.1f}")
            print(f"  Wait min:     {mean_wait:.2f} ± {np.std(wait_vals):.2f}")

            for r in rows:
                all_rows.append({
                    "policy": name,
                    "mode": label,
                    "seed": r["seed"],
                    "service_rate": r["service_rate"],
                    "empty_km": r["empty_km"],
                    "avg_wait": r["avg_wait"],
                    "served": r["served"],
                    "cancelled": r["cancelled"],
                    "reloc": r["reloc"],
                })

    # Save
    out_dir = Path("results/bandra_argmax_pilot_D91")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    with open(out_dir / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nSaved to {out_dir}/sweep_results.json")
    print(f"Saved to {out_dir}/sweep_results.csv")

    # Print summary table matching D91
    print("\n=== D91 Summary Table ===")
    for name in ["IPPO-200-s0", "MAPPO-200-s0"]:
        sub = [r for r in all_rows if r["policy"] == name]
        for mode in ["argmax", "sampled"]:
            msub = [r for r in sub if r["mode"] == mode]
            sr = np.mean([r["service_rate"] for r in msub])
            ek = np.mean([r["empty_km"] for r in msub])
            wt = np.mean([r["avg_wait"] for r in msub])
            print(f"  {name:14s} {mode:8s} SR={sr:.1f}%  empty={ek:.1f}  wait={wt:.2f}")

        a = [r for r in sub if r["mode"] == "argmax"]
        s = [r for r in sub if r["mode"] == "sampled"]
        dsr = np.mean([r["service_rate"] for r in a]) - np.mean([r["service_rate"] for r in s])
        print(f"  {'':14s} {'dSR':>8s} {dsr:+.1f}pp")


if __name__ == "__main__":
    main()