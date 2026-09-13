"""Demand distribution shift evaluation suite.

Freezes trained policies and evaluates degradation across shift scenarios
defined in docs/demand-shift.md.

Usage:
    uv run python -m marl_fleet_rebalancing.experiments.shift_eval

Outputs:
    results/shift_eval/summary.json   — aggregated results by policy × shift
    results/shift_eval/{policy}_{shift}.csv — per-episode KPI traces
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

RESULTS_DIR = Path("results/shift_eval")
CHECKPOINTS = {
    "ippo-proper-seed0": "runs/ippo-nyc-proper-seed0/checkpoint.pt",
    "ippo-proper-seed1": "runs/ippo-nyc-proper-seed1/checkpoint.pt",
    "mappo-proper-seed0": "runs/mappo-nyc-proper-seed0/checkpoint.pt",
    "mappo-proper-seed1": "runs/mappo-nyc-proper-seed1/checkpoint.pt",
}

# Shift scenarios: (label, nyc_window_limit, target_zone, target_zone_pct, shift_param)
# shift_param is shift-type-specific:
#   S3: temporal phase shift — change nyc_split to simulate shifted peak window
#   S5: event spike — target_zone + target_zone_pct concentrates demand in one zone
#   S6: real held-out — nyc_split="test" uses real FHVHV test-split windows
# None target_zone means uniform demand (no spatial shift)
SHIFTS = [
    ("S0",  500, None,  None, None),    # In-distribution control
    ("S1a", 500, 0,    70, None),       # Spatial concentration (70% zone 0)
    ("S1b", 500, 30,   70, None),       # Spatial concentration (70% zone 30)
    ("S2a", 250, None,  None, None),    # Magnitude ×0.5
    ("S2b", 375, None,  None, None),    # Magnitude ×0.75
    ("S2d", 750, None,  None, None),    # Magnitude ×1.5
    ("S2e", 1000, None, None, None),    # Magnitude ×2.0
    ("S3",  500, None,  None, "temporal"),   # Temporal phase shift (val split)
    ("S4a", 500, 60,   60, None),       # New hotspot (60% zone 60)
    ("S5",  500, 0,    20, "event"),       # Event spike: 20% zone 0 burst
    ("S6",  500, None,  None, "real"),      # Real held-out demand (test split)
]
# Note: S2c (magnitude ×1.0) removed 2026-09-14 — byte-identical config to S0
# (verified: S2c reproduced S0 exactly), pure compute waste. See docs/demand-shift.md.

EVAL_EPISODES = 5
BASE_SEED = 1000


def make_shifted_env(
    shift_label: str,
    nyc_window_limit: int,
    target_zone: int | None,
    target_zone_pct: float | None,
    shift_param: str | None = None,
    seed: int = 0,
) -> RideGymAEC:
    """Create an environment with shifted demand configuration."""
    cfg = NYCConfig(
        nyc_window_limit=nyc_window_limit,
        nyc_split="train",
        order_source="nyc",
    )
    # Apply spatial shift via existing config parameters
    if target_zone is not None and target_zone_pct is not None:
        cfg.target_zone = target_zone
        cfg.target_zone_demand_pct = target_zone_pct

    # Apply shift-type-specific parameters
    if shift_param is not None:
        if shift_param == "temporal":
            # S3: Temporal phase shift — shift peak window by changing split
            # The nyc_window_limit controls per-window order cap;
            # temporal shift changes which window (peak vs off-peak) is used.
            # We shift the split cyclic index to simulate a shifted peak.
            if shift_label == "S3":
                # S3 uses val split to simulate shifted peak window
                cfg.nyc_split = "val"
            elif shift_label == "S3b":
                cfg.nyc_split = "test"
            # Note: S3 gets nyc_window_limit=500 same as baseline;
            # the split change simulates a temporal shift of the demand peak.
        elif shift_param == "event":
            # S5: Event spike — 20% of orders concentrated in target_zone as a burst
            # target_zone and target_zone_pct already set above
            pass
        elif shift_param == "real":
            # S6: Real held-out demand — use the test split of real FHVHV data
            cfg.nyc_split = "test"
            # nyc_window_limit=500 already set above; test split has its own windows

    env = RideGymAEC(config=cfg, seed=seed)
    env.reset(seed=seed)
    return env


def evaluate_policy(
    policy: Any,
    env: RideGymAEC,
    num_episodes: int = 1,
    base_seed: int = BASE_SEED,
) -> list[dict[str, float]]:
    """Run evaluation episodes over fresh seeds. Returns list of KPI dicts."""
    episode_results: list[dict[str, float]] = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset(seed=seed)

        acts_idle = acts_dispatch = acts_relocate = 0
        total_reward = 0.0

        while env.agents and not env._terminated:
            agent = env.agent_selection
            _obs, reward, term, trunc, _info = env.last()
            total_reward += float(reward)

            if term or trunc:
                env.step(None)
                continue

            obs = env.observe(agent)
            mask = env.action_mask(agent)

            obs_vec = obs_to_vector(obs)
            action, _logprob, _val = policy.act(
                obs_vec[None, :], mask[None, :], deterministic=True,
            )
            act_val = int(action[0])

            if act_val == 0:
                acts_idle += 1
            elif 1 <= act_val <= env._max_pending:
                acts_dispatch += 1
            else:
                acts_relocate += 1

            env.step(act_val)

        kpi = env.kpi_summary()
        sr = float(kpi["served"]) / max(float(kpi["served"]) + float(kpi["cancelled"]), 1.0)

        episode_results.append({
            "episode": ep + 1,
            "seed": seed,
            "served": float(kpi["served"]),
            "cancelled": float(kpi["cancelled"]),
            "service_rate": sr,
            "avg_wait_min": float(kpi["avg_wait"]),
            "empty_km": float(kpi["empty_km"]),
            "total_reward": total_reward,
            "pct_dispatch": float(100.0 * acts_dispatch / max(acts_idle + acts_dispatch + acts_relocate, 1)),
            "pct_relocate": float(100.0 * acts_relocate / max(acts_idle + acts_dispatch + acts_relocate, 1)),
            "pct_idle": float(100.0 * acts_idle / max(acts_idle + acts_dispatch + acts_relocate, 1)),
        })

    return episode_results


def evaluate_checkpoint(
    policy_name: str,
    checkpoint_path: str,
    shifts: list[tuple],
    num_episodes: int = EVAL_EPISODES,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Evaluate a single checkpoint across all shift scenarios."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {policy_name}")
    print(f"{'='*60}")

    # Load policy using the existing loader (handles checkpoint + config)
    learner, manifest, _ = load_trained_policy(Path(checkpoint_path))
    algo = manifest.get("algo", "ippo")
    print(f"  Loaded {algo} policy, scenario={manifest.get('scenario_id')}")

    results: dict[str, Any] = {}

    for shift_label, w_limit, t_zone, t_pct, shift_param in shifts:
        shift_key = shift_label
        if t_zone is not None:
            shift_key += f"_z{t_zone}"

        shift_desc = f"limit={w_limit}"
        if t_zone is not None:
            shift_desc += f", zone={t_zone} pct={t_pct}%"
        print(f"\n  Shift {shift_label} ({shift_desc})...")

        all_episode_results: list[dict[str, float]] = []

        for seed_idx in range(num_episodes):
            env = make_shifted_env(
                shift_label, w_limit, t_zone, t_pct, shift_param=shift_param, seed=seed_idx,
            )
            ep_results = evaluate_policy(
                learner, env, num_episodes=1,
                base_seed=base_seed + seed_idx,
            )
            all_episode_results.extend(ep_results)
            del env

        # Aggregate episode results
        if all_episode_results:
            avg: dict[str, Any] = {}
            for k in all_episode_results[0].keys():
                vals = [r[k] for r in all_episode_results]
                if isinstance(vals[0], float):
                    avg[k] = {
                        "mean": round(float(np.mean(vals)), 6),
                        "std": round(float(np.std(vals)), 6),
                        "min": round(float(np.min(vals)), 6),
                        "max": round(float(np.max(vals)), 6),
                    }
                else:
                    avg[k] = vals[0]
            results[shift_key] = avg
            sr = avg.get("service_rate", {}).get("mean", 0)
            empty = avg.get("empty_km", {}).get("mean", 0)
            wait = avg.get("avg_wait_min", {}).get("mean", 0)
            reward = avg.get("total_reward", {}).get("mean", 0)
            print(f"    SR={sr*100:.1f}% | wait={wait:.2f}m | empty={empty:.0f}km | reward={reward:.0f}")

    return results


def main() -> None:
    """Run all evaluations and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}

    for policy_name, checkpoint in CHECKPOINTS.items():
        results = evaluate_checkpoint(policy_name, checkpoint, SHIFTS)
        all_results[policy_name] = results

    # Save summary JSON
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved summary to {summary_path}")

    # Save per-shift CSVs
    for policy_name, policy_results in all_results.items():
        for shift_label, shift_data in policy_results.items():
            csv_path = RESULTS_DIR / f"{policy_name}_{shift_label}.csv"
            with open(csv_path, "w") as f:
                f.write("metric,mean,std,min,max\n")
                for metric, values in shift_data.items():
                    if isinstance(values, dict) and "mean" in values:
                        f.write(
                            f"{metric},{values['mean']},{values['std']},"
                            f"{values['min']},{values['max']}\n"
                        )

    # Print degradation summary table
    print("\n\n" + "="*100)
    print("DEGRADATION SUMMARY (Service Rate % vs Shift)")
    print("="*100)
    header = f"{'Policy':<20}"
    for shift_label, _, _, _, _ in SHIFTS:
        header += f"{shift_label:<10}"
    print(header)
    print("-"*100)
    for policy_name, policy_results in all_results.items():
        row = f"{policy_name:<20}"
        for shift_label, _, t_zone, _, _ in SHIFTS:
            key = shift_label + (f"_z{t_zone}" if t_zone is not None else "")
            sr = policy_results.get(key, {}).get("service_rate", {}).get("mean", None)
            if sr is not None:
                row += f"{sr*100:>7.1f}%   "
            else:
                row += f"{'N/A':>10}  "
        print(row)
    print("\n" + "="*100)
    print("EMPTY DISTANCE (km) vs Shift")
    print("="*100)
    header = f"{'Policy':<20}"
    for shift_label, _, _, _, _ in SHIFTS:
        header += f"{shift_label:<10}"
    print(header)
    print("-"*100)
    for policy_name, policy_results in all_results.items():
        row = f"{policy_name:<20}"
        for shift_label, _, t_zone, _, _ in SHIFTS:
            key = shift_label + (f"_z{t_zone}" if t_zone is not None else "")
            empty = policy_results.get(key, {}).get("empty_km", {}).get("mean", None)
            if empty is not None:
                row += f"{empty:>7.0f}km  "
            else:
                row += f"{'N/A':>10}  "
        print(row)
    print("\n" + "="*100)


if __name__ == "__main__":
    main()