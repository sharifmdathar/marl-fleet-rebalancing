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
    "ippo-fixed-seed0": "runs/ippo-nyc-fixed-seed0/checkpoint.pt",
    "ippo-fixed-seed1": "runs/ippo-nyc-fixed-seed1/checkpoint.pt",
    "ippo-fixed-seed2": "runs/ippo-nyc-fixed-seed2/checkpoint.pt",
    "ippo-fixed-seed3": "runs/ippo-nyc-fixed-seed3/checkpoint.pt",
    "ippo-fixed-seed4": "runs/ippo-nyc-fixed-seed4/checkpoint.pt",
    "mappo-fixed-seed0": "runs/mappo-nyc-fixed-seed0/checkpoint.pt",
    "mappo-fixed-seed1": "runs/mappo-nyc-fixed-seed1/checkpoint.pt",
    "mappo-fixed-seed2": "runs/mappo-nyc-fixed-seed2/checkpoint.pt",
    "mappo-fixed-seed3": "runs/mappo-nyc-fixed-seed3/checkpoint.pt",
    "mappo-fixed-seed4": "runs/mappo-nyc-fixed-seed4/checkpoint.pt",
    "mappo-ent002-seed0": "runs/mappo-nyc-ent002-seed0/checkpoint.pt",
    "mappo-ent002-seed1": "runs/mappo-nyc-ent002-seed1/checkpoint.pt",
}

# Shift scenarios: (label, nyc_window_limit, target_zone, target_zone_pct, shift_param)
# shift_param is shift-type-specific:
#   "temporal"  : S3 — temporal phase shift (shift peak window by split)
#   "permute"   : S1 — spatial permutation (hotspots *move* by zone-offset rotation)
#   "concentrate": S4 — cold zone emergence (concentration into verifiably-cold train zone)
#   "burst"     : S5 — time-localized event burst (orders in [burst_start,burst_end) reassigned to burst_zone)
#   "real"      : S6 — real held-out demand (test split)
#   None        : no mechanism (magnitude scaling S2, in-distribution control S0,
#                  deprecated Sn1/Sn4/Sn5 for backward-compat with old CSVs)
# target_zone/target_zone_pct carry scenario-specific meaning per shift_param;
# when shift_param is None, target_zone/target_zone_pct_param are used directly as
# concentration params (old behavior, kept for the deprecated labels below).
SHIFTS = [
    ("S0",   500, None,  None,    None),         # In-distribution control
    ("S1",   500, None,  1.0,     "permute"),    # Spatial permutation, full strength (rotation offset = nz//2)
    ("S1a",  500, 0,    70,      None),           # DEPRECATED: static 70% conc z0 (was mislabeled S1a)
    ("S1b",  500, 30,   70,      None),           # DEPRECATED: static 70% conc z30 (was mislabeled S1b)
    ("S2a",  250, None,  None,    None),          # Magnitude ×0.5
    ("S2b",  375, None,  None,    None),          # Magnitude ×0.75
    ("S2d",  750, None,  None,    None),          # Magnitude ×1.5
    ("S2e",  1000, None, None,    None),          # Magnitude ×2.0
    ("S3",   500, None,  None,    "temporal"),    # Temporal phase shift (val split)
    ("S4",   500, 1,    60,       "concentrate"),  # Cold-zone emergence: 60% orders in zone 1 (cold in train split)
    ("S4a",  500, 60,    60,      None),           # DEPRECATED: static 60% conc z60 (was mislabeled S4a)
    ("S5",   500, 0,     20,       "burst"),       # Event burst: 20% of orders in [20,30) min -> zone 0
    ("S5a",  500, 0,     20,      None),           # DEPRECATED: static 20% conc z0 (was mislabeled S5)
    ("S6",   500, None,  None,    "real"),         # Real held-out demand (test split)
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
    """Create an environment with shifted demand configuration.

    mechanism — who moves, where, and when
    ----------------------------------------
    The three taxonomy scenarios (S1, S4, S5) are parameterized by mechanism,
    not by a single concentration percentage:

    S1   spatial permutation   hotspots *move* by a fixed zone-offset rotation;
           with strength p the order's own OD is remapped through that rotation.
           Same total volume, same timestamps, hotspots relocated not intensified.

    S4   cold zone (concentrate): demand is concentrated into a zone that is
           nearly absent in the training split. Configured via
           target_zone + target_zone_pct. The zone was cold in training so the
           policy has never been rewarded there.

    S5   event burst: a share (burst_share_pct) of orders whose request_time
           falls in [burst_start_min, burst_end_min) gets its origin reassigned
           to burst_zone. Same total volume and timestamps; only the time window
           spikes. Outside the window demand is untouched base demand.

    Old implementations conflated mechanism with concentration percent:
      - S1a/S1b  were 70% concentration to z0/z30 (NOT a moving hotspot)
      - S4a      was 60% concentration to z60 (NOT a new/held-out hotspot)
      - S5       was 20% concentration to z0  (NOT a time-localized burst)

    Those are now renamed (Sn1/Sn4/Sn5) below so the old results stay
    interpretable; the new entries have the mechanisms the taxonomy claims.
    """
    cfg = NYCConfig(
        nyc_window_limit=nyc_window_limit,
        nyc_split="train",
        order_source="nyc",
    )
    # Mechanism dispatch (independent of target_zone so S1 with
    # target_zone=None still activates the permute branch).
    if shift_param is not None:
        if shift_param == "temporal":
            # S3: Temporal phase shift — shift peak window by changing split
            if shift_label == "S3":
                cfg.nyc_split = "val"
            elif shift_label == "S3b":
                cfg.nyc_split = "test"
        elif shift_param == "burst":
            cfg.demand_shift_mode = "burst"
            cfg.burst_zone = target_zone if target_zone is not None else 0
            cfg.burst_share_pct = (
                target_zone_pct if target_zone_pct is not None else 20.0
            )
            cfg.target_zone = None
            cfg.target_zone_demand_pct = None
        elif shift_param == "permute":
            cfg.demand_shift_mode = "permute"
            cfg.perm_strength = (
                target_zone_pct if target_zone_pct is not None else 1.0
            )
            cfg.perm_offset = target_zone if target_zone is not None else None  # env computes n_zones // 2 at runtime
            cfg.target_zone = None
            cfg.target_zone_demand_pct = None
        elif shift_param == "concentrate":
            cfg.demand_shift_mode = "none"
        elif shift_param == "real":
            cfg.nyc_split = "test"
        else:
            raise ValueError(f"Unknown shift_param: {shift_param!r}")

    # Target-zone concentration (S4 cold-zone + deprecated backward-compat)
    if target_zone is not None and target_zone_pct is not None:
        if shift_param is None:
            # Deprecated backward-compat: static concentration
            cfg.target_zone = target_zone
            cfg.target_zone_demand_pct = target_zone_pct
        elif shift_param in ("concentrate",):
            cfg.target_zone = target_zone
            cfg.target_zone_demand_pct = target_zone_pct

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
            # Selection-ordered turn walk (matches the AEC step semantics:
            # step() records for the currently-selected agent). Terminated
            # turns drain without acting; live turns batch all zones into a
            # single deterministic act() — identical actions to per-zone
            # calls (argmax over the same logits, no sampling). Rewards use
            # observe=False (no redundant observation builds).
            order = [int(z) for z in env._agent_order]
            if env.terminations.get(env.agent_selection, False):
                total_reward += float(sum(
                    env.rewards.get(f"zone_{slot}", 0.0) for slot in order
                ))
                for _ in order:
                    if not env.agents:
                        break
                    env.step(None)
                continue
            obs_list = []
            mask_list = []
            for slot in order:
                agent = f"zone_{slot}"
                obs_list.append(obs_to_vector(env.observe(agent)))
                mask_list.append(env.action_mask(agent))
            # Turn reward = sum of every agent's post-flush credit, read once
            # (matches the old per-selection cumulative reads: step() zeroes
            # each agent's tables, so each credit is counted exactly once).
            total_reward += float(sum(
                env.rewards.get(f"zone_{slot}", 0.0) for slot in order
            ))
            actions, _, _ = policy.act(
                np.stack(obs_list), np.stack(mask_list), deterministic=True,
            )
            for slot, act_v in zip(order, actions):
                act_val = int(act_v)

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
        csv_path = RESULTS_DIR / f"{policy_name}.csv"
        with open(csv_path, "w") as f:
            f.write("shift,metric,mean,std,min,max\n")
            for shift_label, shift_data in policy_results.items():
                for metric, values in shift_data.items():
                    if isinstance(values, dict) and "mean" in values:
                        f.write(
                            f"{shift_label},{metric},{values['mean']},"
                            f"{values['std']},{values['min']},{values['max']}\n"
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