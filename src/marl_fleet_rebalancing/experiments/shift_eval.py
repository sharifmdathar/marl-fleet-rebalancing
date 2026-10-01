"""Demand distribution shift evaluation suite.

Freezes trained policies and evaluates degradation across shift scenarios
defined in docs/demand-shift.md.

Usage:
    # Evaluate specific checkpoint:
    uv run python -m marl_fleet_rebalancing.experiments.shift_eval \\
        --checkpoint runs/mappo-nyc-0916-192802/checkpoint.pt \\
        --include-baselines \\
        --episodes 5

    # Evaluate all available checkpoints in runs/ and baselines:
    uv run python -m marl_fleet_rebalancing.experiments.shift_eval \\
        --all-runs \\
        --include-baselines

Outputs:
    results/shift_eval/summary.json          — aggregated results by policy × shift
    results/shift_eval/{policy}.csv          — per-episode KPI traces
    results/shift_eval/degradation_table.md  — markdown summary table
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

RESULTS_DIR = Path("results/shift_eval")

# Active standardized shift scenarios:
# (label, nyc_window_limit, target_zone, target_zone_pct, shift_param, description)
# nyc_window_limit is the order cap per window. The 16-zone study runs 130
# orders against 26 cars (5 orders/driver); the magnitude scenarios scale that
# proportionally, so severity is a multiple of the in-distribution rate rather
# than an absolute order count.
_WINDOW = 130

SHIFTS = [
    ("S0", _WINDOW, None, None, None, "In-distribution base (train)"),
    ("S1", _WINDOW, None, 1.0, "permute", "Spatial permutation (rot offset = nz//2)"),
    ("S2a", int(_WINDOW * 0.5), None, None, None, "Magnitude ×0.5 (surge/trough)"),
    ("S2b", int(_WINDOW * 0.75), None, None, None, "Magnitude ×0.75"),
    ("S2d", int(_WINDOW * 1.5), None, None, None, "Magnitude ×1.5"),
    ("S2e", _WINDOW * 2, None, None, None, "Magnitude ×2.0"),
    ("S3", _WINDOW, None, None, "temporal", "Temporal phase shift (val split)"),
    ("S4", _WINDOW, 1, 60.0, "concentrate", "Cold-zone concentration (60% to z1)"),
    ("S5", _WINDOW, 0, 20.0, "burst", "Event burst (20% in [20,30]m to z0)"),
    ("S6", _WINDOW, None, None, "real", "Real held-out demand (test split)"),
    ("S7", _WINDOW, None, None, "drift", "Drifting hotspot (A=zone12 -> D=zone10)"),
]

EVAL_EPISODES = 5
BASE_SEED = 1000


def make_shifted_env(
    shift_label: str,
    nyc_window_limit: int,
    target_zone: int | None,
    target_zone_pct: float | None,
    shift_param: str | None = None,
    seed: int = 0,
    base_cfg: NYCConfig | None = None,
) -> RideGymAEC:
    """Create an environment with shifted demand configuration."""
    if base_cfg is not None:
        cfg = copy.deepcopy(base_cfg)
    else:
        cfg = NYCConfig()

    cfg.nyc_window_limit = nyc_window_limit
    cfg.nyc_split = "train"
    cfg.order_source = "nyc"
    cfg.demand_shift_mode = "none"
    cfg.target_zone = None
    cfg.target_zone_demand_pct = None
    cfg.burst_zone = None
    cfg.burst_share_pct = None
    cfg.perm_strength = None
    cfg.perm_offset = None

    # Mechanism dispatch
    if shift_param is not None:
        if shift_param == "temporal":
            if shift_label == "S3":
                cfg.nyc_split = "val"
            elif shift_label == "S3b":
                cfg.nyc_split = "test"
        elif shift_param == "burst":
            cfg.demand_shift_mode = "burst"
            cfg.burst_zone = target_zone if target_zone is not None else 0
            cfg.burst_share_pct = target_zone_pct if target_zone_pct is not None else 20.0
            cfg.burst_start_min = 20.0
            cfg.burst_end_min = 30.0
        elif shift_param == "permute":
            cfg.demand_shift_mode = "permute"
            cfg.perm_strength = target_zone_pct if target_zone_pct is not None else 1.0
            cfg.perm_offset = target_zone if target_zone is not None else None
        elif shift_param == "concentrate":
            cfg.demand_shift_mode = "none"
            cfg.target_zone = target_zone
            cfg.target_zone_demand_pct = target_zone_pct
        elif shift_param == "drift":
            cfg.demand_shift_mode = "drift"
        elif shift_param == "real":
            cfg.nyc_split = "test"
        else:
            raise ValueError(f"Unknown shift_param: {shift_param!r}")

    if shift_param is None and target_zone is not None and target_zone_pct is not None:
        cfg.target_zone = target_zone
        cfg.target_zone_demand_pct = target_zone_pct

    env = RideGymAEC(config=cfg, seed=seed)
    env.reset(seed=seed)
    return env


def evaluate_policy_on_env(
    policy: Any,
    env: RideGymAEC,
    policy_type: str = "checkpoint",
    num_episodes: int = 1,
    base_seed: int = BASE_SEED,
    deterministic: bool = True,
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

            total_reward += float(sum(
                env.rewards.get(f"zone_{slot}", 0.0) for slot in order
            ))

            if policy_type == "checkpoint":
                for slot in order:
                    agent = f"zone_{slot}"
                    obs = env.observe(agent)
                    mask = env.action_mask(agent)
                    obs_vec = obs_to_vector(obs)
                    actions, _, _ = policy.act(
                        obs_vec[None, :], mask[None, :], deterministic=deterministic,
                    )
                    act_val = int(actions[0])
                    if act_val == 0:
                        acts_idle += 1
                    elif 1 <= act_val <= env._max_pending:
                        acts_dispatch += 1
                    else:
                        acts_relocate += 1
                    env.step(act_val)

            elif policy_type in ("greedy", "lp"):
                for slot in order:
                    agent = f"zone_{slot}"
                    obs = env.observe(agent)
                    act_val = policy.select_action(agent, env, obs)
                    if act_val == 0:
                        acts_idle += 1
                    elif 1 <= act_val <= env._max_pending:
                        acts_dispatch += 1
                    else:
                        acts_relocate += 1
                    env.step(act_val)

            elif policy_type == "random":
                for slot in order:
                    agent = f"zone_{slot}"
                    mask = env.action_mask(agent)
                    valid_acts = np.where(mask)[0]
                    act_val = int(np.random.choice(valid_acts)) if len(valid_acts) > 0 else 0
                    if act_val == 0:
                        acts_idle += 1
                    elif 1 <= act_val <= env._max_pending:
                        acts_dispatch += 1
                    else:
                        acts_relocate += 1
                    env.step(act_val)
            else:
                raise ValueError(f"Unknown policy_type: {policy_type}")

        kpi = env.kpi_summary()
        sr = float(kpi["served"]) / max(float(kpi["served"]) + float(kpi["cancelled"]), 1.0)
        total_acts = max(acts_idle + acts_dispatch + acts_relocate, 1)

        episode_results.append({
            "episode": float(ep + 1),
            "seed": float(seed),
            "served": float(kpi["served"]),
            "cancelled": float(kpi["cancelled"]),
            "service_rate": sr,
            "avg_wait_min": float(kpi["avg_wait"]),
            "empty_km": float(kpi["empty_km"]),
            "total_reward": total_reward,
            "pct_dispatch": float(100.0 * acts_dispatch / total_acts),
            "pct_relocate": float(100.0 * acts_relocate / total_acts),
            "pct_idle": float(100.0 * acts_idle / total_acts),
        })

    return episode_results


def evaluate_policy_across_shifts(
    policy_name: str,
    policy: Any,
    policy_type: str,
    base_cfg: NYCConfig | None,
    shifts: list[tuple],
    num_episodes: int = EVAL_EPISODES,
    base_seed: int = BASE_SEED,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate a single policy across all shift scenarios."""
    print(f"\n{'='*70}")
    print(f"Evaluating Policy: {policy_name} ({policy_type})")
    print(f"{'='*70}")

    results: dict[str, Any] = {}

    for shift_label, w_limit, t_zone, t_pct, shift_param, desc in shifts:
        shift_key = shift_label
        print(f"  [{shift_label:<4}] {desc:<45} (limit={w_limit})...", end="", flush=True)

        all_episode_results: list[dict[str, float]] = []

        for seed_idx in range(num_episodes):
            env = make_shifted_env(
                shift_label,
                w_limit,
                t_zone,
                t_pct,
                shift_param=shift_param,
                seed=base_seed + seed_idx,
                base_cfg=base_cfg,
            )
            ep_results = evaluate_policy_on_env(
                policy=policy,
                env=env,
                policy_type=policy_type,
                num_episodes=1,
                base_seed=base_seed + seed_idx,
                deterministic=deterministic,
            )
            all_episode_results.extend(ep_results)
            del env

        # Aggregate episode results
        if all_episode_results:
            avg: dict[str, Any] = {}
            for k in all_episode_results[0].keys():
                vals = [r[k] for r in all_episode_results]
                avg[k] = {
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals)), 6),
                    "min": round(float(np.min(vals)), 6),
                    "max": round(float(np.max(vals)), 6),
                }
            avg["episodes"] = len(all_episode_results)
            results[shift_key] = avg

            sr = avg.get("service_rate", {}).get("mean", 0)
            sr_std = avg.get("service_rate", {}).get("std", 0)
            empty = avg.get("empty_km", {}).get("mean", 0)
            wait = avg.get("avg_wait_min", {}).get("mean", 0)
            print(f" -> SR={sr*100:5.1f}% ±{sr_std*100:4.1f}% | Wait={wait:4.2f}m | Empty={empty:5.0f}km")

    return results


def format_markdown_table(all_results: dict[str, dict[str, Any]]) -> str:
    """Generate a clean markdown summary table comparing policies across shifts."""
    lines = []
    lines.append("### Demand Shift Degradation Summary (Service Rate % ± Std)")
    lines.append("")
    header = "| Policy | " + " | ".join([s[0] for s in SHIFTS]) + " | Mean Retention |"
    sep = "|:---| " + " | ".join([":---:" for _ in SHIFTS]) + " | :---: |"
    lines.append(header)
    lines.append(sep)

    for policy_name, policy_res in all_results.items():
        s0_sr = policy_res.get("S0", {}).get("service_rate", {}).get("mean", None)
        retention_list = []
        row_items = [f"**{policy_name}**"]

        for s_label, _, _, _, _, _ in SHIFTS:
            s_data = policy_res.get(s_label, {}).get("service_rate", {})
            sr_mean = s_data.get("mean", None)
            sr_std = s_data.get("std", 0.0)
            if sr_mean is not None:
                row_items.append(f"{sr_mean*100:.1f}% ± {sr_std*100:.1f}%")
                if s0_sr and s0_sr > 0 and s_label != "S0":
                    retention_list.append(sr_mean / s0_sr)
            else:
                row_items.append("N/A")

        mean_ret = f"{np.mean(retention_list)*100:.1f}%" if retention_list else "N/A"
        row_items.append(mean_ret)
        lines.append("| " + " | ".join(row_items) + " |")

    lines.append("")
    lines.append("### Empty Repositioning Distance (km / episode)")
    lines.append("")
    header = "| Policy | " + " | ".join([s[0] for s in SHIFTS]) + " |"
    sep = "|:---| " + " | ".join([":---:" for _ in SHIFTS]) + " |"
    lines.append(header)
    lines.append(sep)

    for policy_name, policy_res in all_results.items():
        row_items = [f"**{policy_name}**"]
        for s_label, _, _, _, _, _ in SHIFTS:
            e_mean = policy_res.get(s_label, {}).get("empty_km", {}).get("mean", None)
            if e_mean is not None:
                row_items.append(f"{e_mean:.0f} km")
            else:
                row_items.append("N/A")
        lines.append("| " + " | ".join(row_items) + " |")

    lines.append("")
    lines.append("### Average Passenger Waiting Time (min)")
    lines.append("")
    header = "| Policy | " + " | ".join([s[0] for s in SHIFTS]) + " |"
    sep = "|:---| " + " | ".join([":---:" for _ in SHIFTS]) + " |"
    lines.append(header)
    lines.append(sep)

    for policy_name, policy_res in all_results.items():
        row_items = [f"**{policy_name}**"]
        for s_label, _, _, _, _, _ in SHIFTS:
            w_mean = policy_res.get(s_label, {}).get("avg_wait_min", {}).get("mean", None)
            if w_mean is not None:
                row_items.append(f"{w_mean:.2f} min")
            else:
                row_items.append("N/A")
        lines.append("| " + " | ".join(row_items) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policies across demand shift suite.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific model checkpoint (e.g. runs/mappo-nyc-.../checkpoint.pt)",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Discover and evaluate all checkpoints found in runs/",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also evaluate non-RL baselines (Greedy & LP)",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Only evaluate non-RL baselines (Greedy & LP)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation seeds per scenario (default: 5)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help="Base seed for evaluation (default: 1000)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic sampling instead of deterministic argmax",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for results",
    )
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    policies_to_evaluate = []

    # 1. Checkpoint discovery
    if args.checkpoint:
        ckpt_p = Path(args.checkpoint)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        p_name = ckpt_p.parent.name
        policies_to_evaluate.append((p_name, "checkpoint", ckpt_p))
    elif args.all_runs:
        runs_dir = Path("runs")
        if runs_dir.exists():
            for p in sorted(runs_dir.iterdir()):
                ckpt = p / "checkpoint.pt"
                if ckpt.exists():
                    policies_to_evaluate.append((p.name, "checkpoint", ckpt))
    elif not args.baselines_only:
        # Default: pick latest run in runs/
        runs_dir = Path("runs")
        found = False
        if runs_dir.exists():
            runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for r in runs:
                ckpt = r / "checkpoint.pt"
                if ckpt.exists():
                    policies_to_evaluate.append((r.name, "checkpoint", ckpt))
                    found = True
                    break
        if not found:
            print("No checkpoints found. Falling back to baselines.")
            args.include_baselines = True

    # 2. Baselines
    if args.include_baselines or args.baselines_only:
        policies_to_evaluate.append(("Greedy-Baseline", "greedy", None))
        policies_to_evaluate.append(("LP-Rebalancer", "lp", None))

    all_results: dict[str, dict[str, Any]] = {}

    for policy_name, p_type, ckpt_or_none in policies_to_evaluate:
        if p_type == "checkpoint":
            learner, manifest, eval_env = load_trained_policy(Path(ckpt_or_none))
            base_cfg = eval_env._cfg
            res = evaluate_policy_across_shifts(
                policy_name=policy_name,
                policy=learner,
                policy_type="checkpoint",
                base_cfg=base_cfg,
                shifts=SHIFTS,
                num_episodes=args.episodes,
                base_seed=args.base_seed,
                deterministic=not args.stochastic,
            )
        elif p_type == "greedy":
            policy = GreedyZoneDispatcher()
            base_cfg = NYCConfig()
            res = evaluate_policy_across_shifts(
                policy_name=policy_name,
                policy=policy,
                policy_type="greedy",
                base_cfg=base_cfg,
                shifts=SHIFTS,
                num_episodes=args.episodes,
                base_seed=args.base_seed,
                deterministic=True,
            )
        elif p_type == "lp":
            policy = ZoneLPRebalancer()
            base_cfg = NYCConfig()
            res = evaluate_policy_across_shifts(
                policy_name=policy_name,
                policy=policy,
                policy_type="lp",
                base_cfg=base_cfg,
                shifts=SHIFTS,
                num_episodes=args.episodes,
                base_seed=args.base_seed,
                deterministic=True,
            )
        elif p_type == "random":
            policy = None
            base_cfg = NYCConfig()
            res = evaluate_policy_across_shifts(
                policy_name=policy_name,
                policy=policy,
                policy_type="random",
                base_cfg=base_cfg,
                shifts=SHIFTS,
                num_episodes=args.episodes,
                base_seed=args.base_seed,
                deterministic=True,
            )
        else:
            continue

        all_results[policy_name] = res

        # Write per-policy CSV
        csv_file = out_path / f"{policy_name}.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("shift,metric,mean,std,min,max\n")
            for shift_k, s_data in res.items():
                for m_k, m_vals in s_data.items():
                    if isinstance(m_vals, dict) and "mean" in m_vals:
                        f.write(
                            f"{shift_k},{m_k},{m_vals['mean']},{m_vals['std']},"
                            f"{m_vals['min']},{m_vals['max']}\n"
                        )

    # Save summary JSON
    summary_path = out_path / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved JSON summary to {summary_path}]")

    # Generate Markdown table
    md_content = format_markdown_table(all_results)
    md_path = out_path / "degradation_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved Markdown report to {md_path}]")

    # Print summary table to stdout
    print("\n" + md_content + "\n")


if __name__ == "__main__":
    main()