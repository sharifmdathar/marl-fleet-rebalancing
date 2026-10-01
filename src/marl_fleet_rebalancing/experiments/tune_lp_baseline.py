"""Tune the LP rebalancer baseline against the fleet's own non-RL floor.

The LP baseline is a *tuned* baseline (AGENTS.md rule 4): its two structural
knobs are the prediction horizon and the value of filling a predicted shortfall.
Both must be swept with the same effort spent on the RL agents, and the sweep
must stay off the held-out shifts (AGENTS.md rule 3).

Default sweep scenarios are S0 (train) and S3 (val). S1/S4/S5/S6 are evaluation
scenarios: looking at them while choosing hyperparameters would be test-set
tuning. Override with ``--shifts`` only if the scenario is not held out.

Usage:
    uv run python -m marl_fleet_rebalancing.experiments.tune_lp_baseline \\
        --lookahead 5 15 30 --penalty-scale 0.25 0.5 1.0 --episodes 3

Outputs:
    results/lp_tuning/sweep.json   — every config's per-shift KPIs
    results/lp_tuning/sweep.csv    — flat table of the same numbers
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from marl_fleet_rebalancing.baselines import (
    GreedyZoneDispatcher,
    LPBaselineConfig,
    ZoneLPRebalancer,
)
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.experiments.shift_eval import (
    SHIFTS,
    evaluate_policy_across_shifts,
)

OUT_DIR = Path("results/lp_tuning")

# Scenarios that may be inspected while tuning (train + val).
TUNING_SHIFTS = ("S0", "S3")  # train + val only; never the held-out eval scenarios


def _kpi(res: dict[str, Any], shift: str, metric: str) -> float:
    return float(res.get(shift, {}).get(metric, {}).get("mean", float("nan")))


def _mean_over(res: dict[str, Any], shifts: list[str], metric: str) -> float:
    vals = [_kpi(res, s, metric) for s in shifts]
    vals = [v for v in vals if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LP rebalancer hyperparameters.")
    parser.add_argument("--lookahead", type=float, nargs="+", default=[5.0, 15.0, 30.0],
                        help="Prediction horizons in minutes (default: 5 15 30)")
    parser.add_argument("--penalty-scale", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                        help="Deficit penalty, as a multiple of mean inter-zone travel time")
    parser.add_argument("--min-idle", type=int, nargs="+", default=[2],
                        help="min_idle_to_rebalance values to try (default: 2)")
    parser.add_argument("--shifts", type=str, nargs="+", default=list(TUNING_SHIFTS),
                        help=f"Shift IDs to tune on (default: {' '.join(TUNING_SHIFTS)}). "
                             "Do NOT add held-out evaluation scenarios.")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Seeds per scenario (default: 3; use >=5 for reported numbers)")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    known = {s[0] for s in SHIFTS}
    unknown = [s for s in args.shifts if s not in known]
    if unknown:
        raise SystemExit(f"Unknown shift ID(s): {unknown}. Known: {sorted(known)}")

    held_out = sorted(set(known) - set(args.shifts))
    shifts = [s for s in SHIFTS if s[0] in args.shifts]
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Tuning shifts : {args.shifts}")
    print(f"Held-out      : {held_out} (not inspected)")
    print(f"Seeds/scenario: {args.episodes} (base_seed={args.base_seed})")
    if args.episodes < 5:
        print("NOTE: <5 seeds is debugging output, not a reportable result (AGENTS.md rule 1).")

    base_cfg = NYCConfig()

    # Reference floor: the tuned baseline must beat this with real effort.
    print("\n" + "=" * 92)
    print("Reference policy: Greedy-Baseline (zero relocation)")
    print("=" * 92)
    greedy_res = evaluate_policy_across_shifts(
        policy_name="Greedy-Baseline",
        policy=GreedyZoneDispatcher(),
        policy_type="greedy",
        base_cfg=base_cfg,
        shifts=shifts,
        num_episodes=args.episodes,
        base_seed=args.base_seed,
    )

    rows: list[dict[str, Any]] = []
    for lookahead, penalty_scale, min_idle in itertools.product(
        args.lookahead, args.penalty_scale, args.min_idle
    ):
        lp_cfg = LPBaselineConfig(
            lookahead_minutes=lookahead,
            deficit_penalty_scale=penalty_scale,
            min_idle_to_rebalance=min_idle,
        )
        name = f"lp_la{lookahead:g}_p{penalty_scale:g}_mi{min_idle}"
        res = evaluate_policy_across_shifts(
            policy_name=name,
            policy=ZoneLPRebalancer(lp_cfg),
            policy_type="lp",
            base_cfg=base_cfg,
            shifts=shifts,
            num_episodes=args.episodes,
            base_seed=args.base_seed,
        )
        rows.append({
            "name": name,
            "lookahead_minutes": lookahead,
            "deficit_penalty_scale": penalty_scale,
            "min_idle_to_rebalance": min_idle,
            "seed_count": args.episodes,
            "base_seed": args.base_seed,
            "shifts": list(args.shifts),
            "results": res,
        })

    # ---------------------------------------------------------------- report
    header = (
        f"{'config':<24} | {'reloc%':>7} | {'empty_km':>8} | "
        + " | ".join(f"SR {s:<3}" for s in args.shifts)
    )
    print("\n" + "=" * 92)
    print("LP SWEEP")
    print("=" * 92)
    print(f"{'greedy (reference)':<24} | {_mean_over(greedy_res, list(args.shifts), 'pct_relocate'):7.2f} | "
          f"{_mean_over(greedy_res, list(args.shifts), 'empty_km'):8.1f} | "
          + " | ".join(f"{100 * _kpi(greedy_res, s, 'service_rate'):5.1f}%" for s in args.shifts))
    print("-" * 92)
    print(header)
    print("-" * 92)
    for row in rows:
        res = row["results"]
        sr_cells = " | ".join(
            f"{100 * _kpi(res, s, 'service_rate'):6.1f}%" for s in args.shifts
        )
        print(
            f"{row['name']:<24} | "
            f"{_mean_over(res, list(args.shifts), 'pct_relocate'):7.2f} | "
            f"{_mean_over(res, list(args.shifts), 'empty_km'):8.1f} | {sr_cells}"
        )

    # Acceptance check per AGENTS.md rules 3/4: the LP must actually act, and
    # must not lose to greedy on the scenarios it was tuned on.
    print("\nAcceptance (tuned-on shifts only):")
    for row in rows:
        res = row["results"]
        reloc = _mean_over(res, list(args.shifts), "pct_relocate")
        beats = all(
            _kpi(res, s, "service_rate") >= _kpi(greedy_res, s, "service_rate") - 1e-9
            for s in args.shifts
        )
        print(f"  {row['name']:<24} relocates={reloc > 0}  "
              f"SR >= greedy on {args.shifts}: {beats}")

    with open(out_path / "sweep.json", "w", encoding="utf-8") as f:
        json.dump({"greedy_reference": greedy_res, "sweeps": rows}, f, indent=2)

    with open(out_path / "sweep.csv", "w", encoding="utf-8") as f:
        f.write("config,lookahead_minutes,deficit_penalty_scale,min_idle_to_rebalance,"
                "shift,service_rate_mean,service_rate_std,pct_relocate_mean,empty_km_mean,"
                "avg_wait_min_mean,seed_count,base_seed\n")
        for row in rows:
            for shift_id in args.shifts:
                f.write(
                    f"{row['name']},{row['lookahead_minutes']},{row['deficit_penalty_scale']},"
                    f"{row['min_idle_to_rebalance']},{shift_id},"
                    f"{_kpi(row['results'], shift_id, 'service_rate')},"
                    f"{float(row['results'].get(shift_id, {}).get('service_rate', {}).get('std', 0.0))},"
                    f"{_kpi(row['results'], shift_id, 'pct_relocate')},"
                    f"{_kpi(row['results'], shift_id, 'empty_km')},"
                    f"{_kpi(row['results'], shift_id, 'avg_wait_min')},"
                    f"{args.episodes},{args.base_seed}\n"
                )
    print(f"\n[Saved sweep to {out_path / 'sweep.json'} and {out_path / 'sweep.csv'}]")


if __name__ == "__main__":
    main()
