"""Tune the LP rebalancer on TRAIN windows, ranked by the study's own objective.

A baseline has to be tuned with effort comparable to the method it is compared
against, on training/validation demand only (AGENTS.md). Every config sees the
same seeds, so the same train windows, and is ranked by the team objective
R = gamma*served - beta*empty_km - alpha*wait - cancel, with the three KPIs
printed next to it. The val and test splits are not touched here.

    uv run python scripts/tune_lp_objective.py --episodes 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_objective_gap import run_episode  # noqa: E402

from marl_fleet_rebalancing.baselines import ZoneLPRebalancer  # noqa: E402
from marl_fleet_rebalancing.baselines.lp_baseline import LPBaselineConfig  # noqa: E402
from marl_fleet_rebalancing.config import NYCConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=20000)
    ap.add_argument("--lookahead", type=float, nargs="+", default=[3.0, 5.0, 8.0, 12.0])
    ap.add_argument("--penalty", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--json-out", default="results/lp_tuning/objective_sweep.json")
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "train"
    cfg.relocator_share = 0.0  # zone rewards then sum to the team objective
    seeds = [args.base_seed + i for i in range(args.episodes)]
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"train split, {args.episodes} episodes per config, seeds {seeds[0]}..{seeds[-1]}")
    print(f"{'lookahead':>9}{'penalty':>9}{'SR%':>8}{'empty km':>10}{'wait':>7}{'reloc':>7}{'objective':>11}")
    results = []
    for look, pen in itertools.product(args.lookahead, args.penalty):
        lp_cfg = LPBaselineConfig(lookahead_minutes=look, deficit_penalty_scale=pen)
        rows = [run_episode(ZoneLPRebalancer(lp_cfg), cfg, s) for s in seeds]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in
                ("service_rate", "empty_km", "avg_wait", "reloc", "reward")}
        results.append({"lookahead_minutes": look, "deficit_penalty_scale": pen, **mean})
        print(f"{look:9.1f}{pen:9.2f}{mean['service_rate']:8.1f}{mean['empty_km']:10.1f}"
              f"{mean['avg_wait']:7.2f}{mean['reloc']:7.1f}{mean['reward']:11.1f}", flush=True)
        out.write_text(json.dumps({"split": "train", "seeds": seeds, "results": results}, indent=2))

    best = max(results, key=lambda r: r["reward"])
    print(f"\nbest by objective: lookahead={best['lookahead_minutes']} "
          f"penalty={best['deficit_penalty_scale']} -> objective {best['reward']:.1f}, "
          f"SR {best['service_rate']:.1f}%, empty {best['empty_km']:.1f} km")


if __name__ == "__main__":
    main()
