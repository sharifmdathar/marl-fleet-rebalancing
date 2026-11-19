#!/usr/bin/env python
"""Full paired 12-window evaluation for the Plan B (D100) checkpoints.

Reuses the exact protocol of ``scripts/eval_full_study.py`` (same scenario
specs, same sequential zone-by-zone AEC loop, same 12 paired seeds
1000..1011, same Eq. (1) reward config) and writes to
``results/plan_b_v1/``. MLP-MAPPO rows are not re-run: they are read from
``results/full_study_v2/episodes.csv`` by the summary step.

    .venv/bin/python scripts/eval_plan_b.py --workers 4
    .venv/bin/python scripts/eval_plan_b.py --summarize-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "plan_b_v1"

# The reusable machinery reads these at import time (and workers re-import it),
# so they must be set before the import below.
os.environ["EVAL_FULL_STUDY_OUT"] = str(OUT)
os.environ["EVAL_FULL_STUDY_EPISODES"] = "12"

sys.path.insert(0, str(ROOT / "scripts"))
import eval_full_study as efs  # noqa: E402

RUNS = ROOT / "runs" / "plan_b"
ARMS = {
    "Attention-MAPPO": RUNS / "attention-mappo-s{seed}",
    "DR-MLP-MAPPO": RUNS / "dr-mlp-mappo-s{seed}",
    "DR-Attention-MAPPO": RUNS / "dr-attention-mappo-s{seed}",
}
SEEDS = range(5)
SAMPLED_SCENARIOS = ["S0", "S1", "S4", "S6"]


def sweep_labels() -> list[str]:
    specs = efs.scenario_specs()
    # The endpoints duplicate main S1/S4 exactly; reuse those rows.
    return [k for k in specs if "@" in k and k not in {"S1@p1.0", "S4@z1@60"}]


def ensure_checkpoints() -> None:
    missing = []
    for name, template in ARMS.items():
        for seed in SEEDS:
            run_dir = Path(str(template).format(seed=seed))
            for ckpt in ("checkpoint_it0200.pt", "checkpoint_best.pt"):
                if not (run_dir / ckpt).exists():
                    missing.append(str(run_dir / ckpt))
    if missing:
        raise FileNotFoundError(
            "missing Plan B checkpoints (train the queue first):\n  "
            + "\n  ".join(missing)
        )


def build_tasks() -> list[dict]:
    specs = efs.scenario_specs()
    sweeps = sweep_labels()
    tasks: list[dict] = []

    def add(arm: dict, scenarios: list[str], study: str) -> None:
        for sc in scenarios:
            assert sc in specs, sc
            tasks.append({"study": study, "arm": arm, "scenario": sc})

    add(efs.arm_spec("LP", "lp"), efs.MAIN_SCENARIOS, "main")
    add(efs.arm_spec("LP", "lp"), sweeps, "sweep")

    for arm_name, template in ARMS.items():
        for seed in SEEDS:
            run_dir = Path(str(template).format(seed=seed))
            it200 = efs.arm_spec(
                arm_name, "mappo", run_dir / "checkpoint_it0200.pt", True, seed,
            )
            add(it200, efs.MAIN_SCENARIOS, "main")
            add(it200, sweeps, "sweep")
            add(
                efs.arm_spec(
                    f"{arm_name}-best", "mappo",
                    run_dir / "checkpoint_best.pt", True, seed,
                ),
                efs.ABL_SCENARIOS, "ablation",
            )
            add(
                efs.arm_spec(
                    f"{arm_name}-it200-sampled", "mappo",
                    run_dir / "checkpoint_it0200.pt", False, seed,
                ),
                SAMPLED_SCENARIOS, "sampled",
            )
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    ensure_checkpoints()
    efs.OUT = OUT
    efs.PARTIAL = OUT / "_partial"
    efs.EPISODES = 12
    efs.build_tasks = build_tasks
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        efs.run_all(args.workers, args.limit_tasks, OUT, args.only)
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_plan_b_summary import summarize  # noqa: E402
    summarize(OUT)


if __name__ == "__main__":
    main()
