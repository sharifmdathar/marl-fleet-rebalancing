#!/usr/bin/env bash
# Train seed replicas of the D28 paired protocol (IPPO + MAPPO) and
# shift-eval each replica into an isolated results directory.
#
# The run of record (results/shift_eval/, training seed 0) is NEVER touched:
# replica evals go to results/shift_eval_seeds/ and replica runs get
# distinct run names (-s<seed>-stageN), so downstream aggregation can
# merge both dirs by seed.
#
# Protocol per stage must stay identical to scripts/run_mappo_d28_pipeline.sh
# (D28: two-stage entropy 0.05 -> 0.01, neighbor relocation, val-selected
# checkpoint). Only --seed and run names differ.
#
# Usage:
#   ./scripts/run_d28_seed_replicas.sh            # seeds 1 and 2, both algos
#   ./scripts/run_d28_seed_replicas.sh 3 4        # explicit seed list
set -euo pipefail

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(1 2)
fi

for SEED in "${SEEDS[@]}"; do
  for ALGO in ippo mappo; do
    S1="runs/${ALGO}-nyc-d28-s${SEED}-stage1"
    S2="runs/${ALGO}-nyc-d28-s${SEED}-stage2"

    echo "========================================================================"
    echo " SEED ${SEED} / ${ALGO^^}: Stage 1 (50 iters @ entropy 0.05)"
    echo "========================================================================"
    uv run python -m marl_fleet_rebalancing.experiments.train \
      --env nyc --algo "${ALGO}" --seed "${SEED}" \
      --iterations 50 \
      --entropy-coef 0.05 \
      --eval-interval 5 \
      --eval-episodes 3 \
      --nyc-window-limit 500 \
      --name "${ALGO}-nyc-d28-s${SEED}-stage1" \
      --num-workers "${NUM_WORKERS:-1}"

    echo "========================================================================"
    echo " SEED ${SEED} / ${ALGO^^}: Stage 2 (150 iters @ entropy 0.01)"
    echo "========================================================================"
    uv run python -m marl_fleet_rebalancing.experiments.train \
      --env nyc --algo "${ALGO}" --seed "${SEED}" \
      --iterations 150 \
      --entropy-coef 0.01 \
      --eval-interval 5 \
      --eval-episodes 3 \
      --nyc-window-limit 500 \
      --resume "${S1}/checkpoint_best.pt" \
      --name "${ALGO}-nyc-d28-s${SEED}-stage2" \
      --num-workers 4

    echo "========================================================================"
    echo " SEED ${SEED} / ${ALGO^^}: 5-seed shift eval (isolated out-dir)"
    echo "========================================================================"
    uv run python -m marl_fleet_rebalancing.experiments.shift_eval \
      --checkpoint "${S2}/checkpoint_best.pt" \
      --episodes 5 \
      --out-dir results/shift_eval_seeds
  done
done

echo "All replicas done. Aggregate with:"
echo "  uv run python scripts/aggregate_seed_replicas.py"
