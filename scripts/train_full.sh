#!/bin/bash
# Full training sweep: IPPO + MAPPO over 40/80/200 iterations x 5 seeds.
#
# Throughput config (measured on this box, Bandra 25 zones, 360 turns/iter):
#   GPU serial (old default)   rollout 12.0s + update 0.3s = 12.3s/iter
#   CPU serial                 rollout  3.7s + update 0.3s =  4.0s/iter
#   CPU + 4 rollout workers    rollout  1.4s + update 0.3s =  1.7s/iter
# The rollout is one batch-1 forward per (turn, zone) - 9k per iteration - so
# every call pays a host->device copy plus a sync. On this net the GPU loses.
# Workers are CPU-only by construction (spawn context), hence --device cpu.
#
# Resumable: a run whose directory holds final_validation.json is skipped.
# A single failing run does not abort the sweep; its rc is logged.
#
# Usage: setsid nohup scripts/train_full.sh > training_output.log 2>&1 &
set -u

cd /home/luffy/marl-fleet-rebalancing || exit 1
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"

# One sweep at a time: a second copy would double RAM (4 GB/run) and OOM the box.
exec 9>/tmp/marl_train_full.lock
if ! flock -n 9; then
    echo "[sweep] another train_full.sh is already running; exiting"
    exit 1
fi

DEVICE="${DEVICE:-cpu}"
WORKERS="${WORKERS:-4}"
ITERS="${ITERS:-${*:-40 80 200}}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ALGOS="${ALGOS:-ippo mappo}"

echo "========================================"
echo "MARL Fleet Rebalancing - Full Training"
echo "device=$DEVICE workers=$WORKERS iters=$ITERS"
echo "started $(date -Is)"
echo "========================================"

train_checkpoint() {
    local algo=$1 iterations=$2 seed=$3
    local name="${algo}-bandra-${iterations}-s${seed}"
    local dir="runs/${name}"

    if [ -f "${dir}/final_validation.json" ]; then
        echo "[sweep] SKIP  ${name} (already complete)"
        return 0
    fi
    if [ -d "${dir}" ]; then
        echo "[sweep] CLEAN ${name} (partial run from an interrupted sweep)"
        rm -rf "${dir}"
    fi

    local t0=$SECONDS
    echo "[sweep] START ${name}  $(date -Is)"
    mise exec -- .venv/bin/python -u -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo "$algo" \
        --iterations "$iterations" \
        --seed "$seed" \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval "$iterations" \
        --device "$DEVICE" \
        --num-workers "$WORKERS" \
        --name "$name"
    local rc=$?
    echo "[sweep] END   ${name} rc=$rc in $((SECONDS - t0))s  $(date -Is)"
    echo ""
}

for algo in $ALGOS; do
    echo "=== ${algo^^} Training ==="
    for iterations in $ITERS; do
        for seed in $SEEDS; do
            train_checkpoint "$algo" "$iterations" "$seed"
        done
    done
done

echo "========================================"
echo "ALL TRAINING COMPLETED $(date -Is)"
echo "========================================"
