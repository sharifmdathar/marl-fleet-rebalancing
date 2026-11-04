#!/bin/bash
# Overnight training launcher: all IPPO + MAPPO configs, then shift eval.
# Skips runs that already have a final checkpoint; deletes partial run dirs.
# Designed to run detached (nohup) and log everything to overnight/launcher.log.

cd /home/luffy/marl-fleet-rebalancing || exit 1
export PATH="$HOME/.local/share/mise/shims:$HOME/.local/bin:$PATH"
mkdir -p overnight

log() { echo "[$(date +%H:%M:%S)] $*"; }

train_one() {
    local algo=$1 iters=$2 seed=$3
    local name="${algo}-bandra-${iters}-s${seed}"
    local dir="runs/${name}"

    if [ -f "${dir}/checkpoint.pt" ]; then
        log "SKIP ${name} (checkpoint exists)"
        return 0
    fi
    if [ -d "${dir}" ]; then
        log "CLEAN partial ${name}"
        rm -rf "${dir}"
    fi

    log "START ${name}"
    if mise exec -- .venv/bin/python -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo "$algo" \
        --iterations "$iters" \
        --seed "$seed" \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval "$iters" \
        --name "$name" >> "overnight/${name}.log" 2>&1
    then
        log "DONE ${name}"
    else
        log "FAILED ${name} (see overnight/${name}.log)"
    fi
}

log "=== overnight training suite starting ==="

for iters in 40 80 200; do
    for seed in 0 1 2 3 4; do
        train_one ippo "$iters" "$seed"
    done
done
for iters in 40 80 200; do
    for seed in 0 1 2 3 4; do
        train_one mappo "$iters" "$seed"
    done
done

log "=== training suite finished; running shift evaluation ==="
if mise exec -- .venv/bin/python -m marl_fleet_rebalancing.experiments.shift_eval \
    --all-runs --include-baselines --episodes 5 \
    > overnight/shift_eval.log 2>&1
then
    log "DONE shift_eval"
else
    log "FAILED shift_eval (see overnight/shift_eval.log)"
fi

log "=== overnight suite complete ==="
