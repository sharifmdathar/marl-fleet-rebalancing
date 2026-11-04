#!/bin/bash
# Run training jobs for the MARL fleet rebalancing improvement study
# Uses mise to manage Python/uv/just dependencies

set -e

cd /home/luffy/marl-fleet-rebalancing

echo "=== Training IPPO checkpoints (40/80/200 iterations, 5 seeds each) ==="

# Train IPPO at 40 iterations
echo "Training IPPO-40..."
for seed in 0 1 2 3 4; do
    echo "IPPO-40 seed $seed..."
    mise exec -- python -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo ippo \
        --seed $seed \
        --iterations 40 \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval 40 \
        --name ippo-bandra-40-s$seed
done

# Train IPPO at 80 iterations
echo "Training IPPO-80..."
for seed in 0 1 2 3 4; do
    echo "IPPO-80 seed $seed..."
    mise exec -- python -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo ippo \
        --seed $seed \
        --iterations 80 \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval 80 \
        --name ippo-bandra-80-s$seed
done

# Train IPPO at 200 iterations
echo "Training IPPO-200..."
for seed in 0 1 2 3 4; do
    echo "IPPO-200 seed $seed..."
    mise exec -- python -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo ippo \
        --seed $seed \
        --iterations 200 \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval 40 \
        --name ippo-bandra-200-s$seed
done

echo "=== Training MAPPO checkpoints ==="

# Train MAPPO at 40 iterations
echo "Training MAPPO-40..."
for seed in 0 1 2 3 4; do
    echo "MAPPO-40 seed $seed..."
    mise exec -- python -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo mappo \
        --seed $seed \
        --iterations 40 \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval 40 \
        --name mappo-bandra-40-s$seed
done

# Train MAPPO at 80 iterations
echo "Training MAPPO-80..."
for seed in 0 1 2 3 4; do
    echo "MAPPO-80 seed $seed..."
    mise exec -- python -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo mappo \
        --seed $seed \
        --iterations 80 \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval 80 \
        --name mappo-bandra-80-s$seed
done

# Train MAPPO at 200 iterations
echo "Training MAPPO-200..."
for seed in 0 1 2 3 4; do
    echo "MAPPO-200 seed $seed..."
    mise exec -- python -m marl_fleet_rebalancing.experiments.train \
        --env bandra \
        --algo mappo \
        --seed $seed \
        --iterations 200 \
        --eval-interval 10 \
        --eval-episodes 12 \
        --checkpoint-interval 40 \
        --name mappo-bandra-200-s$seed
done

echo "=== All training complete ==="
echo "Checkpoints written to runs/ directory"
