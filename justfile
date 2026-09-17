# Justfile for MARL Fleet Rebalancing
# High-productivity commands for training, evaluation, visualization, and QA.

default:
    @just --list

# ==============================================================================
# 1. Training Workflows
# ==============================================================================

# Train an IPPO policy (defaults: NYC Manhattan, 120 iters, seed 0, validation every 4 iters)
train-ippo env="nyc" iters="120" seed="0" workers="1" eval_interval="4" name="":
    #!/usr/bin/env bash
    CMD="uv run python -m marl_fleet_rebalancing.experiments.train --env {{env}} --algo ippo --iterations {{iters}} --seed {{seed}} --num-workers {{workers}}"
    if [ "{{eval_interval}}" != "0" ]; then CMD="$CMD --eval-interval {{eval_interval}} --eval-episodes 3"; fi
    if [ -n "{{name}}" ]; then CMD="$CMD --name {{name}}"; fi
    echo "Executing: $CMD"
    $CMD

# Train a MAPPO policy (centralized critic, NYC Manhattan, 120 iters, validation every 4 iters)
train-mappo env="nyc" iters="120" seed="0" workers="1" eval_interval="4" name="":
    #!/usr/bin/env bash
    CMD="uv run python -m marl_fleet_rebalancing.experiments.train --env {{env}} --algo mappo --iterations {{iters}} --seed {{seed}} --num-workers {{workers}}"
    if [ "{{eval_interval}}" != "0" ]; then CMD="$CMD --eval-interval {{eval_interval}} --eval-episodes 3"; fi
    if [ -n "{{name}}" ]; then CMD="$CMD --name {{name}}"; fi
    echo "Executing: $CMD"
    $CMD

# Quick smoke test training run (2 iterations on Bandra toy env)
train-smoke:
    uv run python -m marl_fleet_rebalancing.experiments.train \
        --env bandra --algo ippo --iterations 2 --eval-interval 1 --eval-episodes 1

# ==============================================================================
# 2. Evaluation & Demand Shift
# ==============================================================================

# Evaluate a checkpoint in-distribution (or on a specific shift, e.g. shift="S5")
eval checkpoint shift="" episodes="5":
    #!/usr/bin/env bash
    CMD="uv run python -m marl_fleet_rebalancing.experiments.evaluate --checkpoint {{checkpoint}} --episodes {{episodes}}"
    if [ -n "{{shift}}" ]; then CMD="$CMD --shift {{shift}}"; fi
    echo "Executing: $CMD"
    $CMD

# Run full S0–S6 degradation suite across all distribution shifts (5 seeds, summary.json & table)
eval-shifts checkpoint episodes="5":
    uv run python -m marl_fleet_rebalancing.experiments.shift_eval \
        --checkpoint {{checkpoint}} --episodes {{episodes}}

# Evaluate tuned LP rebalancer baseline on a specific shift (default S0)
eval-lp shift="S0" env="nyc" episodes="5":
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --policy lp --env {{env}} --shift {{shift}} --episodes {{episodes}}

# Evaluate Greedy baseline on a specific shift (default S0)
eval-greedy shift="S0" env="nyc" episodes="5":
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --policy greedy --env {{env}} --shift {{shift}} --episodes {{episodes}}

# Run full S0–S6 degradation evaluation on all baselines (LP and Greedy)
eval-baselines-shifts episodes="5":
    uv run python -m marl_fleet_rebalancing.experiments.shift_eval \
        --include-baselines --episodes {{episodes}}

# ==============================================================================
# 3. Interactive Visualization & Video Rendering
# ==============================================================================

# Open interactive GUI scrubber for a checkpoint on any shift (S0..S6)
view checkpoint shift="S0":
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --checkpoint {{checkpoint}} --shift {{shift}} --interactive --heatmap

# Open interactive GUI scrubber for LP rebalancer baseline
view-lp shift="S0" env="nyc":
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --policy lp --env {{env}} --shift {{shift}} --interactive --heatmap

# Open interactive GUI scrubber for Greedy baseline
view-greedy shift="S0" env="nyc":
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --policy greedy --env {{env}} --shift {{shift}} --interactive --heatmap

# Render smooth MP4 animation for a checkpoint on any shift (S0..S6)
render checkpoint shift="S0" out="episode.mp4" fps="15":
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --checkpoint {{checkpoint}} --shift {{shift}} --output {{out}} --fps {{fps}} --heatmap

# Render MP4 animation for LP rebalancer baseline on any shift
render-lp shift="S0" env="nyc" out="lp_episode.mp4" fps="15":
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --policy lp --env {{env}} --shift {{shift}} --output {{out}} --fps {{fps}} --heatmap

# ==============================================================================
# 4. Results & Plotting
# ==============================================================================

# Regenerate publication degradation figures from results/shift_eval/summary.json
plot:
    uv run python -m marl_fleet_rebalancing.experiments.plot_shifts

# ==============================================================================
# 5. Quality Assurance & Testing
# ==============================================================================

# Run test suite
test:
    uv run pytest -v

# Run linter
lint:
    uv run ruff check .

# Full QA verification: lint + test
check: lint test
