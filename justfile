# Justfile for MARL Fleet Rebalancing
# Usage:
#   just train-ippo
#   just train-mappo
#   just render-mp4 runs/my_run/checkpoint.pt
#   just view-interactive runs/my_run/checkpoint.pt

default:
    @just --list

# 1. Train decentralized IPPO policy
train-ippo env="bandra" iters="20" rollouts="3" target_zone="" target_pct="" name="":
    #!/usr/bin/env bash
    CMD="uv run python -m marl_fleet_rebalancing.experiments.train --env {{env}} --algo ippo --iterations {{iters}} --rollout-episodes {{rollouts}}"
    if [ -n "{{target_zone}}" ]; then CMD="$CMD --target-zone {{target_zone}}"; fi
    if [ -n "{{target_pct}}" ]; then CMD="$CMD --target-zone-demand-pct {{target_pct}}"; fi
    if [ -n "{{name}}" ]; then CMD="$CMD --name {{name}}"; fi
    echo "Executing: $CMD"
    $CMD

# 2. Train centralized-critic MAPPO policy
train-mappo env="bandra" iters="20" rollouts="3" target_zone="" target_pct="" name="":
    #!/usr/bin/env bash
    CMD="uv run python -m marl_fleet_rebalancing.experiments.train --env {{env}} --algo mappo --iterations {{iters}} --rollout-episodes {{rollouts}}"
    if [ -n "{{target_zone}}" ]; then CMD="$CMD --target-zone {{target_zone}}"; fi
    if [ -n "{{target_pct}}" ]; then CMD="$CMD --target-zone-demand-pct {{target_pct}}"; fi
    if [ -n "{{name}}" ]; then CMD="$CMD --name {{name}}"; fi
    echo "Executing: $CMD"
    $CMD

# 3. Evaluate a checkpoint and render an MP4 visualization
render-mp4 checkpoint target_zone="" target_pct="" out="episode.mp4" fps="10":
    #!/usr/bin/env bash
    CMD="uv run python -m marl_fleet_rebalancing.experiments.evaluate --checkpoint {{checkpoint}} --output {{out}} --fps {{fps}}"
    if [ -n "{{target_zone}}" ]; then CMD="$CMD --target-zone {{target_zone}}"; fi
    if [ -n "{{target_pct}}" ]; then CMD="$CMD --target-zone-demand-pct {{target_pct}}"; fi
    echo "Executing: $CMD"
    $CMD

# 4. Open the interactive GUI scrubber for a checkpoint episode
view-interactive checkpoint target_zone="" target_pct="":
    #!/usr/bin/env bash
    CMD="uv run python -m marl_fleet_rebalancing.experiments.evaluate --checkpoint {{checkpoint}} --interactive"
    if [ -n "{{target_zone}}" ]; then CMD="$CMD --target-zone {{target_zone}}"; fi
    if [ -n "{{target_pct}}" ]; then CMD="$CMD --target-zone-demand-pct {{target_pct}}"; fi
    echo "Executing: $CMD"
    $CMD
