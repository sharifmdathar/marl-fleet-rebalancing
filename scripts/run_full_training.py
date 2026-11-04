#!/usr/bin/env python3
"""
Non-interactive training harness for MARL Fleet Rebalancing improvements.
Trains all IPPO and MAPPO checkpoints automatically.
"""

import subprocess
import sys
from pathlib import Path


def run_wsl_command(cmd: str, description: str = "") -> bool:
    """Run a command through WSL and return success status."""
    full_cmd = ["wsl", "bash", "-c", f"cd /home/luffy/marl-fleet-rebalancing && {cmd}"]
    
    if description:
        print(f"\n{'=' * 60}")
        print(description)
        print(f"{'=' * 60}")
    
    print(f"Running: {cmd[:80]}...")
    
    try:
        result = subprocess.run(full_cmd, capture_output=False, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("ERROR: WSL not found. Please install WSL or check your PATH.")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("MARL Fleet Rebalancing - Automated Training Suite")
    print("Running all IPPO and MAPPO checkpoints (15 runs total)")
    print("=" * 60)
    
    # Run all training and exit
    success = True
    
    # IPPO training
    print("\n" + "=" * 60)
    print("IPPO TRAINING (40/80/200 iterations, 5 seeds each)")
    print("=" * 60)
    
    for iterations in [40, 80, 200]:
        for seed in range(5):
            name = f"ippo-bandra-{iterations}-s{seed}"
            cmd = f"mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo ippo --iterations {iterations} --eval-interval 10 --eval-episodes 12 --checkpoint-interval {iterations} --seed {seed} --name {name}"
            if not run_wsl_command(cmd, f"Training {name}"):
                print(f"WARNING: {name} failed but continuing...")
                success = False
    
    # MAPPO training
    print("\n" + "=" * 60)
    print("MAPPO TRAINING (40/80/200 iterations, 5 seeds each)")
    print("=" * 60)
    
    for iterations in [40, 80, 200]:
        for seed in range(5):
            name = f"mappo-bandra-{iterations}-s{seed}"
            cmd = f"mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo mappo --iterations {iterations} --eval-interval 10 --eval-episodes 12 --checkpoint-interval {iterations} --seed {seed} --name {name}"
            if not run_wsl_command(cmd, f"Training {name}"):
                print(f"WARNING: {name} failed but continuing...")
                success = False
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TRAINING COMPLETED SUCCESSFULLY")
    else:
        print("SOME RUNS FAILED - CHECK LOGS ABOVE")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
