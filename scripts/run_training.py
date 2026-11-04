#!/usr/bin/env python3
"""
Simple training harness for MARL Fleet Rebalancing improvements.
Uses Python directly via subprocess with mise exec.
"""

import subprocess
import sys
from pathlib import Path


def run_wsl_command(cmd: list[str], description: str = "") -> bool:
    """Run a command through WSL and return success status."""
    if description:
        print(f"\n{'─' * 60}")
        print(description)
        print(f"{'─' * 60}\n")
    
    full_cmd = ["wsl", "bash", "-c", "cd /home/luffy/marl-fleet-rebalancing && " + cmd[0]]
    print(f"Running: {cmd[0]}")
    
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except FileNotFoundError:
        print("ERROR: WSL not found. Please install WSL or check your PATH.")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("MARL Fleet Rebalancing - Training Harness")
    print("Python mode with mise exec")
    print("=" * 60)
    
    while True:
        print("\n" + "=" * 60)
        print("Menu:")
        print("1. Train IPPO")
        print("2. Train MAPPO")
        print("3. Exit")
        print("=" * 60)
        
        choice = input("\nSelect (1-3): ").strip()
        
        if choice == "1":
            train_ippo_menu()
        elif choice == "2":
            train_mappo_menu()
        elif choice == "3":
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")


def train_ippo_menu():
    """IPPO training menu."""
    print("\n" + "=" * 60)
    print("IPPO Training Options")
    print("=" * 60)
    print("1. Train IPPO-40 (5 seeds)")
    print("2. Train IPPO-80 (5 seeds)")
    print("3. Train IPPO-200 (5 seeds)")
    print("4. Train all IPPO (40/80/200)")
    print("5. Back to main menu")
    
    choice = input("\nSelect (1-5): ").strip()
    
    if choice == "1":
        run_ippo_training(40)
    elif choice == "2":
        run_ippo_training(80)
    elif choice == "3":
        run_ippo_training(200)
    elif choice == "4":
        run_ippo_training(40)
        run_ippo_training(80)
        run_ippo_training(200)
    elif choice == "5":
        return
    else:
        print("Invalid choice.")


def train_mappo_menu():
    """MAPPO training menu."""
    print("\n" + "=" * 60)
    print("MAPPO Training Options")
    print("=" * 60)
    print("1. Train MAPPO-40 (5 seeds)")
    print("2. Train MAPPO-80 (5 seeds)")
    print("3. Train MAPPO-200 (5 seeds)")
    print("4. Train all MAPPO (40/80/200)")
    print("5. Back to main menu")
    
    choice = input("\nSelect (1-5): ").strip()
    
    if choice == "1":
        run_mappo_training(40)
    elif choice == "2":
        run_mappo_training(80)
    elif choice == "3":
        run_mappo_training(200)
    elif choice == "4":
        run_mappo_training(40)
        run_mappo_training(80)
        run_mappo_training(200)
    elif choice == "5":
        return
    else:
        print("Invalid choice.")


def run_ippo_training(iterations: int):
    """Run IPPO training for all 5 seeds."""
    for seed in range(5):
        name = f"ippo-bandra-{iterations}-s{seed}"
        cmd = f"mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo ippo --iterations {iterations} --eval-interval 10 --eval-episodes 12 --checkpoint-interval {iterations} --name {name}"
        run_wsl_command([cmd], f"Training IPPO-{iterations} seed {seed}")


def run_mappo_training(iterations: int):
    """Run MAPPO training for all 5 seeds."""
    for seed in range(5):
        name = f"mappo-bandra-{iterations}-s{seed}"
        cmd = f"mise exec -- python -m marl_fleet_rebalancing.experiments.train --env bandra --algo mappo --iterations {iterations} --eval-interval 10 --eval-episodes 12 --checkpoint-interval {iterations} --name {name}"
        run_wsl_command([cmd], f"Training MAPPO-{iterations} seed {seed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
        sys.exit(0)
