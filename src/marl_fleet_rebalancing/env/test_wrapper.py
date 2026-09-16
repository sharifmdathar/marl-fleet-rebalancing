"""Smoke test for the RideGymAEC zone-controller wrapper.

Usage:
    uv run python -m marl_fleet_rebalancing.env.test_wrapper          # Bandra
    uv run python -m marl_fleet_rebalancing.env.test_wrapper --nyc    # NYC split
"""

from __future__ import annotations

import argparse

import numpy as np

from marl_fleet_rebalancing.config import BandraConfig, NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


def run(env: RideGymAEC, max_iter: int, expected_turns: int, tol: int = 10) -> None:
    print(f"agents ({len(env.possible_agents)}): {env.possible_agents[:3]} ...")
    print(f"num_zones: {env._num_zones}  num_drivers: {env._num_drivers}")
    print(f"max_pending: {env._max_pending}")
    print(f"action_dim: {env.action_space(env.agent_selection).n}")
    obs = env.observe(env.agent_selection)
    print(f"obs keys: {sorted(obs.keys())}")
    print(f"mask sum (initial): {env.action_mask(env.agent_selection).sum()}")

    total_step_calls = 0
    total_turns = 0
    total_reward = 0.0
    dispatches = 0
    relocations = 0
    last_time = env.env.time
    for agent in env.agent_iter(max_iter=max_iter):
        _obs, reward, term, trunc, _info = env.last()
        if term or trunc:
            action = None
        else:
            mask = env.action_mask(agent)
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
            if 1 <= action <= env._max_pending:
                dispatches += 1
            elif action > env._max_pending:
                relocations += 1
        env.step(action)
        total_step_calls += 1
        total_reward += reward
        if env.env.time > last_time:
            total_turns += 1
            last_time = env.env.time
            if total_turns % 50 == 0:
                print(f"  turn {total_turns}: time={env.env.time:.1f} min, "
                      f"pending={len(env._current_pending)}, "
                      f"agents_alive={len(env.agents)}, "
                      f"cumulative_reward={total_reward:.2f}")
        if env._terminated:
            break

    served = sum(
        1 for o in env.env.orders.values() if o.status.value == "completed"
    )
    cancelled = sum(
        1 for o in env.env.orders.values() if o.status.value == "cancelled"
    )
    print(f"Total wrapper.step() calls: {total_step_calls}")
    print(f"Total turns (env steps): {total_turns}")
    print(f"Zone actions taken: {dispatches} dispatches, {relocations} relocations")
    print(f"Total reward: {total_reward:.2f}")
    print(f"KPIs: served={served} cancelled={cancelled} "
          f"total_orders={len(env.env.orders)}")
    print(f"Expected: {expected_turns} +/- {tol} turns")
    assert expected_turns - tol <= total_turns <= expected_turns + tol, (
        f"unexpected turn count: {total_turns}"
    )
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zone-controller wrapper smoke test")
    parser.add_argument("--nyc", action="store_true", default=False,
                        help="Smoke-test the NYC split mode instead of Bandra")
    parser.add_argument("--limit", type=int, default=None,
                        help="Per-window order cap for the NYC run (speed)")
    args = parser.parse_args()

    if args.nyc:
        cfg = NYCConfig(nyc_window_limit=args.limit)
        env = RideGymAEC(config=cfg, seed=0)
        env.reset(seed=0)
        expected_turns = int(round(cfg.horizon / cfg.dt))
        run(env, max_iter=env._num_zones * (expected_turns + 10), expected_turns=expected_turns)
    else:
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=0)
        env.reset(seed=0)
        expected_turns = int(round(cfg.horizon / cfg.dt))
        run(env, max_iter=env._num_zones * (expected_turns + 10), expected_turns=expected_turns)


if __name__ == "__main__":
    main()
