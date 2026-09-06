"""Smoke test for the RideGymAEC wrapper."""

from __future__ import annotations

import numpy as np

from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.config import BandraConfig


def main() -> None:
    cfg = BandraConfig()
    env = RideGymAEC(config=cfg, seed=0)
    env.reset(seed=0)
    obs, _reward, _term, _trunc, _info = env.last()
    print(f"agents: {env.agents}")
    print(f"num_zones: {env._num_zones}")
    print(f"max_pending: {env._max_pending}")
    print(f"action_dim: {env.action_space(env.agent_selection).n}")
    print(f"obs keys: {list(obs.keys())}")
    print(f"mask sum (initial): {env.action_mask(env.agent_selection).sum()}")

    total_step_calls = 0
    total_turns = 0
    total_reward = 0.0
    last_time = env.env.time
    for agent in env.agent_iter(max_iter=10_000):
        _obs, reward, term, trunc, info = env.last()
        if term or trunc:
            action = None
        else:
            mask = env.action_mask(agent)
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
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

    print(f"Total env.step() calls: {total_step_calls}")
    print(f"Total turns (env steps): {total_turns}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Expected: {total_turns} ~ 300 (30 min / 0.1 dt)")
    assert total_turns >= 295, f"too few turns: {total_turns}"
    assert total_turns <= 305, f"too many turns: {total_turns}"
    print("OK")


if __name__ == "__main__":
    main()
