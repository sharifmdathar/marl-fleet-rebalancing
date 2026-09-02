"""Random matching policy for the ride-pooling env.

Each driver greedily picks one unclaimed pending order from its observation
(if any), otherwise idles. Routing is handled by the env.
"""
from __future__ import annotations

from typing import Any

from ride_gym import RidePoolEnv


def random_actions(env: RidePoolEnv, obs: dict[str, Any]) -> dict[str, Any]:
    assigned: set = set()
    actions: dict[str, Any] = {}
    for vid, s_v in obs.items():
        pool = s_v.get("pending_orders", [])
        available = [o for o in pool if o["order_id"] not in assigned]
        if available:
            order = available[0]
            assigned.add(order["order_id"])
            actions[vid] = {"orders": [order["order_id"]]}
        else:
            actions[vid] = {"orders": []}
    return actions


def rollout(env: RidePoolEnv, seed: int = 0) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        actions = random_actions(env, obs)
        obs, rewards, dones, info = env.step(actions)
        done = dones["__all__"]
    return info
