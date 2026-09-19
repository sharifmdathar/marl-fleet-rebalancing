"""Greedy nearest-idle dispatch baseline for PettingZoo AEC zone controllers.

This policy serves as the standardized non-rebalancing floor (D3 / INTENT.md).
Zone controllers greedily assign idle drivers in their zone to the nearest
pending passenger order within their dispatch radius. They perform zero
proactive relocation (actions > max_pending are never chosen).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC

logger = logging.getLogger(__name__)


class GreedyZoneDispatcher:
    """Greedy nearest-idle dispatch policy for RideGymAEC.

    At each agent's turn:
    1. Query the valid action mask from the environment.
    2. Check for available dispatch actions (indices 1 to max_pending).
    3. Select the closest pending candidate order (lowest valid action index,
       as candidate orders are pre-sorted closest-first).
    4. Never issue proactive relocation actions.
    5. Return action 0 (no-op) if no valid orders can be dispatched.
    """

    def __init__(self) -> None:
        pass

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state at episode boundary."""
        pass

    def select_action(
        self,
        agent: str,
        env: "RideGymAEC",
        obs: dict[str, Any] | None = None,
    ) -> int:
        """Select a greedy action for the currently acting zone agent.

        Args:
            agent: Agent identifier string (e.g. 'zone_0').
            env: The RideGymAEC environment instance.
            obs: Zone observation dictionary (optional).

        Returns:
            int action in [0, 1 + max_pending).
        """
        mask = env.action_mask(agent)
        max_pending = env._max_pending

        # Valid dispatch actions lie in [1, 1 + max_pending]
        dispatch_mask = mask[1 : 1 + max_pending]
        valid_indices = np.where(dispatch_mask)[0]

        if len(valid_indices) > 0:
            candidates = env._zone_candidate_orders[env._zone_int_id(agent)]
            centroid = np.asarray(env.relocation_points[env._zone_int_id(agent)])
            nearest = min(
                valid_indices,
                key=lambda i: np.sum(
                    (np.asarray(candidates[int(i)]["origin"]) - centroid) ** 2
                ),
            )
            return int(1 + nearest)

        return 0
