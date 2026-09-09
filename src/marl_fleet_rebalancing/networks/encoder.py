"""Neural network architectures for the MARL policies.

Zone-controller observation encoder + shared actor-critic used by IPPO and
MAPPO. One actor network is shared across all zone agents (parameter
sharing) — zone identity enters through ``zone_location``, per D13. MAPPO
keeps the shared actor and replaces the decentralized critic head with
:class:`CentralizedCritic` over the global state.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Dict


def obs_to_vector(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a zone-controller observation dict into a fixed-order vector.

    Key order must match the encoder's expected input layout; it is derived
    from the wrapper's observation space (see pettingzoo_env.py / D13).
    """
    return np.concatenate(
        [
            obs["zone_location"].ravel(),
            obs["zone_state"].ravel(),
            obs["all_zone_state"].ravel(),
            obs["pending_orders"].ravel(),
            obs["pending_mask"].ravel(),
            obs["pending_passengers"].ravel(),
            obs["time_frac"].ravel(),
            obs["relocation_points"].ravel(),
        ]
    ).astype(np.float32)


def obs_dim(obs_space: Dict) -> int:
    """Flat observation dimension from the wrapper's Dict observation space."""
    total = 0
    for key in (
        "zone_location", "zone_state", "all_zone_state", "pending_orders",
        "pending_mask", "pending_passengers", "time_frac", "relocation_points",
    ):
        total += int(np.prod(obs_space[key].shape))
    return total


class ZoneActorCritic(nn.Module):
    """Shared actor-critic MLP over flattened zone-controller observations.

    Actor head outputs logits over the joint dispatch+rebalance action space
    (D9); the caller masks illegal actions before sampling. Critic head is the
    decentralized value estimate for IPPO (own observation only).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (masked action logits, value estimates)."""
        h = self.trunk(obs)
        logits = self.actor(h)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits, self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions; returns (action, log_prob, value)."""
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), value


class CentralizedCritic(nn.Module):
    """Global-state value MLP for MAPPO's centralized critic (D4).

    Same trunk architecture as the actor (Linear-Tanh x2) but over the
    wrapper's centralized ``state()`` (all-zone fleet state + pending
    features + clock) instead of a single zone's observation.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimates for a batch of global states."""
        return self.value(self.trunk(state)).squeeze(-1)
