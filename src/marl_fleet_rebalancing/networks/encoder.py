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
    from the wrapper's observation space (see pettingzoo_env.py / D13/D23).
    """
    return np.concatenate(
        [
            obs["zone_location"].ravel(),
            obs["zone_state"].ravel(),
            obs["all_zone_state"].ravel(),
            obs["pending_orders"].ravel(),
            obs["pending_mask"].ravel(),
            obs["pending_passengers"].ravel(),
            obs["neighbor_mask"].ravel(),
            obs["demand_hist"].ravel(),
            obs["time_frac"].ravel(),
        ]
    ).astype(np.float32)


def obs_dim(obs_space: Dict) -> int:
    """Flat observation dimension from the wrapper's Dict observation space."""
    total = 0
    for key in (
        "zone_location", "zone_state", "all_zone_state", "pending_orders",
        "pending_mask", "pending_passengers", "neighbor_mask", "demand_hist",
        "time_frac",
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


class ZoneAttentionEncoder(nn.Module):
    """Structure-aware alternative trunk to :class:`ZoneActorCritic`.

    The flat-MLP trunk concatenates ``all_zone_state`` (62 zones) and
    ``pending_orders`` (30 slots) into one vector, so the network must
    relearn from scratch which entries are which zone/order and that the
    two sets are permutation-invariant. This encoder instead reconstructs
    both sets from the flat vector (same fixed layout as
    :func:`obs_to_vector`, so no wrapper/obs-space change is needed) and
    pools each with masked attention:

    - Orders: query built from the zone's own state, keys/values from a
      per-order embedding, masked by ``pending_mask`` — a permutation-
      invariant summary of "what's pending here" instead of a fixed-slot
      concatenation.
    - Zones: query built from the zone's own state, keys/values from a
      per-zone embedding of ``all_zone_state`` + ``demand_hist``, with an
      additive, learned bias from ``neighbor_mask`` so the network starts
      with "attend to my neighbors first" as an inductive prior but can
      still learn to look further (e.g. under demand shift).

    Same actor/critic head shapes as :class:`ZoneActorCritic`, so it is a
    drop-in replacement selected via ``IPPOConfig.arch`` /
    ``MAPPOConfig.arch`` (D33) — no change to the rollout buffer, obs
    space, or action space.
    """

    def __init__(
        self,
        num_zones: int,
        max_pending: int,
        hist_len: int,
        action_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self._num_zones = num_zones
        self._max_pending = max_pending
        self._hist_len = hist_len

        sizes = [
            ("zone_location", 2), ("zone_state", 4),
            ("all_zone_state", num_zones * 4),
            ("pending_orders", max_pending * 6),
            ("pending_mask", max_pending),
            ("pending_passengers", max_pending),
            ("neighbor_mask", num_zones),
            ("demand_hist", num_zones * hist_len),
            ("time_frac", 1),
        ]
        self._offsets: dict[str, tuple[int, int]] = {}
        o = 0
        for name, size in sizes:
            self._offsets[name] = (o, o + size)
            o += size

        self_ctx_dim = 2 + 4  # zone_location + zone_state
        self.order_embed = nn.Sequential(nn.Linear(7, embed_dim), nn.Tanh())
        self.order_query = nn.Linear(self_ctx_dim, embed_dim)
        self.zone_embed = nn.Sequential(
            nn.Linear(4 + hist_len, embed_dim), nn.Tanh(),
        )
        self.zone_query = nn.Linear(self_ctx_dim, embed_dim)
        self.neighbor_bias = nn.Parameter(torch.tensor(3.0))
        self._attn_scale = embed_dim ** 0.5

        ctx_dim = self_ctx_dim + embed_dim + embed_dim + 1  # + order/zone pools + time_frac
        self.trunk = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def _slice(self, obs: torch.Tensor, name: str) -> torch.Tensor:
        lo, hi = self._offsets[name]
        return obs[..., lo:hi]

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]
        zone_location = self._slice(obs, "zone_location")
        zone_state = self._slice(obs, "zone_state")
        all_zone_state = self._slice(obs, "all_zone_state").view(
            batch, self._num_zones, 4,
        )
        pending_orders = self._slice(obs, "pending_orders").view(
            batch, self._max_pending, 6,
        )
        pending_mask = self._slice(obs, "pending_mask")
        pending_passengers = self._slice(obs, "pending_passengers")
        neighbor_mask = self._slice(obs, "neighbor_mask")
        demand_hist = self._slice(obs, "demand_hist").view(
            batch, self._num_zones, self._hist_len,
        )
        time_frac = self._slice(obs, "time_frac")

        self_ctx = torch.cat([zone_location, zone_state], dim=-1)

        order_feat = torch.cat(
            [pending_orders, pending_passengers.unsqueeze(-1)], dim=-1,
        )
        order_emb = self.order_embed(order_feat)
        oq = self.order_query(self_ctx).unsqueeze(1)
        order_logits = (oq * order_emb).sum(-1) / self._attn_scale
        has_pending = pending_mask.sum(-1, keepdim=True) > 0
        order_logits = order_logits.masked_fill(pending_mask < 0.5, -1e9)
        order_attn = torch.softmax(order_logits, dim=-1) * has_pending.float()
        order_pool = (order_attn.unsqueeze(-1) * order_emb).sum(1)

        zone_feat = torch.cat([all_zone_state, demand_hist], dim=-1)
        zone_emb = self.zone_embed(zone_feat)
        zq = self.zone_query(self_ctx).unsqueeze(1)
        zone_logits = (zq * zone_emb).sum(-1) / self._attn_scale
        zone_logits = zone_logits + self.neighbor_bias * neighbor_mask
        zone_attn = torch.softmax(zone_logits, dim=-1)
        zone_pool = (zone_attn.unsqueeze(-1) * zone_emb).sum(1)

        ctx = torch.cat(
            [zone_location, zone_state, order_pool, zone_pool, time_frac],
            dim=-1,
        )
        h = self.trunk(ctx)
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
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), value


def build_zone_policy(
    arch: str,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    num_zones: int | None = None,
    max_pending: int | None = None,
    hist_len: int | None = None,
) -> nn.Module:
    """Construct the shared zone actor-critic for the requested ``arch``.

    ``"mlp"`` -> :class:`ZoneActorCritic` (flat trunk, D13/D23 baseline).
    ``"attention"`` -> :class:`ZoneAttentionEncoder` (D33; needs
    ``num_zones``/``max_pending``/``hist_len`` to reconstruct the obs
    layout). Both expose the same ``forward``/``act`` signature, so IPPO
    and MAPPO call sites don't branch on architecture.
    """
    if arch == "mlp":
        return ZoneActorCritic(observation_dim, action_dim, hidden_dim)
    if arch == "attention":
        if num_zones is None or max_pending is None or hist_len is None:
            raise ValueError(
                "attention arch requires num_zones, max_pending, hist_len",
            )
        return ZoneAttentionEncoder(
            num_zones, max_pending, hist_len, action_dim,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unknown arch: {arch!r} (expected 'mlp' or 'attention')")


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
