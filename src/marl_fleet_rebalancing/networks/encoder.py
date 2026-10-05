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
            *([obs["inbound"].ravel()] if "inbound" in obs else []),
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
    if "inbound" in obs_space.spaces:
        total += int(np.prod(obs_space["inbound"].shape))
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
        inbound: bool = False,
    ) -> None:
        super().__init__()
        self._inbound = inbound
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
            *([("inbound", num_zones)] if inbound else []),
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
        # D54: destination-conditioned relocation head.
        #
        # Previously ALL action logits came from one pooled vector `h`, and the
        # relocation portion of that vector was a softmax-weighted SUM of zone
        # embeddings. A weighted sum is permutation-invariant over zones, so it
        # can encode "there is a lot of demand somewhere" but cannot bind that
        # demand to a particular destination. Measured: perturbing the demand
        # inputs moved the relocation logits substantially (L1 sensitivity
        # 2.9-15.3) yet the argmax target never changed -- the policy read
        # demand magnitude without direction, and aimed at a fixed zone
        # regardless of where demand was.
        #
        # This head instead scores each candidate destination against the
        # agent's own state, so every destination keeps its identity:
        #
        #     dest_logits[j] = <W_dest . zone_emb[j], W_self . self_ctx> / sqrt(d)
        #
        # plus a per-destination bias from that zone's own raw demand and car
        # counts, and a learned travel-cost term from the centroid geometry. The
        # dispatch and no-op logits are unchanged, so the action space, the
        # rollout buffer and the observation space are all untouched.
        self.dest_key = nn.Linear(embed_dim, hidden_dim)
        self.self_query = nn.Linear(self_ctx_dim, hidden_dim)
        # Raw per-zone scalars, so the head has direct access to demand and
        # supply per destination without having to reconstruct them from an
        # embedding that was pooled away.
        # D81: with `inbound`, each destination also carries the cars already
        # heading there, so zones can avoid piling onto one need.
        self.dest_bias = nn.Linear(4 + hist_len + int(inbound), 1)
        # Per-zone identity, so the head can reason about distance-to-each
        # destination rather than about an anonymous pooled vector.
        self.zone_idx = nn.Embedding(num_zones, 2)
        self.zone_centroid = nn.Linear(2, 2, bias=False)
        self.travel_cost = nn.Linear(2, 1)
        # D75: the term above is linear in (target - own position), so the own
        # position cancels across targets and no distance can be expressed.
        # Feed the real centroid distance instead. Starts at zero effect.
        # ponytail: Euclidean on bbox-normalised coordinates; switch to road
        # travel time if the anisotropy matters.
        self.register_buffer("zone_xy", torch.zeros(num_zones, 2))
        self.dist_cost = nn.Linear(1, 1)
        # Adam moves a weight ~lr per step, so on a 0.1-0.7 input it cannot
        # reach a useful magnitude in a few hundred steps; scale the input up.
        self.register_buffer("dist_input_scale", torch.tensor(10.0))
        nn.init.zeros_(self.dist_cost.weight)
        nn.init.zeros_(self.dist_cost.bias)
        self._dest_scale = hidden_dim ** 0.5

        self.actor = nn.Linear(hidden_dim, action_dim)
        # Zero the pooled head's relocation columns so the new head owns them
        # at initialisation, instead of both contributing.
        with torch.no_grad():
            self.actor.weight[1 + max_pending:, :].zero_()
            self.actor.bias[1 + max_pending:].zero_()
        self.critic = nn.Linear(hidden_dim, 1)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        # Checkpoints from before D75 lack the distance term; keep its zeros.
        new = {k: v for k, v in self.state_dict().items()
               if k.startswith(("zone_xy", "dist_cost"))}
        new["dist_input_scale"] = torch.tensor(1.0)  # pre-scale checkpoints trained unscaled
        return super().load_state_dict({**new, **state_dict}, strict=strict, **kwargs)

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

        # --- D54: destination-conditioned relocation logits ---
        # One logit per candidate destination, scored against this agent's own
        # state, so a destination's identity survives into the decision. The
        # zone embeddings are NOT pooled here; each one is scored on its own.
        keys = self.dest_key(zone_emb)                       # (B, Z, H)
        q = self.self_query(self_ctx).unsqueeze(1)            # (B, 1, H)
        dest = (keys * q).sum(-1) / self._dest_scale          # (B, Z)

        # Direct per-destination bias from that zone's own raw features: this
        # is the term that lets "zone 6 has riders, zone 14 does not" reach the
        # logits without passing through an attention-weighted sum.
        per_dest = [all_zone_state, demand_hist]
        if self._inbound:
            per_dest.append(self._slice(obs, "inbound").unsqueeze(-1))
        dest = dest + self.dest_bias(torch.cat(per_dest, dim=-1)).squeeze(-1)

        # Travel cost from the acting zone's centroid to each destination's, so
        # deadhead distance can enter the choice rather than only being billed
        # afterwards. Each destination is identified by a learned index
        # embedding, which the attention stack already treats as the zone's
        # identity; the relative offset to the acting zone is then computed from
        # those per-zone vectors. This avoids threading zone centroids through
        # the build API, and keeps the term fully learned.
        cents = self.zone_centroid(self.zone_idx.weight)        # (Z, 2)
        geo = torch.stack([
            cents[:, 0].unsqueeze(0) - zone_location[:, 0:1],
            cents[:, 1].unsqueeze(0) - zone_location[:, 1:2],
        ], dim=-1)                                              # (B, Z, 2)
        dest = dest + self.travel_cost(geo).squeeze(-1)
        dist = (self.zone_xy.unsqueeze(0) - zone_location.unsqueeze(1)).norm(dim=-1)
        dest = dest + self.dist_cost((dist * self.dist_input_scale).unsqueeze(-1)).squeeze(-1)

        # Scatter into the action head. Relocation action for zone j is
        # 1 + max_pending + j. Self-targeting is already excluded by the env:
        # `_legal_relocate_targets` skips `j == zone_idx`, and the action mask
        # enforces it, so no extra masking is needed here.
        n_reloc = dest.shape[1]
        base = 1 + self._max_pending
        logits[:, base:base + n_reloc] = logits[:, base:base + n_reloc] + dest

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
        base = 7 + num_zones * (5 + hist_len) + max_pending * 8
        return ZoneAttentionEncoder(
            num_zones, max_pending, hist_len, action_dim,
            hidden_dim=hidden_dim,
            inbound=observation_dim == base + num_zones,
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
        num_agents: int | None = None,
    ) -> None:
        """D39: ``num_agents`` enables a learned per-agent embedding.

        Without it, the only thing distinguishing one zone's value estimate from
        another's is whatever caller-supplied identity is appended to the state.
        In this repo that was the 2 normalised centroid coordinates, and it is a
        bad identity: the global state is captured ONCE per turn (train.py:308)
        and shared by every zone slot, so the critic must separate 16 different
        return functions using only 2 floats of geography. Adjacent zones get
        near-identical embeddings while facing unrelated demand.

        Measured consequence (PRE-D42, superseded -- kept as the record of why
        the embedding was needed): MAPPO explained_var was 0.018-0.039 against
        IPPO's 0.068-0.138. That figure is from the run whose identity was 2
        centroid floats. After this embedding the same runs give MAPPO 0.31-0.44
        vs IPPO 0.07-0.135, i.e. the critic is no longer the weak link. Do not
        cite the old numbers as current -- they describe a critic that no longer
        exists. Note also that explained_var is NOT like-for-like across arms:
        it scores returns derived from the values themselves, and a per-zone head
        has structurally more capacity to fit zone-specific structure.

        A learned nn.Embedding gives each zone its own free parameters, which is
        the standard MAPPO ``[state, agent_id]`` formulation.
        """
        super().__init__()
        self.num_agents = num_agents
        self.agent_embed = (
            nn.Embedding(num_agents, hidden_dim) if num_agents else None
        )
        extra = hidden_dim if self.agent_embed is not None else 0
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + extra, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor,
                agent_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return value estimates for a batch of global states.

        ``agent_ids`` is required when the critic was built with ``num_agents``.
        """
        if self.agent_embed is not None:
            if agent_ids is None:
                raise ValueError(
                    "critic was built with a per-agent embedding; agent_ids "
                    "must be supplied"
                )
            state = torch.cat(
                [state, self.agent_embed(agent_ids)], dim=-1,
            )
        return self.value(self.trunk(state)).squeeze(-1)
