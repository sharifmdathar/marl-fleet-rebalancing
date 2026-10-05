"""MAPPO: centralized-critic PPO for the zone-controller fleet env.

Same study backbone as IPPO (D4): the actor is the shared
:class:`ZoneActorCritic` over own observations with masked logits, so the
IPPO-vs-MAPPO comparison isolates exactly one change — the value estimate
comes from :class:`CentralizedCritic` over the global state instead of the
decentralized own-observation head. PPO-clip, GAE, and the AEC delayed-close
buffer are shared with IPPO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from marl_fleet_rebalancing.agents.ippo import (
    RolloutBuffer,
    _explained_variance,
    _normalize_advantages,
    _real_row_terms,
)
from marl_fleet_rebalancing.networks.encoder import (
    CentralizedCritic,
    build_zone_policy,
)


@dataclass
class MAPPOConfig:
    """MAPPO hyperparameters. Same backbone as IPPO; tuned on
    training/validation demand only."""

    # --- Network ---
    hidden_dim: int = 256
    arch: str = "mlp"              # "mlp" or "attention" (D33); actor only

    # --- Rollout / optimization ---
    rollout_episodes: int = 6      # episodes collected per PPO update
    epochs: int = 4                # passes over the rollout per update
    minibatches: int = 4
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- Advantage estimation ---
    gamma: float = 0.99
    # D46: see the D46 note in ippo.py. Identical in both arms by design --
    # the IPPO-vs-MAPPO comparison must differ ONLY in the critic.
    gae_lambda: float = 0.99

    # --- Optimizer ---
    learning_rate: float = 3e-4

    def to_dict(self) -> dict:
        return asdict(self)


class MAPPO:
    """MAPPO learner: shared actor + centralized state critic."""

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        action_dim: int,
        config: MAPPOConfig,
        device: torch.device | None = None,
        num_zones: int | None = None,
        max_pending: int | None = None,
        hist_len: int | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.actor = build_zone_policy(
            config.arch, observation_dim, action_dim, config.hidden_dim,
            num_zones=num_zones, max_pending=max_pending, hist_len=hist_len,
        ).to(self.device)
        self.critic = CentralizedCritic(
            state_dim, config.hidden_dim,
            num_agents=num_zones,
        ).to(self.device)
        self.num_zones = num_zones
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
        )

    def act(
        self,
        obs_batch: np.ndarray,
        mask_batch: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions for a batch of observations (numpy in, numpy out).

        Returns (actions, logprobs, decentralized values); the rollout loop
        discards the third element and takes values from :meth:`value` over
        the centralized state instead.
        """
        obs_t = torch.as_tensor(obs_batch, device=self.device)
        mask_t = torch.as_tensor(mask_batch, device=self.device)
        action, logprob, value = self.actor.act(
            obs_t, mask_t, deterministic=deterministic,
        )
        return (
            action.cpu().numpy(),
            logprob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def value(self, state_batch: np.ndarray,
             agent_ids: np.ndarray | None = None) -> np.ndarray:
        """Centralized value estimates for a batch of global states.

        ``agent_ids`` identifies WHICH zone each state row is scoring. Required:
        the global state is shared by every agent within a turn, so without an
        identity the critic cannot separate the 16 per-zone return functions.
        """
        state_t = torch.as_tensor(state_batch, device=self.device)
        ids_t = (
            torch.as_tensor(agent_ids, dtype=torch.long, device=self.device)
            if agent_ids is not None else None
        )
        with torch.no_grad():
            value = self.critic.forward(state_t, ids_t)
        return value.cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """PPO-clip update over one filled buffer. Returns loss metrics."""
        assert buffer.states is not None, (
            "MAPPO requires centralized states in the buffer "
            "(RolloutBuffer with state_dim)."
        )
        cfg = self.config
        advantages, returns = buffer.compute_gae(cfg.gamma, cfg.gae_lambda)
        flat = buffer.flatten()
        # CPU-resident: only minibatches move to device (see IPPO.update).
        b_obs = torch.as_tensor(flat["obs"])
        b_mask = torch.as_tensor(flat["action_mask"])
        b_actions = torch.as_tensor(flat["actions"])
        b_logprobs = torch.as_tensor(flat["logprobs"])
        b_states = torch.as_tensor(flat["states"])
        b_advantages = torch.as_tensor(
            advantages.ravel()[buffer.filled.ravel()],
        )
        b_returns = torch.as_tensor(
            returns.ravel()[buffer.filled.ravel()],
        )
        # D39: which zone each stored row belongs to. flat["agent_ids"] is
        # raveled in the same slot-major order as obs/actions.
        b_agent_ids = (
            torch.as_tensor(flat["agent_ids"], dtype=torch.long)
            if "agent_ids" in flat else None
        )
        # Advantage normalization (standard PPO practice).
        # A "real" row has >1 legal action; forced rows carry no policy signal.
        b_real = b_mask.sum(-1) > 1
        b_advantages = _normalize_advantages(b_advantages, b_real)

        num_samples = b_obs.shape[0]
        batch_size = max(num_samples // cfg.minibatches, 1)
        metrics = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "clip_frac": 0.0,
        }
        update_steps = 0
        for _epoch in range(cfg.epochs):
            perm = torch.randperm(num_samples)
            for start in range(0, num_samples, batch_size):
                idx = perm[start : start + batch_size]
                dev = self.device
                logits, _ = self.actor.forward(b_obs[idx].to(dev), b_mask[idx].to(dev))
                values = self.critic.forward(
                    b_states[idx].to(dev),
                    None if b_agent_ids is None
                    else b_agent_ids[idx].to(dev),
                )
                dist = torch.distributions.Categorical(logits=logits)
                policy_loss, entropy, clip_frac = _real_row_terms(
                    dist, b_actions[idx].to(dev), b_logprobs[idx].to(dev),
                    b_advantages[idx].to(dev), b_real[idx].to(dev),
                    cfg.clip_coef,
                )

                value_loss = nn.functional.mse_loss(
                    values, b_returns[idx].to(dev),
                )

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                # Separate clips: the critic's grad norm dwarfs the actor's.
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), cfg.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["clip_frac"] += clip_frac
                update_steps += 1

        for key in metrics:
            metrics[key] /= max(update_steps, 1)
        metrics["real_frac"] = b_real.float().mean().item()
        filled = buffer.filled
        metrics["explained_var"] = _explained_variance(
            buffer.values[filled], returns[filled],
        )
        return metrics
