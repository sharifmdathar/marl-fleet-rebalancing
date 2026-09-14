"""IPPO: Independent PPO for the zone-controller fleet-rebalancing env.

All zone agents share one actor-critic network (parameter sharing over
homogeneous agents) but learn independently: the critic is decentralized
(own observation only) and the reward is the shared global task reward.
This is the IPPO baseline of the study (D4) — MAPPO replaces the critic's
input with the centralized state (Phase 2 step 2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from marl_fleet_rebalancing.networks.encoder import ZoneActorCritic


@dataclass
class IPPOConfig:
    """IPPO hyperparameters. Tuned on training/validation demand only."""

    # --- Network ---
    hidden_dim: int = 256

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
    gae_lambda: float = 0.95

    # --- Optimizer ---
    learning_rate: float = 3e-4

    def to_dict(self) -> dict:
        return asdict(self)


class RolloutBuffer:
    """Fixed-size buffer of per-agent transitions for one PPO update.

    Transition (agent_slot, t) = (obs, action, logp, value, reward, done,
    action_mask). Rewards/dones are filled at the agent's NEXT selection
    (standard AEC delayed-close); GAE runs per agent slot.

    MAPPO mode (``state_dim`` given): each transition additionally stores
    the centralized global state, consumed by the centralized critic.
    IPPO paths (``state_dim=None``) are untouched.
    """

    def __init__(
        self,
        num_agents: int,
        num_turns: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int | None = None,
    ) -> None:
        self.num_agents = num_agents
        self.num_turns = num_turns
        self.capacity = num_agents * num_turns

        self.obs = np.zeros((num_agents, num_turns, obs_dim), dtype=np.float32)
        self.action_mask = np.zeros(
            (num_agents, num_turns, action_dim), dtype=bool,
        )
        self.actions = np.zeros((num_agents, num_turns), dtype=np.int64)
        self.logprobs = np.zeros((num_agents, num_turns), dtype=np.float32)
        self.values = np.zeros((num_agents, num_turns), dtype=np.float32)
        self.rewards = np.zeros((num_agents, num_turns), dtype=np.float32)
        self.dones = np.zeros((num_agents, num_turns), dtype=np.float32)
        self.filled = np.zeros((num_agents, num_turns), dtype=bool)
        self.states = (
            np.zeros((num_agents, num_turns, state_dim), dtype=np.float32)
            if state_dim is not None
            else None
        )

    def store(
        self,
        slot: int,
        turn: int,
        obs_vec: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        logprob: float,
        value: float,
        state: np.ndarray | None = None,
    ) -> None:
        self.obs[slot, turn] = obs_vec
        self.action_mask[slot, turn] = action_mask
        self.actions[slot, turn] = action
        self.logprobs[slot, turn] = logprob
        self.values[slot, turn] = value
        self.filled[slot, turn] = True
        if self.states is not None and state is not None:
            self.states[slot, turn] = state

    def close(
        self, slot: int, turn: int, reward: float, done: bool,
    ) -> None:
        """Attach the (delayed) reward and done flag to a stored transition."""
        self.rewards[slot, turn] = reward
        self.dones[slot, turn] = float(done)

    def compute_gae(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-slot GAE(λ). Returns (advantages, returns) of buffer shape.

        Unfilled trailing slots (never happens when rollouts end at episode
        boundaries) are excluded from the flattened output.
        """
        advantages = np.zeros_like(self.values)
        for slot in range(self.num_agents):
            last_gae = 0.0
            for t in range(self.num_turns - 1, -1, -1):
                if not self.filled[slot, t]:
                    last_gae = 0.0
                    continue
                if self.dones[slot, t] > 0:
                    next_value = 0.0
                    next_gae = 0.0
                elif t + 1 < self.num_turns and self.filled[slot, t + 1]:
                    next_value = self.values[slot, t + 1]
                    next_gae = advantages[slot, t + 1]
                else:
                    # t is the last filled slot of an unfinished episode:
                    # bootstrap with its own value (conservative default).
                    next_value = self.values[slot, t]
                    next_gae = 0.0
                delta = (
                    self.rewards[slot, t]
                    + gamma * next_value
                    - self.values[slot, t]
                )
                last_gae = delta + gamma * gae_lambda * next_gae
                advantages[slot, t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def flatten(self) -> dict[str, np.ndarray]:
        # Fast path: rollouts always end at episode boundaries, so the
        # buffer is fully filled and reshapes are zero-copy views. The
        # masked path below copies (3+ GB at NYC scale) and can push a
        # memory-constrained box into swap during the PPO update.
        # update() only reads these tensors, so views are safe.
        if self.filled.all():
            flat: dict[str, np.ndarray] = {
                "obs": self.obs.reshape(self.capacity, -1),
                "action_mask": self.action_mask.reshape(self.capacity, -1),
                "actions": self.actions.ravel(),
                "logprobs": self.logprobs.ravel(),
            }
            if self.states is not None:
                flat["states"] = self.states.reshape(self.capacity, -1)
            return flat
        mask = self.filled.ravel()
        flat = {
            "obs": self.obs.reshape(self.capacity, -1)[mask],
            "action_mask": self.action_mask.reshape(self.capacity, -1)[mask],
            "actions": self.actions.ravel()[mask],
            "logprobs": self.logprobs.ravel()[mask],
        }
        if self.states is not None:
            flat["states"] = self.states.reshape(self.capacity, -1)[mask]
        return flat


class IPPO:
    """IPPO learner wrapping a shared :class:`ZoneActorCritic`."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: IPPOConfig,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.policy = ZoneActorCritic(
            observation_dim, action_dim, config.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate,
        )

    def act(
        self,
        obs_batch: np.ndarray,
        mask_batch: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions for a batch of observations (numpy in, numpy out)."""
        obs_t = torch.as_tensor(obs_batch, device=self.device)
        mask_t = torch.as_tensor(mask_batch, device=self.device)
        action, logprob, value = self.policy.act(
            obs_t, mask_t, deterministic=deterministic,
        )
        return (
            action.cpu().numpy(),
            logprob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def value(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_batch, device=self.device)
        with torch.no_grad():
            _logits, value = self.policy.forward(obs_t)
        return value.cpu().numpy()

    def state_dict(self) -> dict:
        return self.policy.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.policy.load_state_dict(state_dict)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """PPO-clip update over one filled buffer. Returns loss metrics."""
        cfg = self.config
        advantages, returns = buffer.compute_gae(cfg.gamma, cfg.gae_lambda)
        flat = buffer.flatten()
        # CPU-resident: a full NYC rollout (62 zones x 3600 turns x 5454 obs)
        # is ~5 GB, larger than most GPUs. Only minibatches move to device.
        b_obs = torch.as_tensor(flat["obs"])
        b_mask = torch.as_tensor(flat["action_mask"])
        b_actions = torch.as_tensor(flat["actions"])
        b_logprobs = torch.as_tensor(flat["logprobs"])
        b_advantages = torch.as_tensor(
            advantages.ravel()[buffer.filled.ravel()],
        )
        b_returns = torch.as_tensor(
            returns.ravel()[buffer.filled.ravel()],
        )
        # Advantage normalization (standard PPO practice).
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

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
                logits, values = self.policy.forward(
                    b_obs[idx].to(dev), b_mask[idx].to(dev),
                )
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(b_actions[idx].to(dev))
                entropy = dist.entropy().mean()

                ratio = torch.exp(logprobs - b_logprobs[idx].to(dev))
                adv_idx = b_advantages[idx].to(dev)
                surr1 = ratio * adv_idx
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef,
                ) * adv_idx
                policy_loss = -torch.min(surr1, surr2).mean()

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
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = (
                        ((ratio - 1.0).abs() > cfg.clip_coef)
                        .float()
                        .mean()
                        .item()
                    )
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["clip_frac"] += clip_frac
                update_steps += 1

        for key in metrics:
            metrics[key] /= max(update_steps, 1)
        filled = buffer.filled
        metrics["explained_var"] = _explained_variance(
            buffer.values[filled], returns[filled],
        )
        return metrics


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 - Var(returns - values)/Var(returns); ~0 = no signal, 1 = perfect."""
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / var_returns)
