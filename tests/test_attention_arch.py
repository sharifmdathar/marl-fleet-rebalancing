"""Tests for the attention/GNN-style policy trunk (D33).

Verifies the drop-in replacement contract: same act()/forward() shapes as
ZoneActorCritic, no NaNs on edge cases (empty pending set), and a full
PPO update step (collect_rollout -> update) runs end to end without
crashing. This is a unit/integration test, not a training run: horizon
and iteration counts are minimal and results are not evaluated for
learning progress.
"""

import unittest

import numpy as np
import torch

from marl_fleet_rebalancing.agents.ippo import IPPO, IPPOConfig, RolloutBuffer
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC
from marl_fleet_rebalancing.experiments.train import _episode_turns, collect_rollout
from marl_fleet_rebalancing.networks.encoder import build_zone_policy, obs_dim


class TestAttentionArch(unittest.TestCase):
    def _tiny_env(self) -> RideGymAEC:
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10)
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        return env

    def test_matches_mlp_io_contract(self):
        env = self._tiny_env()
        any_agent = env.possible_agents[0]
        observation_dim = obs_dim(env.observation_space(any_agent))
        action_dim = env._action_dim

        net = build_zone_policy(
            "attention", observation_dim, action_dim, hidden_dim=32,
            num_zones=env._num_zones, max_pending=env._cfg.max_pending,
            hist_len=env._cfg.demand_history_len,
        )
        obs = env.observe(any_agent)
        from marl_fleet_rebalancing.networks.encoder import obs_to_vector

        obs_t = torch.as_tensor(obs_to_vector(obs)[None, :])
        mask_t = torch.as_tensor(env.action_mask(any_agent)[None, :])

        logits, value = net.forward(obs_t, mask_t)
        self.assertEqual(logits.shape, (1, action_dim))
        self.assertEqual(value.shape, (1,))
        self.assertTrue(torch.isfinite(value).all())

        action, _logprob, _val = net.act(obs_t, mask_t)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(bool(mask_t[0, action.item()]))

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            build_zone_policy("transformer-xl", 10, 5, 32)

    def test_ippo_update_runs_with_attention_arch(self):
        """One PPO update (collect_rollout -> update) must not crash or NaN."""
        env = self._tiny_env()
        observation_dim = obs_dim(env.observation_space(env.possible_agents[0]))
        action_dim = env._action_dim

        cfg = IPPOConfig(arch="attention", hidden_dim=32, rollout_episodes=2, minibatches=2)
        learner = IPPO(
            observation_dim=observation_dim,
            action_dim=action_dim,
            config=cfg,
            num_zones=env._num_zones,
            max_pending=env._cfg.max_pending,
            hist_len=env._cfg.demand_history_len,
        )

        total_turns = cfg.rollout_episodes * _episode_turns(env._cfg)
        buffer = RolloutBuffer(
            num_agents=env._num_zones, num_turns=total_turns,
            obs_dim=observation_dim, action_dim=action_dim,
        )
        collect_rollout(env, learner, buffer, total_turns, use_state=False)
        metrics = learner.update(buffer)

        for key in ("policy_loss", "value_loss", "entropy"):
            self.assertTrue(np.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}")


if __name__ == "__main__":
    unittest.main()
