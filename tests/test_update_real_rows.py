"""PPO update ignores forced (single-legal-action) rows in policy/entropy terms."""

import unittest

import numpy as np
import torch

from marl_fleet_rebalancing.agents.ippo import (
    IPPO,
    IPPOConfig,
    RolloutBuffer,
    _normalize_advantages,
    _real_row_terms,
)
from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC
from marl_fleet_rebalancing.experiments.train import _episode_turns, collect_rollout
from marl_fleet_rebalancing.networks.encoder import obs_dim


class TestUpdateRealRows(unittest.TestCase):
    def _run(self, mappo: bool) -> dict:
        env = RideGymAEC(
            config=BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10),
            seed=42,
        )
        env.reset(seed=42)
        o_dim = obs_dim(env.observation_space(env.possible_agents[0]))
        a_dim = env._action_dim
        shape = {
            "num_zones": env._num_zones, "max_pending": env._cfg.max_pending,
            "hist_len": env._cfg.demand_history_len,
        }
        s_dim = len(env.state()) + o_dim if mappo else None
        if mappo:
            learner = MAPPO(
                o_dim, s_dim, a_dim,
                MAPPOConfig(hidden_dim=32, rollout_episodes=2, minibatches=2), **shape,
            )
        else:
            learner = IPPO(
                o_dim, a_dim,
                IPPOConfig(hidden_dim=32, rollout_episodes=2, minibatches=2), **shape,
            )
        turns = 2 * _episode_turns(env._cfg)
        buf = RolloutBuffer(
            num_agents=env._num_zones, num_turns=turns,
            obs_dim=o_dim, action_dim=a_dim, state_dim=s_dim,
        )
        collect_rollout(env, learner, buf, turns, use_state=mappo)
        return learner.update(buf)

    def test_updates_finite_with_real_frac(self):
        for mappo in (False, True):
            m = self._run(mappo)
            for k, v in m.items():
                self.assertTrue(np.isfinite(v), f"{k}={v} (mappo={mappo})")
            self.assertTrue(0.0 < m["real_frac"] < 1.0, m["real_frac"])

    def test_forced_rows_do_not_dilute(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        actions = torch.tensor([0, 1, 2, 0])
        old = torch.log_softmax(logits + 0.3 * torch.randn(4, 3), -1).gather(1, actions[:, None])[:, 0]
        adv = torch.randn(4)
        real = torch.ones(4, dtype=torch.bool)
        base = _real_row_terms(
            torch.distributions.Categorical(logits=logits), actions, old, adv, real, 0.2,
        )
        # Append 6 forced rows: one legal action -> logprob 0, entropy 0.
        forced = torch.full((6, 3), -1e9)
        forced[:, 0] = 0.0
        dist = torch.distributions.Categorical(logits=torch.cat([logits, forced]))
        ext = _real_row_terms(
            dist, torch.cat([actions, torch.zeros(6, dtype=torch.long)]),
            torch.cat([old, torch.zeros(6)]), torch.cat([adv, torch.randn(6)]),
            torch.cat([real, torch.zeros(6, dtype=torch.bool)]), 0.2,
        )
        for a, b in zip(base, ext):
            self.assertAlmostEqual(float(a), float(b), places=6)

    def test_advantage_norm_uses_real_rows_and_fallback(self):
        adv = torch.tensor([1.0, 3.0, 100.0, -100.0])
        real = torch.tensor([True, True, False, False])
        out = _normalize_advantages(adv, real)
        self.assertAlmostEqual(out[:2].mean().item(), 0.0, places=5)
        one = _normalize_advantages(adv, torch.tensor([True, False, False, False]))
        self.assertAlmostEqual(one.mean().item(), 0.0, places=5)

    def test_no_real_rows_gives_zero_terms(self):
        d = torch.distributions.Categorical(logits=torch.zeros(2, 3))
        p, e, c = _real_row_terms(
            d, torch.zeros(2, dtype=torch.long), torch.zeros(2), torch.ones(2),
            torch.zeros(2, dtype=torch.bool), 0.2,
        )
        self.assertEqual((p.item(), e.item(), c), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
