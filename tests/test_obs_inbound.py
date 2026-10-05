import unittest

import numpy as np
import torch

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import build_zone_policy, obs_dim, obs_to_vector


class TestObsInbound(unittest.TestCase):
    def _env(self, on: bool) -> RideGymAEC:
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10)
        cfg.obs_inbound = on
        env = RideGymAEC(config=cfg, seed=3)
        env.reset(seed=3)
        return env

    def test_counts_cars_in_flight_and_claimed_this_turn(self):
        env = self._env(True)
        env._driver_relocation_target[0] = 2
        env._driver_relocation_target[1] = 2
        env._reloc_inbound_this_turn[4] = 1
        obs = env.observe(env.agent_selection)
        expected = np.zeros(env._num_zones, dtype=np.float32)
        expected[2], expected[4] = 2 / 8, 1 / 8
        np.testing.assert_allclose(obs["inbound"], expected)

    def test_off_by_default_keeps_the_old_layout(self):
        off, on = self._env(False), self._env(True)
        a = off.possible_agents[0]
        self.assertNotIn("inbound", off.observe(a))
        d_off = obs_dim(off.observation_space(a))
        d_on = obs_dim(on.observation_space(a))
        self.assertEqual(d_on, d_off + on._num_zones)
        self.assertEqual(len(obs_to_vector(on.observe(a))), d_on)

    def test_policy_reads_the_inbound_field(self):
        env = self._env(True)
        a = env.agent_selection
        net = build_zone_policy(
            "attention", obs_dim(env.observation_space(a)), env._action_dim, 16,
            num_zones=env._num_zones, max_pending=env._cfg.max_pending,
            hist_len=env._cfg.demand_history_len)
        self.assertTrue(net._inbound)
        with torch.no_grad():
            net.dest_bias.weight.zero_()
            net.dest_bias.weight[0, -1] = -4.0
        env._driver_relocation_target[0] = 3
        with_car = torch.as_tensor(obs_to_vector(env.observe(a))[None])
        env._driver_relocation_target.clear()
        without = torch.as_tensor(obs_to_vector(env.observe(a))[None])
        base = 1 + env._cfg.max_pending
        delta = (net.forward(with_car)[0] - net.forward(without)[0])[0, base:]
        self.assertLess(delta[3].item(), -0.1)
        self.assertAlmostEqual(delta[5].item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
