import unittest

import numpy as np

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC


class TestTargetingBonus(unittest.TestCase):
    def test_bonus_scales_with_destination_unmet_demand(self):
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10, team_share=0.0)
        cfg.targeting_bonus = 0.2
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        zone = 0
        env._zone_idle_counts[:] = 1
        legal = env._legal_relocate_targets(zone)
        self.assertGreaterEqual(len(legal), 2)
        best, other = legal[0], legal[1]
        env._pending_per_zone[:] = 0.0
        env._pending_per_zone[best] = 2.0 / max(env._num_drivers, 1)
        env._pending_per_zone[other] = 1.0 / max(env._num_drivers, 1)
        env._decision_bonus = np.zeros(env._num_zones, dtype=np.float32)
        env._decision_bonus[zone] = 0.0
        cfg_bonus = cfg.targeting_bonus
        self.assertAlmostEqual(cfg_bonus, 0.2)
        env._decision_bonus[zone] += cfg_bonus * 1.0
        env._decision_bonus[zone] += cfg_bonus * 0.5
        rewards = env._compute_zone_rewards({}, dispatched_this_turn=[])
        self.assertAlmostEqual(float(rewards[zone]), 0.2 * 1.5, places=5)
        self.assertFalse(np.any(env._decision_bonus))


if __name__ == "__main__":
    unittest.main()
