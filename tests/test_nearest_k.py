import unittest

import numpy as np

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC


class TestNearestK(unittest.TestCase):
    def test_only_the_nearest_legal_targets_are_offered(self):
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10)
        env = RideGymAEC(config=cfg, seed=1)
        env.reset(seed=1)
        env._zone_idle_counts[:] = 1
        everything = env._legal_relocate_targets(0)
        cfg.relocation_nearest_k = 2
        nearest = env._legal_relocate_targets(0)
        xy = np.asarray(env._zone_centroids_norm)
        dist = lambda j: float(np.linalg.norm(xy[j] - xy[0]))
        self.assertEqual(len(nearest), 2)
        self.assertLessEqual(max(map(dist, nearest)),
                             min(dist(j) for j in everything if j not in nearest))
        mask = env.action_mask("zone_0")
        self.assertEqual(int(mask[1 + cfg.max_pending:].sum()), 2)


if __name__ == "__main__":
    unittest.main()
