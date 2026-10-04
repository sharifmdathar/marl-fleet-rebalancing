import unittest

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC


class TestDeficitCredit(unittest.TestCase):
    def _credit(self, waiting: int, ref: float) -> float:
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10, team_share=0.0)
        cfg.relocator_deficit_reference = ref
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        env._pending_per_zone[:] = 0.0
        env._pending_per_zone[2] = waiting / max(env._num_drivers, 1)
        env._driver_relocator[7] = 1
        rewards = env._compute_zone_rewards({}, dispatched_this_turn=[(99, 7, 2)])
        return float(rewards[1])

    def test_flat_when_disabled(self):
        self.assertAlmostEqual(self._credit(1, ref=0.0), self._credit(10, ref=0.0))

    def test_scales_with_waiting_riders(self):
        self.assertLess(self._credit(1, ref=1.7), self._credit(3, ref=1.7))
        self.assertAlmostEqual(self._credit(3, ref=1.7), self._credit(10, ref=1.7))


if __name__ == "__main__":
    unittest.main()
