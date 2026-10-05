import unittest

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC


class TestRelocatorShortageCredit(unittest.TestCase):
    def _credit(self, idle_at_dest: int, waiting: int, requires_shortage: bool) -> float:
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10, team_share=0.0)
        cfg.relocator_requires_shortage = requires_shortage
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        env._zone_idle_counts[:] = 0
        env._zone_idle_counts[2] = idle_at_dest
        env._pending_per_zone[:] = 0.0
        env._pending_per_zone[2] = waiting / max(env._num_drivers, 1)
        env._driver_relocator[7] = 1
        rewards = env._compute_zone_rewards({}, dispatched_this_turn=[(99, 7, 2)])
        return float(rewards[1])

    def test_flat_when_shortage_rule_off(self):
        self.assertGreater(self._credit(idle_at_dest=1, waiting=0, requires_shortage=False), 0.0)

    def test_paid_only_when_destination_short(self):
        self.assertGreater(self._credit(idle_at_dest=0, waiting=3, requires_shortage=True), 0.0)
        self.assertEqual(self._credit(idle_at_dest=1, waiting=3, requires_shortage=True), 0.0)
        self.assertEqual(self._credit(idle_at_dest=0, waiting=0, requires_shortage=True), 0.0)


if __name__ == "__main__":
    unittest.main()
