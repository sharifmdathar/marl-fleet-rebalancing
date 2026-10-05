import unittest

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC


class TestChainCredit(unittest.TestCase):
    def _rewards(self, chain_share: bool, chain: list[int], dispatcher: int = 3):
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10, team_share=0.0, gamma=1.0)
        cfg.relocator_chain_share = chain_share
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        env._driver_relocator[7] = chain[-1]
        env._driver_reloc_chain[7] = list(chain)
        return env, env._compute_zone_rewards({}, dispatched_this_turn=[(99, 7, dispatcher)])

    def test_payment_is_shared_between_the_zones_that_moved_the_car(self):
        _, r = self._rewards(True, [1, 2])
        self.assertAlmostEqual(float(r[1]), 0.325, places=5)
        self.assertAlmostEqual(float(r[2]), 0.325, places=5)

    def test_last_mover_takes_all_when_off(self):
        _, r = self._rewards(False, [1, 2])
        self.assertAlmostEqual(float(r[1]), 0.0, places=5)
        self.assertAlmostEqual(float(r[2]), 0.65, places=5)

    def test_dispatching_zone_is_not_paid_and_repeat_moves_count_once(self):
        _, r = self._rewards(True, [1, 3, 1, 2])
        self.assertAlmostEqual(float(r[1]), 0.325, places=5)
        self.assertAlmostEqual(float(r[2]), 0.325, places=5)
        self.assertAlmostEqual(float(r[3]), 0.0, places=5)

    def test_cancel_takes_back_each_share_and_records_are_cleared(self):
        env, _ = self._rewards(True, [1, 2])
        self.assertNotIn(7, env._driver_reloc_chain)
        self.assertEqual(sorted(z for z, _ in env._order_relocator_shares[99]), [1, 2])


if __name__ == "__main__":
    unittest.main()
