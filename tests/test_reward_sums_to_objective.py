import unittest

import numpy as np

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC


def episode_reward(**overrides) -> tuple[float, float]:
    """Total zone reward, and relocator credit still owed by rides in flight at the horizon."""
    cfg = BandraConfig(horizon=15.0, num_drivers=12, num_orders=60, max_pending=20)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    env = RideGymAEC(config=cfg, seed=5)
    env.reset(seed=5)
    total = [0.0]
    inner = env._compute_zone_rewards

    def recording(*args, **kwargs):
        out = inner(*args, **kwargs)
        total[0] += float(np.sum(out))
        return out

    env._compute_zone_rewards = recording
    rng = np.random.default_rng(0)
    while env.agents:
        legal = np.flatnonzero(env.action_mask(env.agent_selection))
        env.step(int(rng.choice(legal)))
    in_flight = sum(paid for _zone, paid in env._order_relocator_paid.values())
    return total[0], in_flight


class TestRewardSumsToObjective(unittest.TestCase):
    def test_split_credit_sums_to_the_team_objective(self):
        objective, _ = episode_reward(relocator_share=0.0)
        bonus, _ = episode_reward(relocator_share=0.65, relocator_share_is_split=False)
        split, in_flight = episode_reward(relocator_share=0.65, relocator_share_is_split=True)
        self.assertGreater(bonus, objective + 1.0, "episode must exercise relocator credit")
        # The relocator is paid at dispatch and the dispatcher repays at completion,
        # so the only gap is rides that had not finished when the episode ended.
        self.assertAlmostEqual(split - in_flight, objective, places=3)
        self.assertLess(split, bonus)

    def test_chain_sharing_keeps_the_sum(self):
        objective, _ = episode_reward(relocator_share=0.0)
        shared, in_flight = episode_reward(
            relocator_share=0.65, relocator_share_is_split=True, relocator_chain_share=True)
        self.assertAlmostEqual(shared - in_flight, objective, places=3)


if __name__ == "__main__":
    unittest.main()
