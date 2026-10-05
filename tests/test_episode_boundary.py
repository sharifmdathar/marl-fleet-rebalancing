import unittest

import numpy as np

from marl_fleet_rebalancing.agents.ippo import IPPO, IPPOConfig, RolloutBuffer
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC
from marl_fleet_rebalancing.experiments.train import collect_rollout, _episode_turns
from marl_fleet_rebalancing.networks.encoder import obs_dim


class TestEpisodeBoundary(unittest.TestCase):
    def test_every_slot_closes_with_terminal_done_and_no_junk_rows(self):
        cfg = BandraConfig(horizon=3.0, num_drivers=6, num_orders=12, max_pending=10)
        env = RideGymAEC(config=cfg, seed=3)
        env.reset(seed=3)
        od = obs_dim(env.observation_space(env.possible_agents[0]))
        learner = IPPO(observation_dim=od, action_dim=env._action_dim,
                       config=IPPOConfig(hidden_dim=32))
        turns_per_ep = _episode_turns(cfg)
        total = 3 * turns_per_ep
        buf = RolloutBuffer(num_agents=env._num_zones, num_turns=total,
                            obs_dim=od, action_dim=env._action_dim)
        collect_rollout(env, learner, buf, total)

        self.assertTrue(buf.filled.all(), "every row must be a real transition")
        self.assertTrue(np.all(buf.dones[:, -1] == 1.0),
                        "each slot's last transition must close with done=1")


if __name__ == "__main__":
    unittest.main()
