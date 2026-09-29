import unittest
import numpy as np
from pettingzoo.test import api_test

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


class TestEnvAndCredit(unittest.TestCase):
    def test_pettingzoo_aec_api(self):
        cfg = BandraConfig(horizon=5.0, num_drivers=6, num_orders=10, max_pending=10)
        env = RideGymAEC(config=cfg, seed=42)
        api_test(env, num_cycles=10)

    def test_mask_decode_consistency(self):
        cfg = BandraConfig(horizon=5.0, num_drivers=10, num_orders=20, max_pending=10)
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        for agent in env.agents:
            mask = env.action_mask(agent)
            zone_idx = env._zone_int_id(agent)
            for action in range(len(mask)):
                decoded = env._decode_action(action, zone_idx)
                if action == 0:
                    self.assertIsNone(decoded)
                elif mask[action]:
                    self.assertIsNotNone(decoded)
                    if "relocate" in decoded:
                        tgt = decoded["relocate"]
                        self.assertNotEqual(tgt, zone_idx)
                        if getattr(cfg, "neighbor_relocation_only", False):
                            neighbours = env.env.region_neighbours[zone_idx]
                            self.assertIn(tgt, neighbours)
                    elif "orders" in decoded:
                        candidates = env._zone_candidate_orders[zone_idx]
                        cand_ids = [c["order_id"] for c in candidates]
                        self.assertIn(decoded["orders"][0], cand_ids)
                else:
                    # Non-neighbor relocations must decode to None
                    if action > cfg.max_pending:
                        tgt = action - cfg.max_pending - 1
                        if getattr(cfg, "neighbor_relocation_only", False):
                            neighbours = env.env.region_neighbours[zone_idx]
                            if tgt not in neighbours or tgt == zone_idx:
                                self.assertIsNone(decoded)

    def test_determinism_across_resets(self):
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10)
        env1 = RideGymAEC(config=cfg, seed=123)
        env2 = RideGymAEC(config=cfg, seed=123)

        env1.reset(seed=123)
        env2.reset(seed=123)

        for _ in range(5):
            if not env1.agents or not env2.agents:
                break
            a1 = env1.agent_selection
            a2 = env2.agent_selection
            self.assertEqual(a1, a2)

            o1 = obs_to_vector(env1.observe(a1))
            o2 = obs_to_vector(env2.observe(a2))
            np.testing.assert_allclose(o1, o2)

            m1 = env1.action_mask(a1)
            m2 = env2.action_mask(a2)
            np.testing.assert_array_equal(m1, m2)

            valid_actions = np.where(m1)[0]
            act = int(valid_actions[0]) if len(valid_actions) > 0 else 0

            env1.step(act)
            env2.step(act)

    def test_credit_attribution_and_obs_keys(self):
        cfg = BandraConfig(horizon=5.0, num_drivers=8, num_orders=15, max_pending=10, team_share=0.0)
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        agent = env.agent_selection
        obs = env.observe(agent)

        self.assertIn("neighbor_mask", obs)
        self.assertIn("demand_hist", obs)
        self.assertEqual(obs["neighbor_mask"].shape, (env._num_zones,))
        self.assertEqual(obs["demand_hist"].shape, (env._num_zones, cfg.demand_history_len))

        # Check empty deadhead attribution: commander pays, receiver pays 0
        events = {
            0: {"is_empty_move": True, "distance_moved": 1000.0}
        }
        env._driver_commander[0] = 1
        env._driver_relocation_target[0] = 2

        rewards = env._compute_zone_rewards(events, dispatched_this_turn=[])
        # Commander zone 1 must receive negative penalty
        self.assertLess(rewards[1], 0.0)
        # Receiver zone 2 must not be penalized
        self.assertEqual(rewards[2], 0.0)


if __name__ == "__main__":
    unittest.main()
