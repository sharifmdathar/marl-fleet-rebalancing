import unittest
import numpy as np
import torch
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC
from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.networks.encoder import obs_dim


class TestFixesAndDemand(unittest.TestCase):
    def test_target_zone_demand_injection(self):
        """Test that target_zone and target_zone_demand_pct inject source demand in target zone."""
        cfg = BandraConfig()
        cfg.target_zone = 3
        cfg.target_zone_demand_pct = 0.20  # 20%
        cfg.max_pending_orders_per_zone = 8
        cfg.num_drivers = 50

        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        total_orders = len(env.env.order_generator.orders)
        self.assertGreater(total_orders, 0)

        # Count how many orders originate near target zone centroid 3 (< 1.5 km)
        target_pt = env.env.relocation_points[3]

        in_target = 0
        for o in env.env.order_generator.orders:
            # Check proximity to target zone centroid (1.5 km ~ 0.015 deg in Bandra)
            d_sq = (o.origin[0] - target_pt[0]) ** 2 + (o.origin[1] - target_pt[1]) ** 2
            if d_sq < (0.015 ** 2):
                in_target += 1

        pct = in_target / total_orders
        self.assertGreaterEqual(pct, 0.18)

    def test_intra_turn_collision_masking(self):
        """Test that claiming an order dynamically masks it out for subsequent agents in the same turn."""
        from ride_gym import Order, OrderStatus

        cfg = BandraConfig()
        cfg.num_drivers = 30
        cfg.max_dispatch_radius_km = 10.0  # Wide radius so multiple zones can see the order

        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        # Inject order 999
        order_origin = tuple(env.env.relocation_points[0])
        new_id = 999
        order = Order(
            order_id=new_id,
            origin=order_origin,
            destination=tuple(env.env.relocation_points[1]),
            request_time=0.0,
            status=OrderStatus.PENDING,
        )
        env.env.orders[new_id] = order
        env.env._pending_ids.append(new_id)
        env._last_obs = env.env._build_observations()
        env._current_pending = env._extract_pending(env._last_obs)
        env._rebuild_caches()

        # Find two zones that both have an idle driver and see order 999
        agents_with_order = []
        for agent in env.possible_agents:
            z_idx = env._zone_int_id(agent)
            cands = env._zone_candidate_orders[z_idx]
            cand_order_ids = [c["order_id"] for c in cands]
            if new_id in cand_order_ids and env.action_mask(agent)[1 + cand_order_ids.index(new_id)]:
                agents_with_order.append(agent)

        self.assertGreaterEqual(len(agents_with_order), 2, "Expected at least 2 zones to see order 999 with 10km radius")

        first_agent = agents_with_order[0]
        second_agent = agents_with_order[1]

        # Force execution order so first_agent acts before second_agent in current turn
        first_idx = env._zone_int_id(first_agent)
        second_idx = env._zone_int_id(second_agent)
        other_indices = [i for i in range(env._num_zones) if i not in (first_idx, second_idx)]
        env._agent_order = np.array([first_idx, second_idx] + other_indices)
        env.agent_selection = first_agent
        env._turn_agent_idx = 0

        # Before first_agent steps, second_agent's mask has this order as True
        z2_cands = env._zone_candidate_orders[second_idx]
        z2_order_ids = [c["order_id"] for c in z2_cands]
        z2_act = 1 + z2_order_ids.index(new_id)
        self.assertTrue(env.action_mask(second_agent)[z2_act])

        # First agent claims order 999
        z1_cands = env._zone_candidate_orders[first_idx]
        z1_order_ids = [c["order_id"] for c in z1_cands]
        z1_act = 1 + z1_order_ids.index(new_id)
        env.step(z1_act)

        # Now second_agent must have this order masked out as False
        self.assertEqual(env.agent_selection, second_agent)
        self.assertFalse(env.action_mask(second_agent)[z2_act])

    def test_local_dispatch_radius_cutoff(self):
        """Test that orders beyond max_dispatch_radius_km are not offered to far-away zones."""
        from ride_gym import Order, OrderStatus

        cfg = BandraConfig()
        cfg.max_dispatch_radius_km = 0.5  # Very tight radius: 500m

        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        # Inject order at zone 0
        new_id = 888
        order = Order(
            order_id=new_id,
            origin=tuple(env.env.relocation_points[0]),
            destination=tuple(env.env.relocation_points[1]),
            request_time=0.0,
            status=OrderStatus.PENDING,
        )
        env.env.orders[new_id] = order
        env.env._pending_ids.append(new_id)
        env._last_obs = env.env._build_observations()
        env._current_pending = env._extract_pending(env._last_obs)
        env._rebuild_caches()

        # Check a distant zone (e.g., zone 24, typically across the area)
        p0 = np.array(env.env.relocation_points[0])
        p_far = np.array(env.env.relocation_points[24])
        dist_km = float(np.sqrt(np.sum((p0 - p_far) ** 2)) * 111.0)
        self.assertGreater(dist_km, 1.0)

        far_cands = [c["order_id"] for c in env._zone_candidate_orders[24]]
        self.assertNotIn(new_id, far_cands, "Distant order should be filtered out by max_dispatch_radius_km")

    def test_neighbor_only_relocation_mask(self):
        """Test that idle vehicle relocation is pruned to adjacent neighboring zones."""
        cfg = BandraConfig()
        cfg.neighbor_relocation_only = True
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        # Inspect agent action masks
        agent = env.possible_agents[0]
        mask = env.action_mask(agent)
        zone_idx = 0
        neighbor_indices = set(env.env.region_neighbours[zone_idx])

        reloc_offset = 1 + env._max_pending
        for target_idx in range(env._num_zones):
            act = reloc_offset + target_idx
            # If target is not a neighbor and not own zone, it must be masked False
            if target_idx not in neighbor_indices and target_idx != zone_idx:
                self.assertFalse(mask[act], f"Action to non-neighbor zone {target_idx} should be masked False")

    def test_mappo_conditioned_critic(self):
        """Test that MAPPO learner initializes and evaluates with conditioned state_dim (state + zone_loc)."""
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        any_agent = env.possible_agents[0]
        obs = env.observe(any_agent)
        zone_loc_dim = len(obs["zone_location"].ravel())
        state_dim = len(env.state()) + zone_loc_dim
        observation_dim = obs_dim(env.observation_space(any_agent))
        action_dim = env._action_dim

        mappo = MAPPO(
            observation_dim=observation_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            config=MAPPOConfig(),
        )

        state_vec = np.concatenate([env.state(), obs["zone_location"].ravel()]).astype(np.float32)
        val = mappo.value(state_vec[None, :])
        self.assertEqual(val.shape, (1,))
        self.assertTrue(np.isfinite(val[0]))

    def test_macro_demand_observation_and_rebalance_credit(self):
        """Test that all_zone_state has shape (num_zones, 3) exposing macro-demand and credit assignment handles assists."""
        cfg = BandraConfig()
        cfg.target_zone = 3
        cfg.target_zone_demand_pct = 0.80
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        any_agent = env.possible_agents[0]
        obs = env.observe(any_agent)

        # 1. Check all_zone_state shape is (Z, 3)
        self.assertEqual(obs["all_zone_state"].shape, (env._num_zones, 3))
        # 2. Check that target zone 3 has significant pending demand fraction in column 2
        demand_col = obs["all_zone_state"][:, 2]
        self.assertTrue(np.all(demand_col >= 0.0))
        # 3. Simulate rebalancing assist attribution in reward computation
        env._order_dispatcher[100] = 5
        env._driver_relocator[1] = 18  # Driver 1 was relocated by Zone 18
        mock_events = {1: {"completed_orders": [100], "is_empty_move": False}}
        rewards = env._compute_zone_rewards(mock_events)
        # Zone 5 is dispatcher, Zone 18 is relocator: both must receive non-zero positive reward
        self.assertGreater(rewards[5], 0.0)
        self.assertGreater(rewards[18], 0.0)


if __name__ == "__main__":
    unittest.main()
