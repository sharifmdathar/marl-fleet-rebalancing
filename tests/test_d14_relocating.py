"""Unit tests for D14: Opportunistic dispatch of relocating drivers."""

import unittest
from ride_gym.enums import DriverStatus, OrderStatus
from ride_gym import Order
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC


class TestD14RelocatingDispatch(unittest.TestCase):
    def test_d14_relocating_driver_dispatch(self):
        """Verify that empty relocating drivers can be dispatched to local orders (D14)."""
        cfg = BandraConfig(num_orders=0, allow_relocating_dispatch=True)
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        # Set driver 0 to RELOCATING status in its current region
        d0 = env.env.drivers[0]
        d0_region = env.env._current_region(d0.location)
        d0.status = DriverStatus.RELOCATING
        d0.relocation_target = env.env.relocation_points[(d0_region + 1) % env._num_zones]

        # Rebuild caches to observe this driver as relocating
        env._last_obs = env.env._build_observations()
        env._rebuild_caches()

        # Inject an order at driver 0's location
        target_agent = f"zone_{d0_region}"
        new_id = 999
        order = Order(
            order_id=new_id,
            origin=tuple(d0.location),
            destination=env.env.network.node_coord(10),
            request_time=0.0,
            status=OrderStatus.PENDING,
        )
        env.env.orders[new_id] = order
        env.env._pending_ids.append(new_id)
        env._last_obs = env.env._build_observations()
        env._current_pending = env._extract_pending(env._last_obs)
        env._rebuild_caches()

        # Verify target_agent has a valid dispatch action despite driver 0 being RELOCATING
        mask = env.action_mask(target_agent)
        # Action 1 corresponds to candidate order 0
        self.assertTrue(mask[1], "D14: zone with relocating driver should be allowed to dispatch order")

        # Step target agent with dispatch action 1, others with 0
        for _ in range(len(env.agents)):
            agent = env.agent_selection
            _obs, _r, term, trunc, _info = env.last()
            if term or trunc:
                env.step(None)
                continue
            if agent == target_agent:
                env.step(1)  # dispatch candidate 0
            else:
                env.step(0)  # idle

        # Verify driver 0 claimed the order and transitioned to TO_PICKUP or TO_DROPOFF
        self.assertIn(d0.status, (DriverStatus.TO_PICKUP, DriverStatus.TO_DROPOFF))
        self.assertIn(new_id, d0.assigned_orders)
        self.assertIsNone(d0.relocation_target)


if __name__ == "__main__":
    unittest.main()
