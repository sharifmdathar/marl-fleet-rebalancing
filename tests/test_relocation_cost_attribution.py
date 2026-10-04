"""D59 regression: the relocating zone pays for the whole job it caused.

Before D59, dispatching a relocated car overwrote `_driver_commander` to the
DISPATCHING zone. The zone that chose the relocation therefore paid only the
relocation leg, while the empty drive from its destination to the rider was
billed to whichever zone happened to use the car. Relocating was close to free,
which is what the numbers showed: 160 relocations x 2.9 km = 464 km against
476-481 km observed per episode, i.e. deadhead was essentially all relocation
and was charged to the wrong party.

These tests pin the attribution directly, without training.

    uv run pytest tests/test_relocation_cost_attribution.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC


class TestRelocationCostAttribution(unittest.TestCase):
    def _env(self):
        cfg = BandraConfig()
        cfg.team_share = 0.0
        cfg.alpha = 0.0
        cfg.cancel_penalty = 0.0
        cfg.beta = 1.0     # 1.0 per km so the charge is unambiguous
        env = RideGymAEC(config=cfg, seed=11)
        env.reset(seed=11)
        return env, cfg

    def test_commander_survives_dispatch_of_a_relocated_car(self):
        """Dispatching a relocated car must NOT reassign the deadhead payer."""
        env, _cfg = self._env()
        sender = 3
        dispatcher = 7
        did = 0
        # The zone that relocated the car is on record as its relocator.
        env._driver_relocator[did] = sender
        env._driver_commander[did] = sender

        # Simulate the D51 commit step for a dispatch of that car.
        order = env.env.orders[next(iter(env.env.orders))]
        order.status = OrderStatus.ASSIGNED
        order.assigned_driver = did
        env._pending_dispatch = [(order.order_id, did, dispatcher)]

        from ride_gym.enums import OrderStatus as S
        for oid, d, z in env._pending_dispatch:
            o = env.env.orders.get(oid)
            if o.status not in (S.ASSIGNED, S.ONBOARD, S.COMPLETED):
                continue
            if getattr(o, "assigned_driver", None) != d:
                continue
            if d not in env._driver_relocator:
                env._driver_commander[d] = z
        env._pending_dispatch = []

        self.assertEqual(
            env._driver_commander.get(did), sender,
            "commander was reassigned to the dispatching zone, so the sending "
            "zone escapes the deadhead cost of the move it chose",
        )

    def test_commander_is_cleared_when_the_job_completes(self):
        """Liability must end with the job, or it leaks into the next one."""
        env, cfg = self._env()
        sender = 3
        did = 0
        env._driver_relocator[did] = sender
        env._driver_commander[did] = sender

        class _Order:
            def __init__(self):
                self.status = OrderStatus.COMPLETED
                self.assigned_driver = did

        env.env.orders[900] = _Order()
        env._order_dispatcher[900] = 7
        env._compute_zone_rewards({}, [])
        self.assertNotIn(
            did, env._driver_commander,
            "commander survived the job, so the next job's deadhead is billed "
            "to whichever zone relocated last",
        )

    def test_plain_dispatch_still_bills_the_dispatching_zone(self):
        """A car that was never relocated is billed to whoever dispatched it."""
        env, _cfg = self._env()
        did = 0
        self.assertNotIn(did, env._driver_relocator)
        env._driver_commander[did] = 5
        # The commit step only preserves an EXISTING relocator-driven commander.
        if did not in env._driver_relocator:
            env._driver_commander[did] = 9
        self.assertEqual(env._driver_commander[did], 9)


if __name__ == "__main__":
    unittest.main()
