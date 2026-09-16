"""Unit tests for non-RL baselines (GreedyZoneDispatcher and ZoneLPRebalancer)."""

import unittest
import numpy as np

from marl_fleet_rebalancing.baselines import (
    GreedyZoneDispatcher,
    LPBaselineConfig,
    ZoneLPRebalancer,
)
from marl_fleet_rebalancing.baselines.lp_baseline import solve_rebalance_lp
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import evaluate_policy


class TestBaselines(unittest.TestCase):
    def setUp(self):
        # Short horizon config for fast unit tests
        self.cfg = BandraConfig(horizon=60, num_drivers=10, max_pending=20)
        self.env = RideGymAEC(config=self.cfg, seed=123)

    def test_solve_rebalance_lp(self):
        """Test transportation LP solver directly."""
        n_zones = 4
        # Zones 0, 1 surplus (-2, -1); Zones 2, 3 deficit (+2, +1)
        imbalances = np.array([-2.0, -1.0, 2.0, 1.0])
        costs = np.array([
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 4.0, 5.0],
            [5.0, 4.0, 0.0, 1.0],
            [6.0, 5.0, 1.0, 0.0],
        ])
        transfers = solve_rebalance_lp(
            imbalances, costs, n_idle=5, deficit_penalty=10.0,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(transfers.shape, (n_zones, n_zones))
        # Self transfers must be zero
        for i in range(n_zones):
            self.assertEqual(transfers[i, i], 0)
        # Total transfers should not exceed surplus
        self.assertLessEqual(np.sum(transfers), 5)
        # Shortfalls must actually be filled when the penalty beats travel cost.
        self.assertEqual(np.sum(transfers), 3)
        self.assertEqual(np.sum(transfers.sum(axis=0)), 3)

    def test_solve_rebalance_lp_zero_penalty_is_a_no_op(self):
        """With no value on filling shortfalls, the LP must move nothing.

        Every route cost is non-negative and every constraint is an upper bound,
        so the all-zero plan is optimal — this is the failure mode that made the
        baseline indistinguishable from greedy.
        """
        imbalances = np.array([-2.0, -1.0, 2.0, 1.0])
        costs = np.array([
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 4.0, 5.0],
            [5.0, 4.0, 0.0, 1.0],
            [6.0, 5.0, 1.0, 0.0],
        ])
        transfers = solve_rebalance_lp(imbalances, costs, n_idle=5, deficit_penalty=0.0)
        self.assertEqual(np.sum(transfers), 0)

    def test_greedy_dispatcher_action_constraints(self):
        """Greedy dispatcher must never select relocation actions and must respect action_mask."""
        greedy = GreedyZoneDispatcher()
        self.env.reset(seed=42)

        for _ in range(50):
            if not self.env.agents or self.env._terminated:
                break
            agent = self.env.agent_selection
            _obs, _r, term, trunc, _info = self.env.last()
            if term or trunc:
                self.env.step(None)
                continue

            mask = self.env.action_mask(agent)
            act = greedy.select_action(agent, self.env)

            # Action must be valid
            self.assertTrue(mask[act], f"Action {act} not valid under mask for {agent}")
            # Greedy dispatcher must NEVER choose relocation actions (action > max_pending)
            self.assertLessEqual(act, self.env._max_pending)

            self.env.step(act)

    def test_lp_rebalancer_action_constraints(self):
        """LP rebalancer must respect action_mask and valid neighbor transitions."""
        lp = ZoneLPRebalancer(LPBaselineConfig(rebalance_frequency_minutes=0.5))
        self.env.reset(seed=42)

        reloc_count = 0
        for _ in range(100):
            if not self.env.agents or self.env._terminated:
                break
            agent = self.env.agent_selection
            _obs, _r, term, trunc, _info = self.env.last()
            if term or trunc:
                self.env.step(None)
                continue

            mask = self.env.action_mask(agent)
            act = lp.select_action(agent, self.env)

            self.assertTrue(mask[act], f"Action {act} not valid under mask for {agent}")
            if act > self.env._max_pending:
                reloc_count += 1

            self.env.step(act)

    def test_evaluate_policy_with_greedy_and_lp(self):
        """Verify evaluate_policy runs end-to-end for both baseline types."""
        greedy = GreedyZoneDispatcher()
        results_greedy, _ = evaluate_policy(
            policy=greedy,
            env=self.env,
            policy_type="greedy",
            num_episodes=1,
            base_seed=500,
        )
        self.assertEqual(len(results_greedy), 1)
        self.assertIn("service_rate", results_greedy[0])
        self.assertIn("total_reward", results_greedy[0])

        lp = ZoneLPRebalancer()
        results_lp, _ = evaluate_policy(
            policy=lp,
            env=self.env,
            policy_type="lp",
            num_episodes=1,
            base_seed=500,
        )
        self.assertEqual(len(results_lp), 1)
        self.assertIn("service_rate", results_lp[0])
        self.assertIn("total_reward", results_lp[0])


if __name__ == "__main__":
    unittest.main()
