"""Regression tests for validation-based checkpoint selection (train.py).

Locks the protocol bugs found 2026-09-17 in the v18best best-checkpoint runs:

1. Selection evaluated on the TRAINING split (same windows the policy is
   learning on) instead of the held-out validation split.
2. Selection ran in the training env itself, so the seeded val resets
   (9000..9002) rewrote the training RNG stream — training after the first
   selection diverged from the same-seed run without eval enabled.
3. Selection settings (split, seeds, episodes) were not recorded anywhere,
   so a best checkpoint could not be audited.
"""

import json
import unittest

import numpy as np

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC


class TestValidationIsolationProtocol(unittest.TestCase):
    """The contract: a validation eval must leave the training env untouched."""

    def setUp(self):
        self.cfg = BandraConfig(horizon=1.0, num_drivers=4, num_orders=4, max_pending=4)

    def test_val_reset_changes_training_stream_on_shared_env(self):
        """Demonstrates the bug mechanics on one shared env (documents why a
        separate val env is required)."""
        a = RideGymAEC(config=self.cfg, seed=0)
        b = RideGymAEC(config=self.cfg, seed=0)
        a.reset(seed=0)
        b.reset(seed=0)
        for env in (a, b):
            for _ in range(env._num_zones):
                env.step(0)
        self.assertTrue(np.array_equal(a._agent_order, b._agent_order))

        # Simulating selection on the training env: seeded val resets.
        for s in range(9000, 9003):
            a.reset(seed=s)
        a.reset(seed=None)
        b.reset(seed=None)
        self.assertFalse(
            np.array_equal(a._agent_order, b._agent_order),
            "Precondition changed: seeded val resets no longer diverge the stream",
        )
        a.close()
        b.close()

    def test_deterministic_eval_restores_training_stream_state(self):
        """After _evaluate_deterministic on a dedicated val env, the training
        env's RNG stream must be exactly where it was before selection."""
        from marl_fleet_rebalancing.experiments.train import (
            evaluate_deterministic,
        )

        train_env = RideGymAEC(config=self.cfg, seed=0)
        train_env.reset(seed=0)
        for _ in range(train_env._num_zones):
            train_env.step(0)
        stream_before = train_env._rng.bit_generator.state.copy()

        val_env = RideGymAEC(config=self.cfg, seed=777)
        actions: list[np.ndarray] = []

        class _Scripted:
            """Deterministic stand-in: records obs, returns action 0."""

            def act(self, obs_batch, mask_batch, deterministic=False):
                actions.append(np.asarray(obs_batch).copy())
                return np.zeros(len(obs_batch), dtype=int), np.zeros(len(obs_batch)), np.zeros(len(obs_batch))

        evaluate_deterministic(
            val_env, _Scripted(), episodes=2, base_seed=9000,
        )
        stream_after = train_env._rng.bit_generator.state.copy()
        self.assertEqual(stream_before, stream_after)
        episode_turns = round(self.cfg.horizon / self.cfg.dt)
        self.assertEqual(len(actions), 2 * episode_turns)
        val_env.close()
        train_env.close()

    def test_eval_config_records_split_and_seeds(self):
        """Selection provenance must round-trip through TrainConfig.to_dict so
        best_checkpoint.json / manifest can be audited."""
        from marl_fleet_rebalancing.agents.ippo import IPPOConfig
        from marl_fleet_rebalancing.experiments.train import TrainConfig

        cfg = TrainConfig(
            env_config=self.cfg,
            algo_config=IPPOConfig(),
            eval_interval=4,
            eval_episodes=3,
            eval_seed=9000,
        )
        d = cfg.to_dict()
        self.assertEqual(d["eval_interval"], 4)
        self.assertEqual(d["eval_episodes"], 3)
        self.assertEqual(d["eval_seed"], 9000)
        self.assertIn("eval_split", d)
        # Default selection split must be the held-out validation split,
        # never the training split.
        self.assertEqual(d["eval_split"], "val")

    def test_best_checkpoint_json_records_selection_provenance(self):
        """best_checkpoint.json must carry split/seeds/episodes so a selected
        checkpoint is auditable without reading code."""
        from marl_fleet_rebalancing.experiments.train import selection_record

        rec = selection_record(
            iteration=12,
            val_service_rate=0.632,
            eval_cfg={"eval_seed": 9000, "eval_episodes": 3, "val_split": "val"},
        )
        self.assertEqual(rec["iteration"], 12)
        self.assertEqual(rec["val_service_rate"], 0.632)
        self.assertEqual(rec["val_split"], "val")
        self.assertEqual(rec["eval_seed"], 9000)
        self.assertEqual(rec["eval_episodes"], 3)
        # Round-trip through JSON (what train.py writes to disk).
        blob = json.dumps(rec)
        self.assertEqual(json.loads(blob)["val_split"], "val")


if __name__ == "__main__":
    unittest.main()
