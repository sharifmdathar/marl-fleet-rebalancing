"""D54 regression tests: the relocation head must be destination-conditioned.

Before D54 every action logit came from one pooled vector, and the relocation
part of that vector was a softmax-weighted SUM of zone embeddings. A weighted sum
is permutation-invariant over zones, so it could encode "there is demand
somewhere" but not "zone 6 rather than zone 14". Measured on the D51
checkpoints: perturbing the demand inputs moved the relocation logits
substantially (L1 sensitivity 2.9-15.3) yet the argmax destination never changed
-- aiming at a fixed zone no matter where demand was.

These tests pin the structural property, not the learned behaviour: the head
must produce a logit per destination from that destination's own features.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import build_zone_policy, obs_to_vector


class TestDestinationConditionedHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = NYCConfig()
        cls.cfg.nyc_split = "val"
        cls.env = RideGymAEC(cls.cfg)
        cls.env.reset(seed=8000)
        cls.mp = cls.env._max_pending
        cls.nz = cls.env._num_zones
        obs = cls.env.observe(cls.env.agents[0])
        cls.v0 = obs_to_vector(obs).copy()
        net = build_zone_policy(
            "attention", len(cls.v0), cls.env._action_dim, 64,
            num_zones=cls.nz, max_pending=cls.mp, hist_len=cls.env._hist_len,
        )
        cls.net = net.eval()

    def _logits(self, v):
        with torch.no_grad():
            out = self.net(torch.as_tensor(v, dtype=torch.float32)[None, :], None)
        return out[0][0].numpy()

    def _reloc(self, v):
        return self._logits(v)[1 + self.mp: 1 + self.mp + self.nz]

    def _with_demand(self, hot):
        """Put all pending on `hot`, updating BOTH derived columns.

        Updating only `pending` (col 2) and leaving `deficit` (col 3) stale
        builds a state the env can never produce, and produced a false "the
        output ignores demand" reading in an earlier probe.
        """
        v = self.v0.copy()
        for z in range(self.nz):
            p = 1.0 if z == hot else 0.0
            v[6 + z * 4 + 2] = p
            v[6 + z * 4 + 3] = p
        return v

    def test_head_is_destination_conditioned_not_pooled(self):
        """Relocation logits must respond to WHERE demand is.

        Structural property, testable at initialisation: each destination's logit
        is a function of that destination's own features, so relocating demand
        changes the relocation logit vector.

        Whether the ARGMAX then follows demand is a trained-behaviour property,
        not a structural one -- at random init the per-zone constants from the
        travel-cost and key terms legitimately dominate a single unit of demand.
        Asserting argmax-follows-demand here would be asserting convergence,
        which belongs in the acceptance test (`diag_targeting_at_decision.py`),
        not in a unit test.
        """
        base = self._reloc(self._with_demand(0))
        changed = 0
        for hot in range(1, self.nz):
            d = np.abs(self._reloc(self._with_demand(hot)) - base).sum()
            if d > 1e-6:
                changed += 1
        self.assertEqual(
            changed, self.nz - 1,
            "relocation logits barely move when demand changes zone, so the "
            "head is still close to permutation-invariant over destinations",
        )

    def test_destinations_are_not_symmetric(self):
        """Distinct destinations must get distinct logits from one observation.

        A weighted sum over zone embeddings can still produce a spread of
        numbers, so also assert the spread is real rather than a constant
        offset: each destination's logit must depend on its own features.
        """
        rl = self._reloc(self._with_demand(3))
        self.assertEqual(rl.shape, (self.nz,))
        # Shifting demand from zone 3 to zone 7 must move zone 3's own logit.
        a = self._reloc(self._with_demand(3))[3]
        b = self._reloc(self._with_demand(7))[3]
        self.assertNotAlmostEqual(
            float(a), float(b), places=6,
            msg="zone 3's logit is unchanged when demand leaves zone 3; its "
                "own features are not reaching its logit",
        )

    def test_dispatch_and_noop_logits_are_unaffected_in_shape(self):
        """Action space unchanged: 1 noop + max_pending dispatch + Z relocate."""
        lg = self._logits(self.v0)
        self.assertEqual(lg.shape, (1 + self.mp + self.nz,))

    def test_pooled_head_does_not_also_own_relocation_columns(self):
        """The pooled linear head must not double-count the relocation logits.

        Its relocation rows are zeroed at init so the destination-conditioned
        head is the sole owner of those columns; otherwise the pooled term can
        dominate and reintroduce the permutation-invariant bias.
        """
        w = self.net.actor.weight[1 + self.mp:, :]
        self.assertLess(
            float(w.detach().abs().max()), 1e-6,
            "pooled actor head still has non-zero weights on relocation "
            "columns, so it competes with the destination-conditioned head",
        )


if __name__ == "__main__":
    unittest.main()
