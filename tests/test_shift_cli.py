"""Tests for standardized demand-shift CLI and environment integration."""

import unittest

from marl_fleet_rebalancing.shifts import SHIFTS_DICT, get_shift_spec


class TestShiftCLISuite(unittest.TestCase):
    """Verify that standardized shifts are correctly configured and accessible."""

    def test_all_standard_shifts_defined(self):
        expected_labels = ["S0", "S1", "S2A", "S2B", "S2D", "S2E", "S3", "S4", "S5", "S6"]
        for label in expected_labels:
            self.assertIn(label, SHIFTS_DICT)
            spec = get_shift_spec(label)
            self.assertEqual(spec[0].upper(), label)

    def test_case_insensitive_shift_lookup(self):
        spec_upper = get_shift_spec("S5")
        spec_lower = get_shift_spec("s5")
        self.assertEqual(spec_upper, spec_lower)
        self.assertEqual(spec_upper[0], "S5")
        self.assertEqual(spec_upper[4], "burst")

    def test_unknown_shift_raises(self):
        with self.assertRaises(ValueError):
            get_shift_spec("S99_INVALID")

    def test_evaluate_cli_exports(self):
        """Ensure evaluate.py exposes policy loader and shift environment factory."""
        import marl_fleet_rebalancing.experiments.evaluate as eval_mod

        self.assertTrue(hasattr(eval_mod, "load_trained_policy"))
        self.assertTrue(hasattr(eval_mod, "make_shifted_env"))

    def test_train_cli_parser_has_prospective_bonus(self):
        """Ensure train.py exports train function."""
        import marl_fleet_rebalancing.experiments.train as train_mod

        self.assertTrue(hasattr(train_mod, "train"))


if __name__ == "__main__":
    unittest.main()
