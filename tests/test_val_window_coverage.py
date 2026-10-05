import unittest
from pathlib import Path

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


@unittest.skipUnless(Path(NYCConfig().nyc_splits_dir).exists(), "NYC splits not built")
class TestValWindowCoverage(unittest.TestCase):
    def test_seeded_resets_walk_every_held_out_window(self):
        cfg = NYCConfig()
        cfg.nyc_split = "val"
        env = RideGymAEC(cfg)
        gen = env.env.order_generator
        while not hasattr(gen, "order_paths"):
            gen = gen.base_generator
        seen = []
        for seed in range(9000, 9000 + len(gen.order_paths)):
            env.reset(seed=seed)
            seen.append(gen.last_window_index)
        self.assertEqual(sorted(seen), list(range(len(gen.order_paths))))

    def test_same_seed_same_window(self):
        cfg = NYCConfig()
        cfg.nyc_split = "val"
        env = RideGymAEC(cfg)
        gen = env.env.order_generator
        while not hasattr(gen, "order_paths"):
            gen = gen.base_generator
        env.reset(seed=9005)
        first = gen.last_window_index
        env.reset(seed=9005)
        self.assertEqual(first, gen.last_window_index)


if __name__ == "__main__":
    unittest.main()
