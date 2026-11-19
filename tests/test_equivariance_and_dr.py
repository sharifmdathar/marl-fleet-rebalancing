"""D100: exact spatial equivariance + spatial domain randomization.

Two structural properties are pinned here rather than learned behaviour:

1. ``ZoneAttentionEncoder`` must be exactly permutation-equivariant over zones.
   No learned per-zone id may remain, so permuting every per-zone channel
   (including the real centroid buffer) must leave stay/dispatch logits and the
   value unchanged while permuting the relocation logits identically.
2. ``domain_randomize=True`` must produce valid episodes and keep the reward
   invariant (zone rewards still sum to the team objective), while validation
   always runs with randomization disabled.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from marl_fleet_rebalancing.config import BandraConfig, NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import build_zone_policy

Z = 6
P = 5
H = 3
HIDDEN = 32


def _make_net():
    obs_dim = 7 + Z * (5 + H) + P * 8 + Z  # inbound enabled
    net = build_zone_policy(
        "attention", obs_dim, 1 + P + Z, HIDDEN,
        num_zones=Z, max_pending=P, hist_len=H,
    )
    net.eval()
    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        xy = torch.linspace(0.0, 1.0, Z * 2).reshape(Z, 2)
        net.zone_xy.copy_(xy)
        for name, param in net.named_parameters():
            if param.dim() > 0:
                param.copy_(torch.randn(param.shape, generator=rng) * 0.1)
    return net, obs_dim


def _obs_vector():
    obs = torch.zeros(1, 7 + Z * (5 + H) + P * 8 + Z)
    o = 0

    def take(n):
        nonlocal o
        sl = slice(o, o + n)
        o += n
        return sl

    obs[0, take(2)] = torch.tensor([0.4, 0.6])          # acting zone location
    obs[0, take(4)] = torch.tensor([0.3, 0.5, 0.2, 0.1])
    obs[0, take(Z * 4)] = torch.rand(Z * 4)
    take(P * 6)                                         # pending orders unused
    take(P)                                             # pending mask
    take(P)                                             # pending passengers
    obs[0, take(Z)] = (torch.arange(Z) % 2).float()     # neighbor mask
    obs[0, take(Z * H)] = torch.rand(Z * H)             # demand history
    take(1)                                             # time_frac
    obs[0, take(Z)] = torch.rand(Z)                     # inbound
    return obs


def _permute(obs, xy, perm):
    obs = obs.clone()
    base = 0
    # zone_location deliberately NOT permuted: the acting agent keeps its role.
    base += 2 + 4
    all_z = obs[:, base:base + Z * 4].view(1, Z, 4)
    obs[:, base:base + Z * 4] = all_z[:, perm].reshape(1, -1)
    base += Z * 4 + P * 6 + P + P
    nbr = obs[:, base:base + Z]
    obs[:, base:base + Z] = nbr[:, perm]
    base += Z
    hist = obs[:, base:base + Z * H].view(1, Z, H)
    obs[:, base:base + Z * H] = hist[:, perm].reshape(1, -1)
    base += Z * H + 1
    inbound = obs[:, base:base + Z]
    obs[:, base:base + Z] = inbound[:, perm]
    return obs, xy[perm].clone()


class TestZoneAttentionEquivariance(unittest.TestCase):
    def test_no_memorised_zone_ids(self):
        net, _ = _make_net()
        self.assertFalse(hasattr(net, "zone_idx"))
        self.assertFalse(hasattr(net, "zone_centroid"))

    def test_stay_dispatch_and_value_are_permutation_invariant(self):
        net, _ = _make_net()
        obs = _obs_vector()
        perm = torch.tensor([3, 0, 5, 2, 1, 4])
        with torch.no_grad():
            logits_a, value_a = net.forward(obs)
            obs_p, xy_p = _permute(obs, net.zone_xy.clone(), perm)
            net.zone_xy.copy_(xy_p)
            logits_b, value_b = net.forward(obs_p)
            net.zone_xy.copy_(xy_p[torch.argsort(perm)])
        self.assertTrue(
            torch.allclose(logits_a[:, :1 + P], logits_b[:, :1 + P], atol=1e-5),
            msg="stay/dispatch logits changed under a pure zone permutation",
        )
        self.assertTrue(
            torch.allclose(value_a, value_b, atol=1e-5),
            msg="value changed under a pure zone permutation",
        )

    def test_destination_logits_permute_identically(self):
        net, _ = _make_net()
        obs = _obs_vector()
        perm = torch.tensor([3, 0, 5, 2, 1, 4])
        with torch.no_grad():
            logits_a, _ = net.forward(obs)
            obs_p, xy_p = _permute(obs, net.zone_xy.clone(), perm)
            net.zone_xy.copy_(xy_p)
            logits_b, _ = net.forward(obs_p)
            net.zone_xy.copy_(xy_p[torch.argsort(perm)])
        dest_a = logits_a[:, 1 + P:]
        dest_b = logits_b[:, 1 + P:]
        self.assertTrue(
            torch.allclose(dest_a[:, perm], dest_b, atol=1e-5),
            msg="relocation logits do not follow the same zone permutation",
        )
        self.assertGreater(
            float(dest_a.max() - dest_a.min()), 1e-4,
            msg="destination logits are constant, so equivariance is vacuous",
        )


def _episode_reward(overrides: dict) -> tuple[float, float]:
    cfg = BandraConfig(
        horizon=10.0, num_drivers=10, num_orders=30, max_pending=12,
        domain_randomize=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    env = RideGymAEC(config=cfg, seed=11)
    env.reset(seed=11)
    total = [0.0]
    inner = env._compute_zone_rewards

    def recording(*args, **kwargs):
        out = inner(*args, **kwargs)
        total[0] += float(np.sum(out))
        return out

    env._compute_zone_rewards = recording
    rng = np.random.default_rng(0)
    while env.agents:
        legal = np.flatnonzero(env.action_mask(env.agent_selection))
        env.step(int(rng.choice(legal)))
    in_flight = sum(paid for _zone, paid in env._order_relocator_paid.values())
    return total[0], in_flight


class TestSpatialDomainRandomization(unittest.TestCase):
    def test_dr_episode_is_valid_and_reward_still_sums_to_objective(self):
        # NYC path: the transform runs on the drawn base window.
        cfg = NYCConfig(domain_randomize=True, nyc_window_limit=60)
        env = RideGymAEC(config=cfg, seed=7)
        for seed in (7, 8, 9):
            env.reset(seed=seed)
            orders = list(env.env._all_orders)
            self.assertGreater(len(orders), 0)
            for order in orders:
                for point in (order.origin, order.destination):
                    self.assertTrue(np.all(np.isfinite(point)))
                    n = (np.asarray(point[::-1], dtype=float)
                         - np.asarray(env._geo_min[::-1], dtype=float)) \
                        / np.asarray(env._geo_range[::-1], dtype=float)
                    self.assertTrue(np.all(n >= -1e-6) and np.all(n <= 1 + 1e-6))
            # A full no-op episode must terminate cleanly.
            while env.agents:
                env.step(0)
            self.assertGreaterEqual(float(env.kpi_summary()["total"]), 0.0)
        env.close()

        # Same invariant as tests/test_reward_sums_to_objective.py, but with
        # randomization on: only the relocator split can move the total, and
        # rides still in flight at the horizon are the only outstanding gap.
        objective, _ = _episode_reward({"relocator_share": 0.0})
        split, in_flight = _episode_reward(
            {"relocator_share": 0.65, "relocator_share_is_split": True},
        )
        self.assertAlmostEqual(split - in_flight, objective, places=3)

    def test_validation_env_forces_domain_randomize_off(self):
        from marl_fleet_rebalancing.agents.ippo import IPPOConfig
        from marl_fleet_rebalancing.experiments.train import (
            TrainConfig, _make_validation_env,
        )

        cfg = BandraConfig(
            horizon=1.0, num_drivers=4, num_orders=4, max_pending=4,
            domain_randomize=True,
        )
        train_cfg = TrainConfig(
            env_config=cfg, algo_config=IPPOConfig(),
            eval_interval=1, eval_episodes=1, eval_seed=9000,
        )
        val_env = _make_validation_env(train_cfg, seed=9000)
        try:
            self.assertFalse(val_env._cfg.domain_randomize)
        finally:
            val_env.close()


if __name__ == "__main__":
    unittest.main()
