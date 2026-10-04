"""What is the actual mean relocation leg distance?

The break-even invariant `relocator_share > beta * km_per_relocation` has been
checked against a hardcoded 2.49 km in `scripts/diag_reloc_invariant.py`. That
number came from an older run and is now baked into the config comment as if
it were a property of the environment. If the real mean differs, the asserted
margin is wrong in whichever direction the truth lies.

Measures each leg independently: from the turn a relocation is commanded to the
turn the driver stops being `relocating`, summing the `is_empty_move` distance
reported for that driver in between.

    uv run python scripts/diag_reloc_leg_distance.py --episodes 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=3000)
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    n_drv = env._num_drivers

    leg_m: dict[int, float] = {}     # did -> km accumulated on the open leg
    status = [""] * n_drv
    legs: list[float] = []
    total_empty_km = 0.0

    # Events arrive in `info` and are handed straight to the reward function.
    # Intercept the reward call, which is where the env itself reads them, so
    # this observes exactly the distances the reward charges.
    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        nonlocal total_empty_km
        for did, ev in (events or {}).items():
            if ev.get("is_empty_move", False):
                km = float(ev.get("distance_moved", 0.0)) / 1000.0
                total_empty_km += km
                if did < n_drv and status[did] == "relocating":
                    leg_m[did] = leg_m.get(did, 0.0) + km
        out = orig(events, dispatched_this_turn)
        for did in range(n_drv):
            cur = env._last_obs[did]["self"].get("status")
            if status[did] == "relocating" and cur != "relocating":
                km = leg_m.pop(did, 0.0)
                if km > 0.0:
                    legs.append(km)
            status[did] = cur
        return out

    env._compute_zone_rewards = traced

    for ep in range(args.episodes):
        env.reset(seed=args.base_seed + ep)
        leg_m.clear()
        status = [""] * n_drv
        while env.agents:
            a = env.agent_selection
            m = env.action_mask(a)
            d = np.flatnonzero(m[1: 1 + env._max_pending])
            r = np.flatnonzero(m[1 + env._max_pending:])
            act = int(d[0]) + 1 if len(d) else (
                int(r[0]) + 1 + env._max_pending if len(r) else 0)
            env.step(act)

    n = len(legs)
    print("=" * 68)
    print("Relocation leg distance, measured per leg")
    print("=" * 68)
    print(f"completed legs               : {n}")
    print(f"total empty km, all causes   : {total_empty_km:.1f}")
    if not n:
        print("no legs observed")
        return
    srt = sorted(legs)
    print()
    print(f"mean   leg distance          : {sum(legs)/n:.2f} km")
    print(f"median leg distance          : {srt[n//2]:.2f} km")
    print(f"p90    leg distance          : {srt[int(0.9*n)]:.2f} km")
    print(f"max    leg distance          : {srt[-1]:.2f} km")
    print()
    print("break-even check against the HARD-CODED 2.49 km:")
    beta, share, gamma = cfg.beta, cfg.relocator_share, cfg.gamma
    for label, km in (("hardcoded 2.49", 2.49), ("measured mean", sum(legs)/n),
                      ("median", srt[n//2]), ("p90", srt[int(0.9*n)])):
        cost = beta * km
        print(f"   {label:<16} {km:6.2f} km -> cost {cost:.4f}, "
              f"margin {share*gamma - cost:+.4f} ({share*gamma/cost:.2f}x)")


if __name__ == "__main__":
    main()
