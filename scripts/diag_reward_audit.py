"""Audit of the current reward path. Findings, each with a reproduction.

Run: uv run python scripts/diag_reward_audit.py --episodes 3

1. Phantom dispatch records. `_flush_turn` records `_order_dispatcher[oid]`,
   `_driver_commander[did] = zone_idx`, and `_order_relocator.pop(did)` for
   every dispatch it *requests*, before RideGym can refuse it. RideGym drops
   bids on non-pending orders silently. So a dispatch that never happened still
   (a) leaves a false dispatcher record, (b) overwrites the commander to the
   dispatching zone, and (c) destroys the relocating zone's credit.

2. Relocation leg distance. The break-even invariant was checked against a
   hardcoded 2.49 km carried over from an older run. Re-measured here.

3. Partial-leg credit. `beta * km` is charged per turn for distance actually
   flown, but `relocator_share * gamma` is paid in full at dispatch. A leg
   dispatched to a rider mid-flight (normal: `allow_relocating_dispatch=True`)
   earns the whole credit while paying part of the cost.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.enums import OrderStatus

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
    mp = env._max_pending

    c = Counter()
    leg_m: dict[int, float] = {}
    legs: list[float] = []
    status = [""] * n_drv
    paid_credit_km: list[float] = []

    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        for did, ev in (events or {}).items():
            if ev.get("is_empty_move", False):
                km = float(ev.get("distance_moved", 0.0)) / 1000.0
                if did < n_drv and status[did] == "relocating":
                    leg_m[did] = leg_m.get(did, 0.0) + km
        firing = [(oid, did, z) for oid, did, z in (dispatched_this_turn or ())
                  if did in env._driver_relocator]
        out = orig(events, dispatched_this_turn)
        for oid, did, zone_idx in firing:
            paid_credit_km.append(leg_m.pop(did, 0.0))
        for did in range(n_drv):
            cur = env._last_obs[did]["self"].get("status")
            if status[did] == "relocating" and cur != "relocating":
                km = leg_m.pop(did, 0.0)
                if km > 0.0:
                    legs.append(km)
            status[did] = cur
        return out

    env._compute_zone_rewards = traced

    orig_flush = env._flush_turn
    seen: set[int] = set()

    def flush_traced() -> None:
        reloc_before = dict(env._driver_relocator)
        cmd_before = dict(env._driver_commander)
        orig_flush()
        for oid in env._order_dispatcher:
            if oid in seen:
                continue
            seen.add(oid)
            o = env.env.orders.get(oid)
            if o is None:
                continue
            if o.status in (OrderStatus.PENDING, OrderStatus.CANCELLED):
                c['void'] += 1
                did = getattr(o, "assigned_driver", None)
                if did is not None:
                    sender = reloc_before.get(did)
                    holder = cmd_before.get(did)
                    if sender is not None and sender != holder:
                        c['void_crosszone'] += 1
            c['recorded'] += 1

    env._flush_turn = flush_traced

    for ep in range(args.episodes):
        env.reset(seed=args.base_seed + ep)
        leg_m.clear()
        status = [""] * n_drv
        seen.clear()
        c['recorded'] = 0
        c['void'] = 0
        c['void_crosszone'] = 0
        legs.clear()
        paid_credit_km.clear()
        while env.agents:
            a = env.agent_selection
            m = env.action_mask(a)
            d = np.flatnonzero(m[1: 1 + mp])
            r = np.flatnonzero(m[1 + mp:])
            act = int(d[0]) + 1 if len(d) else (
                int(r[0]) + 1 + mp if len(r) else 0)
            env.step(act)

        print(f"--- episode {ep}")
        print(f"    dispatches recorded      : {c['recorded']}")
        print(f"    voided by RideGym        : {c['void']} "
              f"({100*c['void']/max(c['recorded'],1):.1f}%)")
        print(f"    of which cross-zone      : {c['void_crosszone']}")
        print(f"    legs measured            : {len(legs)}")
        if legs:
            print(f"    mean leg km              : {sum(legs)/len(legs):.2f}")
            print(f"    median leg km            : {sorted(legs)[len(legs)//2]:.2f}")
        if paid_credit_km:
            z = sum(1 for k in paid_credit_km if k > 0)
            full = sum(1 for k in paid_credit_km
                       if 0 < k * cfg.beta < cfg.relocator_share * cfg.gamma)
            print(f"    credits paid             : {len(paid_credit_km)} "
                  f"({z} after flying)")
            print(f"    credit > cost actually paid: {full}/{z or 1} "
                  f"({100*full/max(z,1):.0f}%)")

    print()
    print("Reading: 'voided' means the wrapper credited a dispatch that never")
    print("happened. 'credit > cost' means a partial leg was paid in full.")


if __name__ == "__main__":
    main()
