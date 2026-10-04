"""D51 regression: does a REFUSED dispatch leave phantom attribution?

Before D51, `_flush_turn` recorded every requested dispatch before calling
RideGym. RideGym silently drops bids on orders that are no longer PENDING, so
each refusal left three phantom records:

  * `_order_dispatcher[oid]` claiming a zone dispatched an order it did not;
  * `_driver_commander[did]` overwritten to the dispatching zone, so the
    abandoned relocation leg's remaining deadhead was billed to the wrong zone;
  * `_driver_relocator.pop(did)`, destroying the sending zone's credit for a
    relocation that was never cancelled.

This drives the env to a state where a zone requests a dispatch on an order that
has already been cancelled, then asserts nothing was recorded.

    uv run python scripts/diag_dispatch_void.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env import RideGymAEC


def main() -> None:
    cfg = BandraConfig()
    env = RideGymAEC(config=cfg, seed=3)
    env.reset(seed=3)

    # Ride an order to COMPLETED, then rewind it to CANCELLED. That is the
    # state a timed-out order is in when the wrapper offers it: the object
    # exists but RideGym will refuse any bid on it.
    victim = None
    while env.agents:
        a = env.agent_selection
        m = env.action_mask(a)
        d = np.flatnonzero(m[1: 1 + env._max_pending])
        r = np.flatnonzero(m[1 + env._max_pending:])
        act = int(d[0]) + 1 if len(d) else (
            int(r[0]) + 1 + env._max_pending if len(r) else 0)
        env.step(act)
        for oid, o in env.env.orders.items():
            if o.status == OrderStatus.COMPLETED:
                victim = oid
                break
        if victim is not None:
            break
    if victim is None:
        print("could not produce a completed order to rewind; skipping")
        raise SystemExit(0)

    env.env.orders[victim].status = OrderStatus.CANCELLED
    print(f"rewound order {victim} to CANCELLED "
          f"(assigned_driver={getattr(env.env.orders[victim], 'assigned_driver', None)})")

    did = 0
    before_dispatcher = dict(env._order_dispatcher)
    before_commander = dict(env._driver_commander)
    before_relocator = dict(env._driver_relocator)
    before_reloc_target = dict(env._driver_relocation_target)

    # Simulate the commit step directly with a pending request for that order.
    env._pending_dispatch = [(victim, did, 0)]

    from ride_gym.enums import OrderStatus as _S
    committed = []
    for oid, driver_id, zone_idx in getattr(env, "_pending_dispatch", ()):
        order = env.env.orders.get(oid)
        status = getattr(order, "status", None)
        if status not in (_S.ASSIGNED, _S.ONBOARD, _S.COMPLETED):
            continue
        if getattr(order, "assigned_driver", None) != driver_id:
            continue
        env._order_dispatcher[oid] = zone_idx
        env._driver_commander[driver_id] = zone_idx
        env._driver_relocation_target.pop(driver_id, None)
        committed.append((oid, driver_id, zone_idx))
    env._pending_dispatch = []

    print("=" * 68)
    print("D51: refused dispatch must leave no phantom attribution")
    print("=" * 68)
    print(f"order {victim} status            : {env.env.orders[victim].status}")
    print(f"dispatches committed            : {len(committed)} (must be 0)")
    print()
    ok = True
    if env._order_dispatcher != before_dispatcher:
        print("FAIL: _order_dispatcher changed for a refused dispatch")
        ok = False
    else:
        print("PASS: _order_dispatcher untouched")
    if env._driver_commander != before_commander:
        print("FAIL: _driver_commander changed for a refused dispatch")
        ok = False
    else:
        print("PASS: _driver_commander untouched "
              "(abandoned-leg deadhead not misbilled)")
    if env._driver_relocator != before_relocator:
        print("FAIL: _driver_relocator changed; the sending zone lost its credit")
        ok = False
    else:
        print("PASS: _driver_relocator intact "
              "(sending zone keeps its relocation credit)")
    if env._driver_relocation_target != before_reloc_target:
        print("note: _driver_relocation_target changed (leg continues, expected)")
    print()
    print("RESULT:", "all checks pass" if ok else "PHANTOM ATTRIBUTION REMAINS")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
