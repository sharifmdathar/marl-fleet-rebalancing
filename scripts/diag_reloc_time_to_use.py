"""How fast does a relocated car actually get used?

The relocation decision has an asymmetry that makes it hard to learn:

    cost   -0.149   paid IMMEDIATELY, the turn the zone commands the move
    credit +0.45    paid on ride COMPLETION, measured 21-40 turns later

Nothing arrives in the same frame as the cost. Two fixes have been tried:
paying on arrival (D28, off -- paid for cars that never served anyone) and
raising gae_lambda (D46). Neither shortens the gap between the decision and
the evidence that it was right.

There is a third signal that is both fast AND verified: the moment a relocated
car is DISPATCHED to a rider. That is proof the relocation produced a service,
and it happens long before the ride completes.

This measures turns-from-relocation-arrival to first-dispatch. A relocation
whose car is dispatched within a couple of turns demonstrably worked; one
whose car sits idle for 30 turns did not, and no amount of credit should have
been paid for it.

Status lifecycle observed in the env: relocating -> to_pickup (arrived and
already assigned) or relocating -> idle (arrived with nothing waiting).
Arrival is any transition OUT of "relocating".

Read-only: tracks status transitions after the env's own step and never calls
`_newly_completed_ids`, so `_seen_completed_ids` is never consumed early.

    uv run python scripts/diag_reloc_time_to_use.py --episodes 4
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    mp = env._max_pending
    n_drv = env._num_drivers

    t = 0
    prev_status = [""] * n_drv
    # A driver can be relocated more than once per episode, so everything is
    # keyed by RELOCATION LEG, not by driver id. Keying by driver silently
    # overwrote `arrived` with a later leg while `used` kept the earlier
    # leg's dispatch, which produced negative gaps. Legs are opened when a
    # driver enters "relocating" and closed when it leaves.
    next_leg = 1
    open_leg: dict[int, int] = {}       # did -> leg currently in transit
    last_leg: dict[int, int] = {}       # did -> most recent closed leg
    arrived: dict[int, int] = {}        # leg -> arrival turn
    used: dict[int, int] = {}           # leg -> first dispatch turn after arrival
    never_moved = 0                     # dispatches of never-relocated drivers

    orig_flush = env._flush_turn

    def traced() -> None:
        nonlocal t, next_leg
        orig_flush()
        t += 1
        for did in range(n_drv):
            cur = env._last_obs[did]["self"].get("status")
            was = prev_status[did]
            if was != "relocating" and cur == "relocating":
                open_leg[did] = next_leg
                next_leg += 1
            elif was == "relocating" and cur != "relocating":
                leg = open_leg.pop(did, None)
                if leg is not None:
                    last_leg[did] = leg
                    arrived[leg] = t
                    if cur == "to_pickup":
                        used.setdefault(leg, t)
            prev_status[did] = cur

    env._flush_turn = traced

    for ep in range(args.episodes):
        env.reset(seed=args.base_seed + ep)
        t = 0
        next_leg = 1
        prev_status = [""] * n_drv
        open_leg.clear()
        last_leg.clear()
        arrived.clear()
        used.clear()
        never_moved = 0
        while env.agents:
            a = env.agent_selection
            m = env.action_mask(a)
            d = np.flatnonzero(m[1: 1 + mp])
            r = np.flatnonzero(m[1 + mp:])
            act = int(d[0]) + 1 if len(d) else (int(r[0]) + 1 + mp if len(r) else 0)
            pending = {
                oid for oid, o in env.env.orders.items()
                if str(getattr(o, "status", "")).endswith("PENDING")
            }
            env.step(act)
            for oid in pending:
                o = env.env.orders.get(oid)
                if o is None or str(getattr(o, "status", "")).endswith("PENDING"):
                    continue
                did = getattr(o, "assigned_driver", None)
                if did is None:
                    continue
                did = int(did)
                leg = last_leg.get(did)
                if leg is not None and leg in arrived:
                    used.setdefault(leg, t)
                elif leg is None:
                    never_moved += 1

    print("=" * 70)
    print("Turns from relocation arrival to first dispatch")
    print("=" * 70)
    gaps = sorted(used[leg] - arrived[leg] for leg in used if leg in arrived)
    assert all(g >= 0 for g in gaps), f"negative gap: {gaps[:5]}"
    print(f"relocation legs that arrived   : {len(arrived)}")
    print(f"arrivals that got dispatched   : {len(gaps)}")
    print(f"arrivals that never dispatched : {len(arrived) - len(gaps)}")
    if not gaps:
        print("\nno relocated car was dispatched -- cannot measure")
        return
    n = len(gaps)
    print(f"dispatched at all              : {100*n/len(arrived):.1f}% of arrivals")

    buckets: Counter[str] = Counter()
    for g in gaps:
        if g <= 0:
            buckets["0    already assigned on arrival"] += 1
        elif g <= 2:
            buckets["1-2  immediate"] += 1
        elif g <= 5:
            buckets["3-5"] += 1
        elif g <= 10:
            buckets["6-10"] += 1
        elif g <= 20:
            buckets["11-20"] += 1
        else:
            buckets[">20  too late to be evidence"] += 1

    print(f"\n{'turns after arrival':>32}  {'cars':>6}  {'share':>7}")
    for k in ("0    already assigned on arrival", "1-2  immediate", "3-5",
              "6-10", "11-20", ">20  too late to be evidence"):
        if buckets[k]:
            print(f"{k:>32}  {buckets[k]:>6}  {100*buckets[k]/n:6.1f}%")

    print(f"\nmedian {gaps[n//2]} turns, mean {sum(gaps)/n:.1f}")
    print("\ncumulative:")
    for thr in (0, 1, 2, 5, 10):
        c = 100 * sum(1 for g in gaps if g <= thr) / n
        print(f"  dispatched within {thr:2d} turn(s) of arrival: {c:5.1f}%")

    print(f"\ncontrol: {never_moved} dispatches by drivers with no completed leg")


if __name__ == "__main__":
    main()
