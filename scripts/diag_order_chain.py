"""For every order that never got served, where exactly does the chain break?

For each order the file records, over the zone-turns it spent PENDING:
  * turns where a dispatch action for THAT order was legal in ITS zone, and
  * turns where a relocating driver had arrived in its zone.

Then every cancelled order is split by whether a dispatch was ever legal:
"never dispatchable" (nothing the policy could have done from the zone's side)
vs "dispatchable but timed out" (a legal dispatch existed and was not taken, or
was taken too late).

The earlier version of this file counted `id(o)` as the order key across
episodes, which silently mixes object identities between episodes, and ran only
the degenerate relocate-only arm over 2 episodes. It now keys on order_id, runs
every arm, and pairs on identical seeds.

Self-check: every chosen action is asserted legal in the same mask and asserted
to decode as the policy declared.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.enums import OrderStatus

from diag_common import ObservedEnv, make_policies, paired, verify_action

from marl_fleet_rebalancing.config import NYCConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--zone-a", type=int, default=8)
    ap.add_argument("--zone-b", type=int, default=13)
    ap.add_argument("--pct", type=float, default=60.0)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split
    cfg.demand_shift_mode = "hotspot2"
    cfg.hotspot_zone_a = args.zone_a
    cfg.hotspot_zone_b = args.zone_b
    cfg.hotspot_pct = args.pct
    cfg.hotspot_split = 0.5

    env = ObservedEnv(cfg)
    policies = make_policies()
    arms = ["dispatch-only", "dispatch-first", "relocate-first-best",
            "relocate-only(BROKEN)"]
    seeds = [args.base_seed + i for i in range(args.episodes)]

    print("scenario: hotspot2 A=%d B=%d pct=%.0f%%  split=%s  seeds=%s (%d paired "
          "episodes per arm)" % (args.zone_a, args.zone_b, args.pct, args.split,
                                 seeds, args.episodes))
    print("order key is order_id (the earlier version keyed on id(o), which "
          "aliases across episodes)\n")

    agg: dict[str, Counter] = {}
    per_ep_cancelled_never: dict[str, list[int]] = {}
    per_ep_sr: dict[str, list[float]] = {}
    for arm in arms:
        st = Counter()
        never_disp: list[int] = []
        srs: list[float] = []
        for seed in seeds:
            env.reset(seed=seed)
            env.reset_observer()
            rng = np.random.default_rng(seed * 1_000_003 + 7)
            # order_id -> zone-turns it spent pending
            turns_pending: Counter = Counter()
            # order_id -> turns a dispatch for it was legal in its own zone
            turns_dispatch_legal: Counter = Counter()
            # order_id -> zone it was dispatched from (filled on observation)
            while env.agents:
                zone = env._zone_int_id(env.agent_selection)
                cand_ids = {c["order_id"] for c in env._zone_candidate_orders[zone]}
                dispatchable = env._zone_dispatchable_counts[zone] > 0
                relocating_here = any(
                    env._last_obs[d]["self"]["status"] == "relocating"
                    and int(max(0, min(int(env._last_obs[d]["self"]
                                             .get("current_region", 0)),
                                       env._num_zones - 1))) == zone
                    for d in range(env._num_drivers)
                )
                for o in env.env.orders.values():
                    if o.status != OrderStatus.PENDING:
                        continue
                    oid = int(o.order_id)
                    turns_pending[oid] += 1
                    if oid in cand_ids and dispatchable:
                        turns_dispatch_legal[oid] += 1
                    if oid in cand_ids and relocating_here:
                        st["pending_turn_relocating_car_here"] += 1
                agent = env.agent_selection
                action, intended = policies[arm](env, agent, rng)
                action = verify_action(env, agent, action, intended)
                env.step(action)
                env.observe_turn()

            ep_never = 0
            for o in env.env.orders.values():
                oid = int(o.order_id)
                if o.status == OrderStatus.COMPLETED:
                    st["completed"] += 1
                    if turns_dispatch_legal[oid] > 0:
                        st["completed_had_legal_dispatch"] += 1
                    else:
                        st["completed_never_dispatchable"] += 1
                elif o.status == OrderStatus.CANCELLED:
                    st["cancelled"] += 1
                    if turns_dispatch_legal[oid] == 0:
                        st["cancelled_never_dispatchable"] += 1
                        ep_never += 1
                    else:
                        st["cancelled_despite_legal"] += 1
                    st["cancelled_pending_turns"] += turns_pending[oid]
            never_disp.append(ep_never)
            kpi = env.kpi_summary()
            srs.append(100.0 * float(kpi["served"])
                       / max(float(kpi["served"]) + float(kpi["cancelled"]), 1.0))
        agg[arm] = st
        per_ep_cancelled_never[arm] = never_disp
        per_ep_sr[arm] = srs

    for arm in arms:
        st = agg[arm]
        tot = max(st["completed"] + st["cancelled"], 1)
        print("=" * 92)
        print("%s   (service rate %.1f%%)"
              % (arm, np.mean(per_ep_sr[arm])))
        print("=" * 92)
        print("  completed : %5d  (%.1f%% of resolved)" % (st["completed"],
                                                           100 * st["completed"] / tot))
        print("  cancelled : %5d  (%.1f%%)" % (st["cancelled"], 100 * st["cancelled"] / tot))
        print("  of CANCELLED orders, by whether a dispatch was EVER legal in "
              "their own zone:")
        print("    never dispatchable       : %5d  (%.1f%% of cancelled)"
              % (st["cancelled_never_dispatchable"],
                 100 * st["cancelled_never_dispatchable"] / max(st["cancelled"], 1)))
        print("    dispatchable but timed out: %5d  (%.1f%%)"
              % (st["cancelled_despite_legal"],
                 100 * st["cancelled_despite_legal"] / max(st["cancelled"], 1)))
        print("  of COMPLETED orders:")
        print("    had a legal dispatch turn : %5d  (%.1f%%)"
              % (st["completed_had_legal_dispatch"],
                 100 * st["completed_had_legal_dispatch"]
                 / max(st["completed"], 1)))
        print("    never dispatchable at all: %5d  (%.1f%%)"
              % (st["completed_never_dispatchable"],
                 100 * st["completed_never_dispatchable"]
                 / max(st["completed"], 1)))
        print("  zone-turns where a pending order's zone held a RELOCATING car: %d"
              % st["pending_turn_relocating_car_here"])
        print()

    print("=" * 92)
    print("PAIRED vs relocate-only(BROKEN) -- the policy that reported 0 served")
    print("=" * 92)
    print("  pooled levels over %d episodes (NOT paired -- see the paired table "
          "below for\n  the statistic any claim should rest on):" % args.episodes)
    for arm in arms:
        print("     %-24s completed=%5d cancelled=%5d  never_dispatchable=%5d"
              % (arm, agg[arm]["completed"], agg[arm]["cancelled"],
                 agg[arm]["cancelled_never_dispatchable"]))
    print()

    print("  -- PAIRED service rate vs relocate-only(BROKEN)")
    for arm in arms:
        if arm == "relocate-only(BROKEN)":
            continue
        p = paired(per_ep_sr[arm], per_ep_sr["relocate-only(BROKEN)"], "sr")
        print("     %-24s %+8.2fpp +/- %5.2f sd (sem %5.2f, n=%d, %dW/%dL/%dT)"
              % (arm, p["mean"], p["std"], p["sem"], p["n"], p["wins"],
                 p["losses"], p["ties"]))


if __name__ == "__main__":
    main()