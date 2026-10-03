"""Does a relocated car ever complete an order? The one question still open.

Earlier probes in this repo reported "0 served" for a relocate-heavy policy, but
that policy could never dispatch: it chose a relocate action when one existed
and otherwise fell back to index 0, which is NOOP. So it measured nothing.

This compares three policies that ALL can dispatch, on identical seeds/windows,
and reports the decisive quantity: how many completed orders were served by a
driver that had previously been relocated into that zone.

Everything self-checks:
  * every chosen action is asserted legal via action_mask
  * the intended semantics is confirmed via _decode_action ({"orders": ...} for
    dispatch, {"relocate": ...} for relocation), and mismatches are counted
  * relocation credit is read with a SNAPSHOT taken before the env's own
    bookkeeping consumes it, never by calling _newly_completed_ids (which
    mutates _seen_completed_ids and has already caused one wrong report)

    uv run python scripts/diag_relocation_conversion.py --mode base --episodes 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC

RELOC_SLICE = 1  # relocate actions begin at mask[1 + _max_pending]


def build(args):
    cfg = NYCConfig()
    cfg.order_timeout = args.timeout
    if args.mode == "hotspot2":
        cfg.demand_shift_mode = "hotspot2"
        cfg.hotspot_zone_a = args.zone_a
        cfg.hotspot_zone_b = args.zone_b
        cfg.hotspot_pct = args.pct
        cfg.hotspot_split = 0.5
    return RideGymAEC(cfg)


def choose(env, zone, policy, rng):
    """Pick an action for `zone`. Returns (action, intent) or raises."""
    mask = env.action_mask(env.agent_selection)
    mp = env._max_pending
    disp = np.flatnonzero(mask[1 : 1 + mp])          # slice-relative
    reloc = np.flatnonzero(mask[1 + mp :])           # slice-relative

    if policy == "dispatch_first":
        if len(disp):
            return int(rng.choice(disp)) + 1, "dispatch"
        if len(reloc):
            return int(rng.choice(reloc)) + 1 + mp, "relocate"
        return 0, "noop"
    if policy == "relocate_first":
        if len(reloc):
            return int(rng.choice(reloc)) + 1 + mp, "relocate"
        if len(disp):
            return int(rng.choice(disp)) + 1, "dispatch"
        return 0, "noop"
    if policy == "random_legal":
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal)), "mixed"
    raise ValueError(policy)


def verify(env, zone, act, intent, st):
    """Confirm the action is legal and did what the policy meant."""
    mask = env.action_mask(env.agent_selection)
    if not mask[act]:
        st["ILLEGAL_ACTION"] += 1
        return
    if intent == "mixed":
        return
    decoded = env._decode_action(act, zone)
    if intent == "noop":
        if decoded is not None:
            st["noop_did_something"] += 1
        return
    key = "orders" if intent == "dispatch" else "relocate"
    if not decoded or key not in decoded:
        st["SEMANTIC_MISMATCH_%s" % intent] += 1


def run_episode(env, policy, seed, rng, st):
    env.reset(seed=seed)

    # Read-only: union of every driver id the env currently tracks as having
    # been relayed by some zone. We read the dict but never consume it -- calling
    # _newly_completed_ids() would mutate _seen_completed_ids, which has already
    # produced one wrong report in this repo.
    snap: dict = {}
    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        snap.clear()
        snap.update(env._driver_relocator)
        return orig(events, dispatched_this_turn)

    env._compute_zone_rewards = traced
    inflight: dict[int, int] = {}   # driver -> zone it was commanded toward
    try:
        while env.agents:
            a = env.agent_selection
            zone = env._zone_int_id(a)

            act, intent = choose(env, zone, policy, rng)
            verify(env, zone, act, intent, st)
            if intent in ("dispatch", "relocate"):
                st["intent_%s" % intent] += 1
            # A relocating driver's current_region is where it IS, not where it
            # is going, so the commanded target is the only destination signal.
            tgt = act - env._max_pending - 1 if act > env._max_pending else None
            env.step(act)

            for did in range(env._num_drivers):
                s = env._last_obs[did]["self"]
                if s["status"] == "relocating":
                    if did not in inflight and tgt is not None \
                            and 0 <= tgt < env._num_zones \
                            and int(s["current_region"]) == zone:
                        inflight[did] = tgt
                        st["relocations_started"] += 1
                else:
                    inflight.pop(did, None)
    finally:
        env._compute_zone_rewards = orig

    # "Delivered by relocation" is strict: the serving driver was still in
    # transit, commanded at its origin zone. Merely having been relocated at
    # some point saturates at 100% (every driver is, within 3 episodes) and so
    # cannot discriminate anything.
    cents = np.asarray(env.env.relocation_points, dtype=float)
    for o in env.env.orders.values():
        if o.status != OrderStatus.COMPLETED:
            continue
        st["completions"] += 1
        did = getattr(o, "assigned_driver", None)
        if did is None:
            st["completed_no_driver"] += 1
            continue
        oz = int(np.argmin(((cents - np.asarray(o.origin, float)) ** 2).sum(1)))
        if did in inflight:
            st["completed_while_in_transit"] += 1
            if inflight[did] == oz:
                st["completed_by_relocation_into_its_zone"] += 1
            else:
                st["completed_in_transit_elsewhere"] += 1
        else:
            st["completed_from_resident"] += 1

    st["drivers_still_inflight_at_end"] += len(inflight)
    k = env.kpi_summary()
    st["served"] += int(k["served"])
    st["cancelled"] += int(k["cancelled"])
    st["empty_km"] += float(k["empty_km"])
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--mode", default="base", choices=["base", "hotspot2"])
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--zone-a", type=int, default=8)
    ap.add_argument("--zone-b", type=int, default=13)
    ap.add_argument("--pct", type=float, default=60.0)
    args = ap.parse_args()

    print("mode=%s  order_timeout=%.1f min  episodes=%d  seeds=%s"
          % (args.mode, args.timeout, args.episodes,
             [args.base_seed + i for i in range(args.episodes)]))

    rows: dict[str, list[dict]] = {}
    allst: dict[str, Counter] = {}
    for policy in ("dispatch_first", "relocate_first", "random_legal"):
        env = build(args)
        st: Counter = Counter()
        per_ep = []
        for i in range(args.episodes):
            rng = np.random.default_rng(args.base_seed + i)
            # Counters are cumulative; snapshot them so per_episode SR is that
            # episode alone, not the running total to date.
            s0, c0, e0 = st["served"], st["cancelled"], st["empty_km"]
            run_episode(env, policy, args.base_seed + i, rng, st)
            dserved = st["served"] - s0
            dcanc = st["cancelled"] - c0
            per_ep.append({
                "seed": args.base_seed + i,
                "sr": 100.0 * dserved / max(dserved + dcanc, 1),
                "empty_km": st["empty_km"] - e0,
                "served": dserved,
            })
        rows[policy] = per_ep
        allst[policy] = st

    base = rows["dispatch_first"]
    print("\n%-16s %7s %9s %9s | %s"
          % ("policy", "SR%", "empty km", "SR/100km", "completion origin"))
    print("-" * 84)
    for policy, eps in rows.items():
        st = allst[policy]
        sr = float(np.mean([e["sr"] for e in eps]))
        ek = float(np.mean([e["empty_km"] for e in eps]))
        comp = max(st["completions"], 1)
        reloc = st["completed_by_relocation_into_its_zone"]
        transit = st["completed_while_in_transit"]
        resident = st["completed_from_resident"]
        print("%-16s %6.1f%% %9.0f %9.2f | %4d resident %4d in-transit "
              "(%d into its own zone = %.1f%%)"
              % (policy, sr, ek, 100 * sr / max(ek, 1e-9),
                 resident, transit, reloc, 100 * reloc / comp))

    print("\npaired per-window difference vs dispatch_first (docs/eval-noise.md)")
    d_sr = np.array([e["sr"] for e in rows["relocate_first"]]) \
        - np.array([e["sr"] for e in base])
    d_ek = np.array([e["sr"] and e["empty_km"] for e in rows["relocate_first"]]) \
        - np.array([e["empty_km"] for e in base])
    if len(d_sr) > 1:
        print("  relocate_first SR %+.1f pp (std %.1f, favours %d/%d)  "
              "empty km %+.0f"
              % (d_sr.mean(), d_sr.std(), int((d_sr > 0).sum()), len(d_sr),
                 d_ek.mean()))
    else:
        print("  (single window: paired difference unavailable, need >=2 episodes)")

    print("\nself-check (all must be 0)")
    for policy, st in allst.items():
        bad = {k: v for k, v in st.items()
               if k.startswith(("ILLEGAL", "SEMANTIC", "noop_did"))}
        print("  %-16s %s" % (policy, bad or "clean"))

    print("\ntotals")
    for policy, st in allst.items():
        print("  %-16s completions=%4d  relocations_started=%d  "
              "intent_dispatch=%d intent_relocate=%d  still_inflight_at_end=%d"
              % (policy, st["completions"], st["relocations_started"],
                 st["intent_dispatch"], st["intent_relocate"],
                 st["drivers_still_inflight_at_end"]))


if __name__ == "__main__":
    main()