"""Is low service caused by too few cars, or by the mask holding cars back?

For every zone-turn that has at least one waiting order, this records:
  * whether a dispatchable car is already in the zone,
  * otherwise whether a neighbour holds an idle car,
  * and if so whether a LEGAL relocate action from that neighbour to this zone
    existed at that instant.

The question is always asked from the SUPPLY side (does a zone that actually
holds an idle car have this starved zone in its own legal-target list?), because
asking the starved zone returns [] by construction -- no idle car, no targets
(pettingzoo_env.py:718-722) -- which reads as a mask block that is not there.

The earlier version asked the question correctly but always NOOP-ed, over 2
episodes on one split: debugging output, not a result. This runs every policy arm
so the fleet distribution is a consequence of a stated policy rather than of a
frozen no-op world, and pairs the arms on identical seeds.

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
    arms = ["dispatch-only", "dispatch-first", "relocate-first-best", "random-legal"]
    seeds = [args.base_seed + i for i in range(args.episodes)]

    print("scenario: hotspot2 A=%d B=%d pct=%.0f%%  split=%s  seeds=%s (%d paired "
          "episodes per arm)" % (args.zone_a, args.zone_b, args.pct, args.split,
                                 seeds, args.episodes))
    print("supply-side question: does a zone HOLDING an idle car have this "
          "starved zone in\nits own _legal_relocate_targets list? "
          "(pettingzoo_env.py:718-769)\n")

    per_arm: dict[str, Counter] = {}
    per_arm_sr: dict[str, list[float]] = {}
    # per-episode mask-block counts, so the comparison can be PAIRED
    per_ep: dict[str, list[int]] = {}
    for arm in arms:
        st = Counter()
        srs: list[float] = []
        blocked: list[int] = []
        for seed in seeds:
            ep_blocked = 0
            env.reset(seed=seed)
            env.reset_observer()
            rng = np.random.default_rng(seed * 1_000_003 + 7)
            while env.agents:
                zone = env._zone_int_id(env.agent_selection)
                idle = int(env._zone_idle_counts[zone])
                pend = int(round(env._pending_per_zone[zone] * env._num_drivers))

                if pend >= 1:
                    st["turns_with_demand"] += 1
                    nbrs = [n for n in env.env.region_neighbours[zone] if n != zone]
                    nbr_idle = sum(int(env._zone_idle_counts[n]) for n in nbrs)
                    nbr_pend = sum(
                        int(round(env._pending_per_zone[n] * env._num_drivers))
                        for n in nbrs
                    )
                    if idle > 0:
                        st["car_here"] += 1
                    else:
                        st["no_car_here"] += 1
                        if nbr_idle > 0:
                            st["cars_in_neighbour"] += 1
                            if nbr_pend >= 1:
                                st["neighbour_also_demanded"] += 1
                            can_come = any(
                                zone in env._legal_relocate_targets(n)
                                for n in nbrs
                                if int(env._zone_idle_counts[n]) > 0
                            )
                            if can_come:
                                st["legal_move_exists"] += 1
                            else:
                                st["MASK_BLOCKS_ALL"] += 1
                                ep_blocked += 1

                agent = env.agent_selection
                action, intended = policies[arm](env, agent, rng)
                action = verify_action(env, agent, action, intended)
                env.step(action)
                env.observe_turn()
            kpi = env.kpi_summary()
            st["served"] += int(kpi["served"])
            st["cancelled"] += int(kpi["cancelled"])
            srs.append(100.0 * float(kpi["served"])
                       / max(float(kpi["served"]) + float(kpi["cancelled"]), 1.0))
            blocked.append(ep_blocked)
        per_arm[arm] = st
        per_arm_sr[arm] = srs
        per_ep[arm] = blocked

    def pct(n, d):
        return 100.0 * n / max(d, 1)

    for arm in arms:
        st = per_arm[arm]
        t = st["turns_with_demand"]
        d = st["no_car_here"]
        print("=" * 92)
        print("%s   (service rate %.1f%%, served %d, cancelled %d)"
              % (arm, np.mean(per_arm_sr[arm]), st["served"], st["cancelled"]))
        print("=" * 92)
        print("  zone-turns WITH pending demand : %d" % t)
        print("    a car was already idle here  : %5.1f%%" % pct(st["car_here"], t))
        print("    NO car here                  : %5.1f%%" % pct(st["no_car_here"], t))
        print("  of the no-car-here turns (%d):" % d)
        print("    a neighbour held a car       : %5.1f%%" % pct(st["cars_in_neighbour"], d))
        print("    that neighbour also demanded : %5.1f%%" % pct(st["neighbour_also_demanded"], d))
        print("    a LEGAL inbound move existed : %5.1f%%" % pct(st["legal_move_exists"], d))
        print("    MASK BLOCKED EVERYTHING      : %5.1f%%" % pct(st["MASK_BLOCKS_ALL"], d))
        print()

    print("=" * 92)
    print("PAIRED vs dispatch-only (never relocates), per-episode")
    print("=" * 92)
    
    print("  -- mask-blocked no-car-here turns, per episode")
    for arm in arms:
        if arm == "dispatch-only":
            continue
        p = paired(per_ep[arm], per_ep["dispatch-only"], "mask_blocked")
        print("     %-24s %s" % (arm, fmt(p)))
    print()
    print("  -- service rate %")
    for arm in arms:
        if arm == "dispatch-only":
            continue
        p = paired(per_arm_sr[arm], per_arm_sr["dispatch-only"], "sr")
        print("     %-24s %s" % (arm, fmt_pp(p)))
    print()

    print("Interpretation: 'MASK BLOCKED EVERYTHING' is the share of no-car-here")
    print("demanded turns where cars existed next door and no legal inbound move")
    print("was offered. It is a property of _legal_relocate_targets under the")
    print("policy in force, not a permanent property of the env: the same code")
    print("offers different menus under different policies, which is why every")
    print("arm is reported rather than one.")


def fmt(p: dict) -> str:
    return (f"{p['mean']:+8.2f} turns +/- {p['std']:6.2f} sd (sem {p['sem']:5.2f}, "
            f"n={p['n']}, {p['wins']}W/{p['losses']}L/{p['ties']}T)")


def fmt_pp(p: dict) -> str:
    return (f"{p['mean']:+8.2f}pp +/- {p['std']:6.2f} sd (sem {p['sem']:5.2f}, "
            f"n={p['n']}, {p['wins']}W/{p['losses']}L/{p['ties']}T)")


if __name__ == "__main__":
    main()