"""Can a car legally reach the two-hotspot zones when they have unmet demand?

For every turn in which a hotspot zone has at least one waiting order and no
idle car of its own, this asks whether any adjacent zone that actually holds an
idle car offers the hotspot in its own legal-target list -- and if not, which
predicate in _legal_relocate_targets rejected it.

The predicates come straight from pettingzoo_env.py:718-769, so the breakdown is
a decomposition of the real rule, not a re-implementation of it:
  surplus_ok(j)   idle(j) <= max(pending(j), 1)
  claim(j)        inbound claims this turn >= max(pending(j), 1)   (D34)
  demand(j)       pending(j) >= 1, or hops strictly closer to the nearest
                  demanded zone when no neighbour is demanded (D35)

The earlier version ran the greedy policy with a slice bug in its fallback and
offered a "relocate" arm that could never dispatch, so its served count said
nothing. This version uses the shared verified policies and reports paired
differences.

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
    n = env._num_zones
    hot = [args.zone_a, args.zone_b]
    policies = make_policies()
    arms = ["dispatch-only", "dispatch-first", "relocate-first-best",
            "relocate-only(BROKEN)"]
    seeds = [args.base_seed + i for i in range(args.episodes)]

    nbr = {z: sorted(env.env.region_neighbours[z]) for z in range(n)}
    nbr_set = {z: set(v) for z, v in nbr.items()}
    # The adjacency graph is DIRECTED: k-nearest-centre with ties broken by zone
    # index, so a zone j in nbr[i] does not imply i in nbr[j]. A relocation is
    # legal only if the DESTINATION is in the SOURCE's neighbour list
    # (pettingzoo_env.py:725-730, :805-808), so one-way edges are unreachable.
    npairs = n * (n - 1) // 2
    oneway = [(i, j) for i in range(n) for j in range(i + 1, n)
              if (j in nbr_set[i]) != (i in nbr_set[j])]
    print("adjacency graph: out-degree %d per zone over %d zones; %d/%d zone "
          "pairs (%.1f%%) are ONE-WAY"
          % (len(nbr[0]), n, len(oneway), npairs, 100.0 * len(oneway) / npairs))
    print("  e.g. zone %d lists %s; zone %s lists %s"
          % (args.zone_a, nbr[args.zone_a], args.zone_b, nbr[args.zone_b]))
    for a, b in ((args.zone_a, args.zone_b), (args.zone_b, args.zone_a)):
        print("  %d -> %d legal by adjacency: %s" % (a, b, b in nbr_set[a]))
    print("hotspots: A=%d B=%d  shared neighbours: %s"
          % (args.zone_a, args.zone_b,
             sorted(nbr_set[args.zone_a] & nbr_set[args.zone_b]) or "NONE"))
    print("  A neighbours: %s" % nbr[args.zone_a])
    print("  B neighbours: %s" % nbr[args.zone_b])
    cents = np.asarray(env.env.relocation_points, float)
    for a, b in ((args.zone_a, args.zone_b),):
        lon = (cents[a][0] - cents[b][0]) * 111.0 * np.cos(np.radians(cents[a][1]))
        lat = (cents[a][1] - cents[b][1]) * 111.0
        print("  A-B distance: %.2f km" % np.hypot(lon, lat))
    print("  seeds=%s (%d paired episodes per arm)\n" % (seeds, args.episodes))

    def pend(z: int) -> int:
        return int(round(env._pending_per_zone[z] * env._num_drivers))

    agg: dict[str, Counter] = {}
    per_ep_legal: dict[str, list[int]] = {}
    per_ep_sr: dict[str, list[float]] = {}
    for arm in arms:
        st = Counter()
        legal_per_ep: list[int] = []
        srs: list[float] = []
        for seed in seeds:
            env.reset(seed=seed)
            env.reset_observer()
            rng = np.random.default_rng(seed * 1_000_003 + 7)
            ep_legal = 0
            while env.agents:
                for h in hot:
                    if pend(h) < 1:
                        continue
                    st["hotspot_demand_turns"] += 1
                    if int(env._zone_idle_counts[h]) > 0:
                        st["car_present"] += 1
                        continue
                    # A donor is a zone that holds an idle car AND has h in its
                    # OWN neighbour list, which is the precondition for a legal
                    # relocation into h. Checking only h's out-edges would count
                    # zones that can never reach h.
                    donors = [z for z in range(n)
                              if int(env._zone_idle_counts[z]) > 0 and h in nbr_set[z]]
                    if not donors:
                        if any(int(env._zone_idle_counts[z]) > 0 for z in nbr[h]):
                            st["no_reachable_donor_adjacent"] += 1
                        else:
                            st["no_donor_adjacent"] += 1
                        continue
                    st["donor_adjacent"] += 1
                    can = [z for z in donors if h in env._legal_relocate_targets(z)]
                    if can:
                        st["legal_inbound"] += 1
                        ep_legal += 1
                    else:
                        st["BLOCKED"] += 1
                        # Decompose the real rule's predicates for one donor.
                        z = donors[0]
                        tgt_p = pend(h)
                        if int(env._zone_idle_counts[h]) > max(tgt_p, 1):
                            st["why_surplus"] += 1
                        elif (int(env._reloc_inbound_this_turn.get(h, 0))
                              >= max(tgt_p, 1)):
                            st["why_claim"] += 1
                        elif tgt_p >= 1:
                            st["why_other_despite_demand"] += 1
                        else:
                            st["why_no_demand"] += 1

                agent = env.agent_selection
                action, intended = policies[arm](env, agent, rng)
                action = verify_action(env, agent, action, intended)
                env.step(action)
                env.observe_turn()
            kpi = env.kpi_summary()
            st["served"] += int(kpi["served"])
            st["cancelled"] += int(kpi["cancelled"])
            st["empty_km"] += float(kpi["empty_km"])
            legal_per_ep.append(ep_legal)
            srs.append(100.0 * float(kpi["served"])
                       / max(float(kpi["served"]) + float(kpi["cancelled"]), 1.0))
        agg[arm] = st
        per_ep_legal[arm] = legal_per_ep
        per_ep_sr[arm] = srs

    def pc(x, d):
        return 100.0 * x / max(d, 1)

    for arm in arms:
        st = agg[arm]
        t = st["hotspot_demand_turns"]
        d = st["donor_adjacent"]
        b = st["BLOCKED"]
        print("=" * 92)
        print("%s   SR %.1f%%   served %d cancelled %d empty_km %.0f"
              % (arm, np.mean(per_ep_sr[arm]), st["served"], st["cancelled"],
                 st["empty_km"]))
        print("=" * 92)
        print("  hotspot-turns with unmet demand : %d" % t)
        print("    car already in the hotspot    : %5.1f%%" % pc(st["car_present"], t))
        print("    no adjacent zone held a car   : %5.1f%%" % pc(st["no_donor_adjacent"], t))
        print("    cars adjacent but NONE can legally reach it (one-way adjacency):"
              " %5.1f%%" % pc(st["no_reachable_donor_adjacent"], t))
        print("    a legal donor was adjacent    : %5.1f%%" % pc(d, t))
        print("      legal inbound move existed  : %5.1f%%  of donor turns"
              % pc(st["legal_inbound"], d))
        print("      BLOCKED                     : %5.1f%%  of donor turns"
              % pc(b, d))
        for k, lbl in (("why_surplus", "surplus_ok failed"),
                       ("why_claim", "D34 claim already full"),
                       ("why_other_despite_demand",
                        "demand present, unaccounted"),
                       ("why_no_demand", "no demand at hotspot")):
            if st[k]:
                print("        %-42s %5.1f%% of blocked" % (lbl, pc(st[k], b)))
        print()

    print("=" * 92)
    print("PAIRED, each arm vs dispatch-only (never relocates)")
    print("=" * 92)
    for key, lbl, arr in (("legal_inbound turns/ep", "legal inbound turns",
                           per_ep_legal),
                          ("service rate %", "service rate", per_ep_sr)):
        print("  -- %s" % key)
        for arm in arms:
            if arm == "dispatch-only":
                continue
            p = paired(arr[arm], arr["dispatch-only"], lbl)
            print("     %-24s %s" % (arm, fmt(p)))
        print()

    print("Note: 'legal inbound exists' is a property of the rule AND the policy")
    print("in force -- the same _legal_relocate_targets code offers different")
    print("menus depending on where the idle cars are, which the policy moves.")
    print("A high BLOCKED share under relocate-only(BROKEN) is that policy")
    print("churning the fleet out of the hotspots, not the env refusing cars.")


def fmt(p: dict) -> str:
    unit = "pp" if "rate" in p["label"] else ""
    return (f"{p['mean']:+9.2f}{unit} +/- {p['std']:6.2f} sd (sem {p['sem']:5.2f}, "
            f"n={p['n']}, {p['wins']}W/{p['losses']}L/{p['ties']}T)")


if __name__ == "__main__":
    main()