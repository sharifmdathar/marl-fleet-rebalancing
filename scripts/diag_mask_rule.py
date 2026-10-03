"""Audit of the relocation-legality rule: what does the env actually offer?

This predates the other diagnostics, and its "current rule" row had drifted out
of date: it re-implemented `_legal_relocate_targets` as "idle(j) <= pending(j)"
and called that the current rule, but the env has since gained D34 (intra-turn
claim, pettingzoo_env.py:752-757) and D35 (destination must have unmet demand,
:733-768). It also compared `env._pending_per_zone * num_drivers` without the
round() the env applies (:735-736), and always NOOP-ed, so the fleet
distribution was an artefact of a frozen world.

This version calls the env's OWN function for the current rule and re-implements
only the candidate alternatives, using the env's own rounding. It also records
how many of the env's targets were rejected by each candidate rule, so the
comparison is against a measured baseline rather than a remembered one.

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

from diag_common import ObservedEnv, make_policies, verify_action

from marl_fleet_rebalancing.config import NYCConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split

    env = ObservedEnv(cfg)
    policies = make_policies()
    seeds = [args.base_seed + i for i in range(args.episodes)]
    print("scenario: BASELINE nyc (demand_shift_mode=%s), split=%s, seeds=%s "
          "(%d paired episodes)" % (cfg.demand_shift_mode, args.split, seeds,
                                    args.episodes))
    print("env rule flags: require_destination_demand=%s claim_relocation_targets=%s "
          "neighbor_relocation_only=%s"
          % (cfg.require_destination_demand, cfg.claim_relocation_targets,
             cfg.neighbor_relocation_only))
    print("'current' is env._legal_relocate_targets() itself, called directly.\n")

    keys = ("current", "pending_ge1", "deficit", "any_neighbour")
    stats = {k: dict(offered=0, free=0, best_free=0, no_target=0, turns=0)
             for k in keys}
    rejected_by = Counter()
    admitted_by = Counter()
    pending_reachable = 0
    deficit_reachable = 0
    total_turns = 0
    zone_no_idle = 0
    srv = []

    for seed in seeds:
        env.reset(seed=seed)
        env.reset_observer()
        rng = np.random.default_rng(seed * 1_000_003 + 7)
        while env.agents:
            zone = env._zone_int_id(env.agent_selection)
            total_turns += 1
            # The env's own rounding (:735-736) -- using the raw float product
            # silently disagrees with the rule it is meant to describe.
            pending = np.asarray(
                [int(np.round(env._pending_per_zone[j] * env._num_drivers))
                 for j in range(env._num_zones)], dtype=np.int64)
            idle = np.asarray(env._zone_idle_counts)
            nbrs = [j for j in env.env.region_neighbours[zone] if j != zone]

            def record(key, legal):
                s = stats[key]
                s["turns"] += 1
                if not legal:
                    s["no_target"] += 1
                    return
                s["offered"] += len(legal)
                free = [j for j in legal if pending[j] <= 0]
                s["free"] += len(free)
                best = max(legal, key=lambda j: pending[j])
                if pending[best] <= 0:
                    s["best_free"] += 1

            # current: the env's own function, not a re-implementation
            cur = env._legal_relocate_targets(zone)
            record("current", cur)
            if not any(idle[j] > 0 for j in nbrs):
                zone_no_idle += 1
            # candidate A: destination must have at least one waiting rider,
            # keeping the env's surplus test and neighbour restriction
            record("pending_ge1", [j for j in nbrs
                                   if pending[j] >= 1
                                   and int(idle[j]) <= max(int(pending[j]), 1)])
            # candidate B: destination must have a real deficit
            record("deficit", [j for j in nbrs if int(idle[j]) < pending[j]])
            # candidate C: no restriction at all beyond adjacency
            record("any_neighbour", nbrs)

            # Cross-check: which candidates would have dropped each target.
            for j in cur:
                if pending[j] < 1:
                    rejected_by["no waiting rider"] += 1
                if int(idle[j]) >= pending[j]:
                    rejected_by["not in deficit"] += 1

            # Which branch of the rule admitted each target? pettingzoo_env.py
            # :758-768 has two: an outright demand test, and a multi-hop
            # "strictly closer to the nearest demand" fallback that admits a
            # demand-free zone. Reading the source's own admission condition is
            # what distinguishes them.
            hops = env._hops_to_nearest_demand() if cfg.require_destination_demand \
                else None
            nbrs_all = [j for j in env.env.region_neighbours[zone] if j != zone]
            # demanded_here as the env computes it (:742-744): a neighbour counts
            # only if it has a rider AND passes the surplus test.
            demand_here = any(pending[j] >= 1
                              and int(idle[j]) <= max(int(pending[j]), 1)
                              for j in nbrs_all)
            for j in cur:
                if pending[j] >= 1:
                    admitted_by["has a waiting rider"] += 1
                elif (hops is not None and not demand_here
                      and hops.get(j, 99) < hops.get(zone, 99)):
                    admitted_by["multi-hop approach, no rider there yet"] += 1
                else:
                    admitted_by["unclassified"] += 1

            if any(pending[j] >= 1 for j in nbrs):
                pending_reachable += 1
            if any(int(idle[j]) < pending[j] for j in nbrs):
                deficit_reachable += 1

            agent = env.agent_selection
            action, intended = policies["dispatch-first"](env, agent, rng)
            action = verify_action(env, agent, action, intended)
            env.step(action)
            env.observe_turn()
        kpi = env.kpi_summary()
        srv.append(100.0 * float(kpi["served"])
                   / max(float(kpi["served"]) + float(kpi["cancelled"]), 1.0))

    print("zone-turns sampled: %d over %d episodes (dispatch-first, SR %.1f%%)"
          % (total_turns, args.episodes, np.mean(srv)))
    print("turns where the zone held no idle car, so the rule returned [] by "
          "construction: %d (%.1f%%)" % (zone_no_idle, 100.0 * zone_no_idle / total_turns))
    print()
    print("%-14s %11s %13s %14s %13s"
          % ("rule", "targets/ep", "demand-free", "best-is-free", "no-target"))
    for key, label in (("current", "current (env)"),
                       ("pending_ge1", "pending>=1"),
                       ("deficit", "deficit"),
                       ("any_neighbour", "any neighbour")):
        s = stats[key]
        t = max(s["turns"], 1)
        print("%-14s %11.2f %12.1f%% %13.1f%% %12.1f%%"
              % (label, s["offered"] / t,
                 100 * s["free"] / max(s["offered"], 1),
                 100 * s["best_free"] / t,
                 100 * s["no_target"] / t))
    print()
    print("which branch of the rule admitted each of the env's %d offered targets:"
          % stats["current"]["offered"])
    for k, v in admitted_by.most_common():
        print("  %-44s %6d  (%.1f%%)"
              % (k, v, 100.0 * v / max(sum(admitted_by.values()), 1)))
    print()
    print("how the env's own menu compares to the stricter candidates:")
    print("  targets the env offered that 'pending>=1' would reject: %d"
          % rejected_by["no waiting rider"])
    print("  targets the env offered that 'deficit' would reject:      %d"
          % rejected_by["not in deficit"])
    print("  (of %d offered; a target can fail both)" % stats["current"]["offered"])
    print()
    print("turns where a neighbour had pending demand : %.1f%%"
          % (100 * pending_reachable / total_turns))
    print("turns where a neighbour had a real deficit  : %.1f%%"
          % (100 * deficit_reachable / total_turns))
    print()
    print("'best-is-free' is the share of turns where even the highest-demand")
    print("legal target has no waiting rider: the ceiling on relocation precision.")


if __name__ == "__main__":
    main()