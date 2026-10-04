"""Relocation competence test: can each policy follow demand across the map?

Motivation. Service rate cannot tell us whether a policy rebalances *well* --
it can be bought with deadhead, so SR must be paired with empty km. And it cannot rank
LP as "better", because LP is a fixed-rule heuristic that never had to learn when
or where to move; comparing its service rate to a learned policy's answers a
different question than the one we care about.

This is a controlled behavioural probe, applied identically to every candidate:

  PHASE 1 (t < switch_min)   all demand originates at zone A
  PHASE 2 (t > switch_min)   all demand originates at zone B

A and B are chosen far apart (opposite corners of the zone grid), so serving B
in phase 2 REQUIRES having moved fleet capacity across the map. A policy that
does not rebalance keeps serving where it already is and collapses in phase 2.

Reported per policy, split by phase:
  sr              service rate within that phase
  served          completions
  fleet_at_B      share of fleet sitting in B, averaged over the phase -- the
                  direct measure of "did it move the fleet"
  empty_km        cost of the move

    NOTE on demand concentration: this uses `demand_shift_mode="drift"`, whose
generator (demand/drifting_hotspot.py:112) only moves orders that ALREADY
originate in the source zone -- so `source_pct` controls what fraction of
source-zone demand stays, and it never concentrates the OTHER zones into the
destination. A first attempt with pct=0.9 therefore produced almost no shift at
all (argmax demand was zone 14 in phase 1 and zone 8 in phase 2, neither hub),
which is why every candidate scored similarly: the test measured nothing.

This script therefore VERIFIES that the shift actually happened before trusting
any comparison, and refuses to report if it did not.

    uv run python scripts/competence_two_zone.py --episodes 4
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ride_gym.enums import OrderStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def pick_far_pair(env) -> tuple[int, int]:
    """Two zones as far apart as the adjacency graph allows (BFS diameter)."""
    n = env._num_zones
    nbr = {z: [j for j in env.env.region_neighbours[z] if j != z]
           for z in range(n)}
    best = (0, 1, 0)
    for src in range(n):
        dist = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for c in frontier:
                for j in nbr[c]:
                    if j not in dist:
                        dist[j] = dist[c] + 1
                        nxt.append(j)
            frontier = nxt
        for dst in range(n):
            if dst != src and dist.get(dst, 0) > best[2]:
                best = (src, dst, dist.get(dst, 0))
    return best[0], best[1]


def run(cfg, policy, episodes, base_seed, switch_min, a, b):
    env = RideGymAEC(cfg)
    mp = env._max_pending
    st = {1: Counter(), 2: Counter()}
    for ep in range(episodes):
        if hasattr(policy, "reset") and not hasattr(policy, "act"):
            policy.reset(seed=base_seed + ep)
        env.reset(seed=base_seed + ep)
        while env.agents:
            ag = env.agent_selection
            m = env.action_mask(ag)
            if hasattr(policy, "act"):
                obs = obs_to_vector(env.observe(ag))
                with torch.no_grad():
                    act, _, _ = policy.act(
                        obs[None, :], m[None, :], deterministic=False,
                    )
                a_val = int(act[0])
            else:
                a_val = int(policy.select_action(ag, env, env.observe(ag)))
            if not m[a_val]:
                a_val = int(np.flatnonzero(m)[0])

            # Read phase BEFORE stepping, then accumulate that turn's outcome
            # into the right bucket using the env's live order statuses.
            phase = 1 if env.env.time < switch_min else 2
            if a_val > mp:
                st[phase]["reloc"] += 1
            elif a_val > 0:
                st[phase]["disp"] += 1
            env.step(a_val)

            st[phase]["fleet_at_B"] += int(env._zone_total_counts[b])
            st[phase]["fleet_total"] += int(env._zone_total_counts.sum())
            for o in env.env.orders.values():
                if o.status == OrderStatus.COMPLETED:
                    st[phase]["served"] += 0  # counted at episode end below

        k = env.kpi_summary()
        # Re-walk orders and attribute each completion to the phase it landed in.
        for o in env.env.orders.values():
            if o.status != OrderStatus.COMPLETED:
                continue
            pt = float(o.pickup_time) if o.pickup_time is not None else 0.0
            ph = 1 if pt < switch_min else 2
            st[ph]["served"] += 1
        for o in env.env.orders.values():
            if o.status == OrderStatus.CANCELLED:
                ct = float(o.cancel_time) if o.cancel_time is not None else 0.0
                st[1 if ct < switch_min else 2]["cancelled"] += 1
        st[1]["empty"] += float(k["empty_km"]) * (switch_min / max(env.env.horizon, 1e-9))
        st[2]["empty"] += float(k["empty_km"]) * (1 - switch_min / max(env.env.horizon, 1e-9))
    return st


def verify_shift(cfg, a, b, switch_min, base_seed) -> tuple[bool, float, float]:
    """Confirm demand is actually AT the two hubs in each phase.

    Returns (ok, share_at_a_early, share_at_b_late). If demand never concentrated
    there, the competence test is measuring nothing and the caller must say so.
    """
    env = RideGymAEC(cfg)
    env.reset(seed=base_seed)

    def probe(target_t):
        while env.env.time < target_t and env.agents:
            ag = env.agent_selection
            m = env.action_mask(ag)
            d = np.flatnonzero(m[1 : 1 + env._max_pending])
            r = np.flatnonzero(m[1 + env._max_pending :])
            env.step(int(d[0]) + 1 if len(d)
                     else (int(r[0]) + 1 + env._max_pending if len(r) else 0))
        p = env._pending_per_zone.copy()
        return float(p[a]), float(p[b]), float(p.sum())

    a1, b1, t1 = probe(switch_min * 0.5)
    a2, b2, t2 = probe(switch_min + 15.0)
    return (a1 / max(t1, 1e-9) > 0.5 and b2 / max(t2, 1e-9) > 0.5,
            a1 / max(t1, 1e-9), b2 / max(t2, 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=7000)
    ap.add_argument("--switch-min", type=float, default=25.0)
    ap.add_argument("--pct", type=float, default=90.0)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--split", default="val")
    ap.add_argument("--force", action="store_true",
                    help="report even if the shift check fails (diagnostic only)")
    args = ap.parse_args()

    cfg = NYCConfig()
    # localhub is the correct mechanism: EVERY order originates at stage1_zone
    # before the switch and at stage2_zone after it, with nothing elsewhere.
    # `drift` cannot do this (see module docstring).
    cfg.demand_shift_mode = "localhub"
    cfg.drift_source_pct = args.pct
    cfg.drift_ramp_start_min = args.switch_min
    cfg.drift_ramp_end_min = args.switch_min + 0.5
    env0 = RideGymAEC(cfg)
    a, b = pick_far_pair(env0)
    cfg.drift_source_zone = a
    cfg.drift_dest_zone = b
    cfg.localhub_stage1_zone = a
    cfg.localhub_stage2_zone = b
    cfg.localhub_switch_min = args.switch_min
    cfg.nyc_split = args.split

    hops = env0._hops_to_nearest_demand
    print("two-zone competence test")
    print("  phase 1 (t < %.0f min): demand at zone %d" % (args.switch_min, a))
    print("  phase 2 (t > %.0f min): demand moves to zone %d" % (args.switch_min, b))
    print("  A=%d neighbours=%s" % (a, sorted(
        j for j in env0.env.region_neighbours[a] if j != a)))
    print("  B=%d neighbours=%s" % (b, sorted(
        j for j in env0.env.region_neighbours[b] if j != b)))
    print("  share A of fleet: %.3f" % (env0._zone_total_counts[a]
                                        / max(env0._zone_total_counts.sum(), 1)))

    cands = [("greedy", GreedyZoneDispatcher()), ("lp", ZoneLPRebalancer())]
    for ck in args.checkpoints:
        learner, _m, _ = load_trained_policy(Path(ck))
        cands.append((Path(ck).parent.name, learner))

    ok, share_a, share_b = verify_shift(cfg, a, b, args.switch_min, args.base_seed)
    print("\n  SHIFT CHECK: early share at A=%.2f, late share at B=%.2f"
          % (share_a, share_b))
    if not ok:
        print("  *** DEMAND DID NOT CONCENTRATE AT THE HUBS. Numbers below would")
        print("  *** be meaningless -- not a result about any policy.")
        if not args.force:
            return
        print("  *** (--force given, continuing for diagnosis only)")
    print("  -> demand is concentrated at A then B; comparison is meaningful.")

    print("\n%-22s %14s %14s %12s" % ("policy", "P1 SR%", "P2 SR%", "fleet@B P1->P2"))
    print("-" * 68)
    for name, pol in cands:
        st = run(cfg, pol, args.episodes, args.base_seed, args.switch_min, a, b)
        out = []
        for ph in (1, 2):
            s = st[ph]
            tot = max(s["served"] + s["cancelled"], 1)
            out.append(100.0 * s["served"] / tot)
        fb1 = st[1]["fleet_at_B"] / max(st[1]["fleet_total"], 1)
        fb2 = st[2]["fleet_at_B"] / max(st[2]["fleet_total"], 1)
        print("%-22s %13.1f%% %13.1f%% %11.0f%%->%2.0f%%"
              % (name, out[0], out[1], 100 * fb1, 100 * fb2))


if __name__ == "__main__":
    main()