"""What ends a relocation leg: arrival at the target, or a dispatch en route?

The aim/outcome probe showed cars landing in zones other than the one aimed at.
The per-turn trace shows why: the driver's status leaves `relocating` by turning
`to_pickup`, i.e. it was dispatched to a rider mid-journey
(`allow_relocating_dispatch=True`, D14). The leg never reached its destination.

That matters for two separate reasons:

  1. MEASUREMENT. If most legs are cut short, then comparing the zone an agent
     aimed at against the zone its car ended up in does not measure targeting at
     all -- it measures where an unrelated rider happened to be. Any precision
     figure computed that way is confounded.

  2. REWARD. D48 pays the relocator when its car is DISPATCHED, so an
     interrupted leg still earns the full credit while having flown only part of
     the distance (D50 finding 3). This script measures how often that happens,
     which is the share of relocations where the credit is earned early.

Classifies every relocation leg by how it ended.

    uv run python scripts/diag_reloc_leg_outcome.py --episodes 3
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
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--follow", action="store_true",
                    help="force relocations to trace leg outcomes")
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    mp = env._max_pending
    n_zones = env._num_zones
    n_agents = env._num_zones
    pts = np.asarray(env.env.relocation_points, dtype=float)

    outcome = Counter()
    leg_km: dict[int, float] = {}
    leg_km_at_end: list[float] = []

    env.reset(seed=args.base_seed)
    watched: dict[int, int] = {}

    while env.agents:
        # One AEC step per agent; a turn completes after all agents act.
        for _ in range(n_agents):
            if not env.agents:
                break
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            m = env.action_mask(ag)

            act = 0
            if args.follow and not watched and env._zone_idle_counts[zi] > 0:
                legal = env._legal_relocate_targets(zi)
                if legal:
                    far = max(legal, key=lambda j: float(np.linalg.norm(
                        pts[j] - pts[zi])))
                    cand = 1 + mp + far
                    if m[cand]:
                        idle = [d for d in range(env._num_drivers)
                                if env._last_obs[d]["self"].get("status") == "idle"
                                and max(0, min(int(env._last_obs[d]["self"]
                                                   .get("current_region", 0)),
                                               n_zones - 1)) == zi]
                        if idle:
                            watched[idle[0]] = far
                            act = cand
            if act == 0 or not m[act]:
                d = np.flatnonzero(m[1: 1 + mp])
                act = int(d[0]) + 1 if d.size else 0
            if not m[act]:
                act = int(np.flatnonzero(m)[0])
            env.step(act)

        # Classify legs that ended this turn.
        for did, want in list(watched.items()):
            dd = env._last_obs[did]["self"]
            st = dd.get("status")
            if st == "relocating":
                continue
            loc = np.asarray(dd.get("location", [0.0, 0.0]), dtype=float)
            nearest = int(np.argmin(np.linalg.norm(pts - loc[None, :], axis=1)))
            if st == "to_pickup":
                outcome["dispatched_en_route"] += 1
            elif st == "idle":
                outcome["arrived"] += 1
                outcome["arrived_in_target" if dd.get("current_region") == want
                        else "arrived_elsewhere"] += 1
            else:
                outcome[f"other_{st}"] += 1
            leg_km_at_end.append(leg_km.get(did, 0.0))
            watched.pop(did)

    total = sum(v for k, v in outcome.items()
                if k in ("arrived", "dispatched_en_route"))
    print("=" * 70)
    print("How relocation legs end")
    print("=" * 70)
    if total == 0:
        print("no legs observed (pass --follow to force some)")
        return
    for k, v in sorted(outcome.items()):
        print(f"  {k:<24} {v:5d}  ({100*v/total:5.1f}%)")
    print()
    interrupted = outcome["dispatched_en_route"]
    print(f"legs cut short by an opportunistic dispatch: "
          f"{interrupted}/{total} ({100*interrupted/total:.1f}%)")
    print()
    if interrupted > total / 2:
        print("  VERDICT: most legs do NOT reach their destination. A car is")
        print("  dispatched to a rider mid-journey (D14 allow_relocating_dispatch).")
        print("  So intended-zone vs landed-zone does NOT measure targeting:")
        print("  the landing zone reflects where a rider was, not where the")
        print("  policy aimed. Any precision figure built on that comparison is")
        print("  confounded, and D48 pays full credit for the interrupted leg.")
    else:
        print("  VERDICT: most legs complete. Landed zone is a fair readout of aim.")


if __name__ == "__main__":
    main()
