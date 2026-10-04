"""Trace relocations that actually happen, from the first turn a car is free.

The earlier version forced a relocation from turn 0 and never fired, because a
zone only has an idle car on ~8.8% of zone-turns. This waits for the first
genuine opportunity, commands a relocation at the farthest legal target, then
prints the driver state every turn until it stops being `relocating`.

That separates the candidate causes for a low aim/outcome agreement rate:
  * the driver is sent to a point belonging to a different zone
  * the driver arrives but `current_region` disagrees with the nearest centroid
  * the driver is dispatched to a rider mid-leg and never reaches the target
  * the leg completes and lands in the zone that was aimed at (no bug)

    uv run python scripts/diag_reloc_single_trace.py --max-traces 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-traces", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=8000)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    mp = env._max_pending
    n_zones = env._num_zones
    cents = np.asarray(env.env.relocation_points, dtype=float)

    env.reset(seed=args.base_seed)

    tracked: dict[int, int] = {}
    done = 0
    turn = 0
    n_agents = env._num_zones

    while env.agents and done < args.max_traces:
        # AEC steps ONE agent per env.step(); the env only advances the
        # simulation on the LAST agent of a turn. Checking arrival after a
        # single step therefore always reports the driver as still idle at its
        # origin -- which reads as "aimed at zone X, landed at zone Y" for
        # every move. Drive a whole turn per iteration.
        for _ in range(n_agents):
            if not env.agents:
                break
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            m = env.action_mask(ag)

            act = 0
            # Take the first opportunity to command a relocation to the farthest
            # legal target, so the trace is deterministic and spans the map.
            if not tracked and env._zone_idle_counts[zi] > 0:
                legal = env._legal_relocate_targets(zi)
                if legal:
                    far = max(legal, key=lambda j: float(np.linalg.norm(
                        cents[j] - cents[zi])))
                    cand = 1 + mp + far
                    if m[cand]:
                        idle = [d for d in range(env._num_drivers)
                                if env._last_obs[d]["self"].get("status") == "idle"
                                and max(0, min(int(env._last_obs[d]["self"]
                                                   .get("current_region", 0)),
                                               n_zones - 1)) == zi]
                        if idle:
                            did = idle[0]
                            tracked[did] = far
                            src = np.asarray(env._last_obs[did]["self"]
                                             ["location"], dtype=float)
                            print(f"\n=== turn {turn}: zone {zi} sends driver "
                                  f"{did} -> zone {far}")
                            print(f"    action index       : {cand}")
                            print(f"    from point         : {src}")
                            print(f"    target point       : "
                                  f"{np.asarray(env.env.relocation_points[far], float)}")
                            print(f"    distance to target : "
                                  f"{float(np.linalg.norm(np.asarray(env.env.relocation_points[far], float) - src)):,.0f} m")
                            act = cand

            if act == 0 or not m[act]:
                d = np.flatnonzero(m[1: 1 + mp])
                act = int(d[0]) + 1 if d.size else 0
            if not m[act]:
                act = int(np.flatnonzero(m)[0])
            env.step(act)
        turn += 1

        for did, want in list(tracked.items()):
            dd = env._last_obs[did]["self"]
            st = dd.get("status")
            loc = np.asarray(dd.get("location", [0.0, 0.0]), dtype=float)
            dist = np.linalg.norm(cents - loc[None, :], axis=1)
            nearest = int(np.argmin(dist))
            print(f"    t+{turn:3d} status={st:<12} "
                  f"region={dd.get('current_region')} nearest={nearest} "
                  f"dist_to_wanted={dist[want]:8.1f} m")
            if st != "relocating":
                landed = dd.get("current_region")
                print(f"    RESULT: wanted zone {want}, region={landed}, "
                      f"nearest_by_coords={nearest}")
                if landed == want:
                    print("    MATCH -- arrived in the zone the agent chose.")
                else:
                    tgt = np.asarray(env.env.relocation_points[want], float)
                    print(f"    MISMATCH -- landed in zone {landed}.")
                    print(f"    final location {loc}, "
                          f"{float(np.linalg.norm(loc - tgt)):,.0f} m from "
                          f"the intended target point")
                tracked.pop(did)
                done += 1

    print(f"\ntraced {done} relocation(s) over {turn} turns")


if __name__ == "__main__":
    main()
