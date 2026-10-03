"""Is the pending>=1 rule stranding cars that should be closing in?

A hotspot zone may not be a direct neighbour of the car's zone. Under
pending>=1 the car can only move when the destination itself has unmet
demand, so a car k hops from a hotspot cannot start moving. This measures how
often that happens, and compares candidate rules:

  pending   : destination has unmet demand (current)
  closer    : destination is strictly closer to unmet demand than the origin
  either    : pending OR closer (allows multi-hop approach to a hotspot)
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


def bfs_hops(nbrs, src, targets):
    """Hop distance from src to each target, or None."""
    dist = {src: 0}
    q = deque([src])
    found = {}
    while q:
        cur = q.popleft()
        for j in nbrs[cur]:
            if j in dist:
                continue
            dist[j] = dist[cur] + 1
            if j in targets:
                found[j] = dist[j]
            q.append(j)
    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    n_zones = env._num_zones
    nbrs = {z: [n for n in env.env.region_neighbours[z] if n != z] for z in range(n_zones)}

    stats = {k: dict(offered=0, free=0, turns=0, none=0) for k in ("pending", "closer", "either")}
    stranded = 0
    turns_with_idle = 0
    total = 0

    for ep in range(args.episodes):
        env.reset(seed=args.base_seed + ep)
        while env.agents:
            zone = env._zone_int_id(env.agent_selection)
            pending = env._pending_per_zone * env._num_drivers
            idle = env._zone_idle_counts
            total += 1
            if idle[zone] > 0:
                turns_with_idle += 1
                demand_zones = {j for j in range(n_zones) if pending[j] >= 1}
                # Car has a spare; is any demanded zone reachable in one hop?
                if demand_zones and not (demand_zones & set(nbrs[zone])):
                    # ...but is one reachable at all (2+ hops)?
                    if any(bfs_hops(nbrs, zone, demand_zones).get(z, 99) < 99
                           for z in demand_zones):
                        stranded += 1

                if demand_zones:
                    # Hop distance from EVERY zone to the nearest demanded zone.
                    # Keying this by demanded zones only (the first attempt)
                    # silently reduced the "closer" rule back to "pending".
                    d2d = {}
                    for z in range(n_zones):
                        if pending[z] >= 1:
                            d2d[z] = 0
                    frontier = deque(d2d)
                    while frontier:
                        cur = frontier.popleft()
                        for j in nbrs[cur]:
                            if j not in d2d:
                                d2d[j] = d2d[cur] + 1
                                frontier.append(j)
                    here = d2d.get(zone, 99)
                    for key in stats:
                        if key == "pending":
                            legal = [j for j in nbrs[zone]
                                     if pending[j] >= 1 and idle[j] <= max(int(round(pending[j])), 1)]
                        elif key == "closer":
                            legal = [j for j in nbrs[zone]
                                     if d2d.get(j, 99) < here
                                     and idle[j] <= max(int(round(pending[j])), 1)]
                        else:
                            legal = [j for j in nbrs[zone]
                                     if (pending[j] >= 1 or d2d.get(j, 99) < here)
                                     and idle[j] <= max(int(round(pending[j])), 1)]
                        s = stats[key]
                        s["turns"] += 1
                        if not legal:
                            s["none"] += 1
                        else:
                            s["offered"] += len(legal)
                            s["free"] += sum(1 for j in legal if pending[j] <= 0)

            m = env.action_mask(env.agent_selection)
            env.step(int(np.flatnonzero(m)[0]) if m.any() else 0)

    print("turns: %d   with an idle car: %d\n" % (total, turns_with_idle))
    print("  turns where a spare car exists but NO neighbour has demand,")
    print("  while a demanded zone is 2+ hops away (stranded): %d  (%.1f%% of spare-car turns)"
          % (stranded, 100 * stranded / max(turns_with_idle, 1)))
    print()
    print("  %-10s %11s %13s %13s" % ("rule", "targets/ep", "demand-free", "no-target%"))
    for key in ("pending", "closer", "either"):
        s = stats[key]
        t = max(s["turns"], 1)
        print("  %-10s %11.1f %12.1f%% %12.1f%%"
              % (key, s["offered"] / t, 100 * s["free"] / max(s["offered"], 1),
                 100 * s["none"] / t))

if __name__ == "__main__":
    main()
