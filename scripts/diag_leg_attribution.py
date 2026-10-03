"""Per-relocation attribution: was THIS move good, and what did it cost?

Motivation: every relocation metric so far is an aggregate -- "81.6% of legs went
to a demand-free zone", "80.3% churn". None of those say whether a particular
move paid off. Empty km is flat across beta 0.06/0.10/0.15 and across D37, so we
cannot tell whether the flatness means "all moves are equally useful" or "no
move is useful".

This tracks every individual relocation leg and closes it when the driver leaves
RELOCATING status, recording:

  leg               one idle -> relocating -> resolved cycle for one driver
  ended_serving     the driver then completed an order (the leg PAID)
  ended_idle        the driver went back to idle without serving (leg VOID)
  served_in_target  and specifically served an order originating in the zone
                    the leg was aimed at (leg paid AND aimed correctly)
  leg_km            deadhead attributed to this leg, from the leg's own events

Nothing here writes to env state: statuses are read from _last_obs, order
outcomes from env.env.orders, distances from the per-turn info events. No call
to _newly_completed_ids(), which mutates _seen_completed_ids.

    uv run python scripts/diag_leg_attribution.py --policy lp
    uv run python scripts/diag_leg_attribution.py \
        --checkpoint runs/d37-s0/checkpoint_best.pt
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

from marl_fleet_rebalancing.baselines import ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

_M_PER_KM = 1000.0


def run(kind, ckpt, episodes, base_seed):
    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    mp = env._max_pending
    n = env._num_zones
    cents = np.asarray(env.env.relocation_points, dtype=float)

    learner = None
    pol = None
    if kind == "checkpoint":
        learner, _m, _ = load_trained_policy(Path(ckpt))
    elif kind == "lp":
        pol = ZoneLPRebalancer()

    st = Counter()
    prev_status = {}
    open_leg = {}          # did -> dict(target=..., start_zone=..., t0=...)
    served_after = {}      # did -> zone the order originated from, if it served
    leg_records = []

    for ep in range(episodes):
        if pol is not None:
            pol.reset(seed=base_seed + ep)
        env.reset(seed=base_seed + ep)
        prev_status = {}
        open_leg = {}
        served_after = {}

        while env.agents:
            a = env.agent_selection
            z = env._zone_int_id(a)
            mask = env.action_mask(a)

            if learner is not None:
                obs = obs_to_vector(env.observe(a))
                with torch.no_grad():
                    act, _, _ = learner.act(obs[None, :], mask[None, :],
                                            deterministic=False)
                action = int(act[0])
            else:
                action = int(pol.select_action(a, env, env.observe(a)))
            if not mask[action]:
                action = int(np.flatnonzero(mask)[0])

            target = action - mp - 1 if action > mp else None
            if target is not None:
                pend = float(env._pending_per_zone[target] * env._num_drivers)
                st["cmds"] += 1
                st["cmd_demand_free"] += int(round(pend) < 1)
                st["cmd_demanded"] += int(round(pend) >= 1)

            env.step(action)

            # Read-only status tracking; close legs.
            for did in range(env._num_drivers):
                s = env._last_obs[did]["self"]
                status = s["status"]
                was = prev_status.get(did)
                prev_status[did] = status

                if status == "relocating" and was != "relocating":
                    open_leg[did] = {
                        "did": did,
                        "target": target if target is not None else None,
                        "start": z,
                        "t0": float(env.env.time),
                    }
                elif was == "relocating" and status != "relocating":
                    leg = open_leg.pop(did, None)
                    if leg is not None:
                        leg["t1"] = float(env.env.time)
                        leg["end_status"] = status
                        leg_records.append(leg)

        # Attribute completions to drivers. A LIST per driver, not a single
        # value: a driver can serve several orders (capacity 4), and keeping
        # only the last one silently mislabels every earlier leg.
        serve_events = []
        for o in env.env.orders.values():
            if o.status != OrderStatus.COMPLETED:
                continue
            did = getattr(o, "assigned_driver", None)
            if did is None:
                continue
            oz = int(np.argmin(((cents - np.asarray(o.origin, float)) ** 2).sum(1)))
            pt = o.pickup_time
            serve_events.append((did, oz, float(pt) if pt is not None else float("inf")))

        st["completions"] += len(serve_events)
        for leg in leg_records:
            st["legs_closed"] += 1
            # A leg "paid" if this driver picked up any order AFTER the leg
            # resolved. Counted by scanning all of the driver's serve events,
            # so multi-order drivers are handled correctly.
            after = [(oz, pt) for (d, oz, pt) in serve_events
                     if d == leg["did"] and pt >= leg["t1"] - 1e-6]
            if after:
                st["leg_paid"] += 1
                tgt = leg["target"]
                hit = int(any(tgt is not None and oz == tgt for oz, _pt in after))
                st["leg_paid_in_target"] += hit
                st["leg_paid_elsewhere"] += int(tgt is not None) - hit
            else:
                st["leg_void"] += 1
        leg_records = []

    L = max(st["legs_closed"], 1)
    return st, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lp", choices=["lp", "checkpoint"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    kind = "checkpoint" if args.checkpoint else args.policy
    print("running %s, %d episodes" % (args.checkpoint or args.policy,
                                       args.episodes))
    st, L = run(kind, args.checkpoint, args.episodes, args.base_seed)

    print("\nPER-RELOCATION ATTRIBUTION")
    print("  relocate commands issued   %6d  (%.1f%% to a demand-free zone)"
          % (st["cmds"], 100.0 * st["cmd_demand_free"] / max(st["cmds"], 1)))
    print("  legs closed (driver left RELOCATING) %6d" % st["legs_closed"])
    print("  legs that PAID (driver then served)  %6d  (%.1f%%)"
          % (st["leg_paid"], 100.0 * st["leg_paid"] / L))
    print("     served in the zone aimed at       %6d  (%.1f%% of legs)"
          % (st["leg_paid_in_target"], 100.0 * st["leg_paid_in_target"] / L))
    print("     served somewhere else             %6d  (%.1f%% of legs)"
          % (st["leg_paid_elsewhere"], 100.0 * st["leg_paid_elsewhere"] / L))
    print("  legs VOID (no service followed)      %6d  (%.1f%%)"
          % (st["leg_void"], 100.0 * st["leg_void"] / L))


if __name__ == "__main__":
    main()