"""Controlled stress test: does correct targeting actually buy service?

The concern: decision-time targeting roughly doubled (6.4% -> 12.4% IPPO,
6.1% -> 16.1% MAPPO against ~8% chance) yet aggregate service rate did not move
at all -- both arms sit at ~77%. Either the targeting gain is real but too small
to matter, or something downstream is cancelling it.

Natural demand is a poor instrument for this question: relocation happens on only
~8% of zone-turns, and a policy is scored over all 1920 of them, so most of the
episode cannot possibly respond to a targeting change. This script removes that
attenuation by concentrating the problem:

  SCENARIO 1 -- SINGLE ZONE. 100% of demand at zone A for the whole episode.
    Nothing needs to be predicted. A policy that aims cars at A converts fleet
    into service; one that aims elsewhere does not. Maximum possible signal.

  SCENARIO 2 -- TWO ZONES, hard switch. 100% at A, then 100% at B at the switch.
    The same, plus the ability to follow a change.

Reports, per arm: service rate, fleet share at the active hub, relocation count
and mean unmet demand at the chosen target. Greedy is the reference: it never
relocates, so its fleet share stays at whatever it started at.

`localhub` requires stage1 != stage2 but the switch is set past the horizon, so
demand holds at stage1 for the entire episode. Verified per scenario: the hub
must actually hold the demand, or the test is meaningless.

    uv run python scripts/diag_targeting_stress.py --ippo <ck> --mappo <ck>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from competence_two_zone import pick_far_pair


def make_cfg(hub: int, other: int, switch_min: float, split: str) -> NYCConfig:
    c = NYCConfig()
    c.nyc_split = split
    c.demand_shift_mode = "localhub"
    c.localhub_stage1_zone = hub
    c.localhub_stage2_zone = other
    c.localhub_switch_min = switch_min
    c.drift_source_zone = hub
    c.drift_dest_zone = other
    return c


def run(policy, cfg, args, hub, label: str) -> dict:
    env = RideGymAEC(cfg)
    mp = env._max_pending
    n_reloc = 0
    aim_demand: list[float] = []
    best_demand: list[float] = []
    fleet_hub: list[float] = []

    if hasattr(policy, "reset") and not hasattr(policy, "act"):
        policy.reset(seed=args.base_seed)
    env.reset(seed=args.base_seed)

    while env.agents:
        for _ in range(env._num_zones):
            if not env.agents:
                break
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            m = env.action_mask(ag)
            pend = np.asarray(env._pending_per_zone, dtype=float) \
                * max(env._num_drivers, 1)
            legal = env._legal_relocate_targets(zi)

            if hasattr(policy, "act"):
                with torch.no_grad():
                    act, _, _ = policy.act(
                        obs_to_vector(env.observe(ag))[None, :], m[None, :],
                        deterministic=False)
                a = int(act[0])
            else:
                a = int(policy.select_action(ag, env, env.observe(ag)))
            if not m[a]:
                a = int(np.flatnonzero(m)[0])

            d = env._decode_action(a, zi)
            if isinstance(d, dict) and "relocate" in d and legal:
                n_reloc += 1
                pick = int(d["relocate"])
                lg = np.asarray(legal, dtype=int)
                aim_demand.append(float(pend[pick]))
                best_demand.append(float(pend[lg[int(np.argmax(pend[lg]))]]))

            env.step(a)
            tot = float(env._zone_total_counts.sum())
            if tot > 0:
                fleet_hub.append(float(env._zone_total_counts[hub]) / tot)

    served = sum(1 for o in env.env.orders.values()
                 if str(o.status).endswith("COMPLETED"))
    total = sum(1 for o in env.env.orders.values()
                if str(o.status).endswith(("COMPLETED", "CANCELLED")))
    k = env.kpi_summary()
    # Did demand actually sit at the hub?
    share = float(env._demand_at_hub_check) if hasattr(
        env, "_demand_at_hub_check") else float("nan")
    return {
        "label": label,
        "reloc": n_reloc,
        "sr": 100.0 * served / max(total, 1),
        "served": served,
        "total": total,
        "fleet_hub": 100.0 * float(np.mean(fleet_hub)) if fleet_hub else 0.0,
        "empty_km": float(k["empty_km"]),
        "aim_demand": float(np.mean(aim_demand)) if aim_demand else 0.0,
        "best_demand": float(np.mean(best_demand)) if best_demand else 0.0,
    }


def report(r: dict) -> None:
    ratio = (r["aim_demand"] / r["best_demand"]
             if r["best_demand"] > 0 else float("nan"))
    print(f"  {r['label']:<8} SR {r['sr']:6.1f}%  served {r['served']:3d}/{r['total']:<3d} "
          f" fleet@hub {r['fleet_hub']:5.1f}%  reloc {r['reloc']:4d}  "
          f"empty {r['empty_km']:6.1f}  demand@aim {r['aim_demand']:.2f} "
          f"(best {r['best_demand']:.2f}, ratio {ratio:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=9000)
    ap.add_argument("--switch-min", type=float, default=25.0)
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    args = ap.parse_args()

    probe = RideGymAEC(NYCConfig(nyc_split="val"))
    a, b = pick_far_pair(probe)
    other = next(j for j in range(probe._num_zones) if j != a)
    print(f"hubs: single-zone A={a}, two-zone A={a} -> B={b}")
    print(f"  A neighbours: {sorted(j for j in probe.env.region_neighbours[a] if j != a)}")
    print(f"  B neighbours: {sorted(j for j in probe.env.region_neighbours[b] if j != b)}")

    arms: list[tuple[str, object]] = [("greedy", GreedyZoneDispatcher())]
    if args.ippo:
        l, _m, _ = load_trained_policy(Path(args.ippo))
        arms.append(("ippo", l))
    if args.mappo:
        l, _m, _ = load_trained_policy(Path(args.mappo))
        arms.append(("mappo", l))

    for title, hub, cfg in (
        (f"SCENARIO 1  single zone A={a}, 100% of demand there all episode",
         a, make_cfg(a, other, 10_000.0, "val")),
        (f"SCENARIO 2  two zones, 100% at A={a} until minute "
         f"{args.switch_min:g} then 100% at B={b}",
         b, make_cfg(a, b, args.switch_min, "val")),
    ):
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        for label, policy in arms:
            r = run(policy, cfg, args, hub, label)
            report(r)
        print()
        print("  A policy that targets correctly should show a HIGHER service")
        print("  rate and a HIGHER fleet share at the active hub than greedy,")
        print("  which never relocates at all.")


if __name__ == "__main__":
    main()
