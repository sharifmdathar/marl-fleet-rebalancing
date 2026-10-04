"""Reward-term sweep, judged on TARGETING first.

D58 rescaled alpha from 0.2 to 0.0132 on the measured magnitudes of the four
reward terms, and targeting fell from 55.06% to 20.95%. The reason is visible
only in hindsight: the waiting penalty is charged to the zone CONTAINING the
waiting rider, so it is the only term that distinguishes which zone needs a
car. Served credit is paid for any completion regardless of where the car came
from, so cutting the waiting weight removed most of the pressure to aim.

So this reports, for each candidate:

  PRIMARY   relocation precision against measured chance (argmax)
  SECONDARY service rate, empty km, average wait

A candidate that improves service rate while precision falls is a WORSE policy,
not a better one -- it serves riders by relocating indiscriminately, which is
what ec0.001 did (precision 11.65%, the worst measured, with the best service
rate of the sweep).

Every candidate is scored on identical seeds, and chance is recomputed per run
rather than assumed.

    uv run python scripts/diag_reward_sweep.py --runs al0.10=runs/al0.10-mappo-s0 \
        --runs al0.20=runs/al0.20-mappo-s0 --runs cur=runs/d59-mappo-s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def targeting(policy, cfg, seeds) -> dict:
    """Decision-time precision at argmax, plus the served-cost metrics."""
    env = RideGymAEC(cfg)
    mp = env._max_pending
    moves = hits = 0
    chance = 0.0
    sr_num = sr_den = 0
    empty = wait = 0.0
    served = canc = 0

    for sd in seeds:
        if hasattr(policy, "reset") and not hasattr(policy, "act"):
            policy.reset(seed=sd)
        env.reset(seed=sd)
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
                with torch.no_grad():
                    a, _, _ = policy.act(
                        obs_to_vector(env.observe(ag))[None, :], m[None, :],
                        deterministic=True)
                a = int(a[0])
                if not m[a]:
                    a = int(np.flatnonzero(m)[0])
                d = env._decode_action(a, zi)
                if isinstance(d, dict) and "relocate" in d and legal:
                    pick = int(d["relocate"])
                    lg = np.asarray(legal, dtype=int)
                    moves += 1
                    chance += 1.0 / len(legal)
                    if pick == int(lg[int(np.argmax(pend[lg]))]):
                        hits += 1
                env.step(a)
        k = env.kpi_summary()
        served += int(k["served"])
        canc += int(k["cancelled"])
        empty += float(k["empty_km"])
        wait += float(k["avg_wait"])

    n = len(seeds)
    return {
        "moves": moves,
        "precision": 100.0 * hits / max(moves, 1),
        "chance": 100.0 * chance / max(moves, 1),
        "service_rate": 100.0 * served / max(served + canc, 1),
        "empty_km": empty / n,
        "avg_wait": wait / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", default=[],
                    metavar="NAME=DIR", help="label:run-dir")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=8100)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split
    seeds = [args.base_seed + i for i in range(args.episodes)]

    print(f"paired over {len(seeds)} seeds, argmax actions, "
          f"precision is the PRIMARY metric\n")
    print(f"{'run':<10} {'alpha':>7} {'precision':>10} {'chance':>8} "
          f"{'lift':>6} {'SR%':>7} {'empty km':>10} {'wait':>7}")
    print("-" * 70)
    rows = []
    for spec in args.runs:
        nm, _, d = spec.partition("=")
        ck = Path(d) / "checkpoint_best.pt"
        if not ck.exists():
            print(f"{nm:<10} (no checkpoint)")
            continue
        alpha = json.loads((Path(d) / "config.json").read_text())
        alpha = alpha["env_config"]["alpha"]
        learner, _m, _ = load_trained_policy(ck)
        r = targeting(learner, cfg, seeds)
        r["alpha"] = alpha
        r["name"] = nm
        rows.append(r)
        lift = r["precision"] / max(r["chance"], 1e-9)
        print(f"{nm:<10} {alpha:>7.4f} {r['precision']:9.2f}% "
              f"{r['chance']:7.2f}% {lift:5.2f}x {r['service_rate']:6.2f}% "
              f"{r['empty_km']:10.1f} {r['avg_wait']:7.2f}")

    if len(rows) > 1:
        best = max(rows, key=lambda r: r["precision"])
        print()
        print("=" * 70)
        print(f"BEST TARGETING: {best['name']} "
              f"(alpha={best['alpha']}, {best['precision']:.2f}% vs "
              f"{best['chance']:.2f}% chance)")
        print("=" * 70)
        print("\nA candidate with high service rate but low precision is NOT")
        print("better -- it serves riders by relocating indiscriminately. The")
        print("waiting term is what distinguishes 'which zone needs a car', so")
        print("its weight is the thing under test here, not service rate.")


if __name__ == "__main__":
    main()