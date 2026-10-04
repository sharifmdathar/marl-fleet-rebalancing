"""Multi-objective evaluation: service rate, empty km and waiting time, together.

The bar for this project is not "higher service rate". Service rate can be bought
with deadhead -- `AGENTS.md` records a policy scoring +8.8pp over LP while being
2.4x worse per empty km. So a change is only an improvement if it moves service
rate, empty distance AND waiting time in the right direction *at the same time*.

Reports all three plus the efficiency view (service points per 100 empty km),
and flags explicitly whether a candidate beats the reference on all three, which
is the acceptance condition.

Paired: every candidate sees identical seeds and identical windows, so the
comparison is within-episode rather than between two means (`docs/eval-noise.md`).

    uv run python scripts/eval_tradeoff.py --baseline greedy --ippo <ck> --mappo <ck>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def run_episode(policy, cfg, seed: int) -> dict:
    env = RideGymAEC(cfg)
    mp = env._max_pending
    if hasattr(policy, "reset") and not hasattr(policy, "act"):
        policy.reset(seed=seed)
    env.reset(seed=seed)

    reloc = 0
    while env.agents:
        for _ in range(env._num_zones):
            if not env.agents:
                break
            ag = env.agent_selection
            m = env.action_mask(ag)
            if hasattr(policy, "act"):
                with torch.no_grad():
                    a, _, _ = policy.act(
                        obs_to_vector(env.observe(ag))[None, :], m[None, :],
                        deterministic=False)
                a = int(a[0])
            else:
                a = int(policy.select_action(ag, env, env.observe(ag)))
            if not m[a]:
                a = int(np.flatnonzero(m)[0])
            if a > mp:
                reloc += 1
            env.step(a)

    k = env.kpi_summary()
    served = int(k["served"])
    cancelled = int(k["cancelled"])
    terminal = served + cancelled
    return {
        "seed": seed,
        "served": served,
        "cancelled": cancelled,
        "service_rate": 100.0 * served / max(terminal, 1),
        "empty_km": float(k["empty_km"]),
        "avg_wait": float(k["avg_wait"]),
        "mean_trip_km": float(k.get("mean_trip_km", 0.0)),
        "reloc": reloc,
        "sr_per_100km": (100.0 * served / max(terminal, 1))
        / max(float(k["empty_km"]) / 100.0, 1e-9),
    }


def summarise(rows: list[dict]) -> dict:
    out = {}
    for f in ("service_rate", "empty_km", "avg_wait", "sr_per_100km",
              "reloc", "cancelled"):
        v = [r[f] for r in rows]
        out[f] = (float(np.mean(v)), float(np.std(v)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=12000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--baseline", default="greedy",
                    choices=["greedy", "lp"])
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--lp", action="store_true")
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split
    seeds = [args.base_seed + i for i in range(args.episodes)]

    arms: list[tuple[str, object]] = []
    if args.greedy:
        arms.append(("greedy", GreedyZoneDispatcher()))
    if args.lp:
        arms.append(("lp", ZoneLPRebalancer()))
    if args.ippo:
        l, _m, _ = load_trained_policy(Path(args.ippo))
        arms.append(("ippo", l))
    if args.mappo:
        l, _m, _ = load_trained_policy(Path(args.mappo))
        arms.append(("mappo", l))
    if not arms:
        print("pass --greedy/--lp/--ippo/--mappo")
        return

    print(f"paired over {args.episodes} identical seeds "
          f"({seeds[0]}..{seeds[-1]}), split={args.split}")
    print(f"alpha={cfg.alpha} beta={cfg.beta} gamma={cfg.gamma} "
          f"cancel={cfg.cancel_penalty}\n")

    print(f"{'arm':<8} {'SR%':>13} {'empty km':>13} {'avg wait':>13} "
          f"{'SR/100km':>11} {'reloc':>8} {'cancel':>8}")
    print("-" * 84)
    res = {}
    for name, policy in arms:
        rows = [run_episode(policy, cfg, s) for s in seeds]
        res[name] = summarise(rows)
        r = res[name]
        print(f"{name:<8} "
              f"{r['service_rate'][0]:6.2f}+-{r['service_rate'][1]:4.2f} "
              f"{r['empty_km'][0]:7.1f}+-{r['empty_km'][1]:5.1f} "
              f"{r['avg_wait'][0]:7.2f}+-{r['avg_wait'][1]:4.2f} "
              f"{r['sr_per_100km'][0]:11.1f} "
              f"{r['reloc'][0]:8.1f} {r['cancelled'][0]:8.1f}")

    base = args.baseline if args.baseline in res else list(res)[0]
    print()
    print("=" * 84)
    print(f"Acceptance vs {base}: must improve ALL THREE (SR up, empty km down,")
    print("avg wait down). Anything else is a trade, not an improvement.")
    print("=" * 84)
    b = res[base]
    for name in res:
        if name == base:
            continue
        r = res[name]
        sr = r["service_rate"][0] - b["service_rate"][0]
        ek = r["empty_km"][0] - b["empty_km"][0]
        wt = r["avg_wait"][0] - b["avg_wait"][0]
        allthree = (sr > 0) and (ek < 0) and (wt < 0)
        print(f"  {name:<8} SR {sr:+6.2f}pp | empty {ek:+7.1f}km | "
              f"wait {wt:+5.2f}min  -> "
              f"{'ALL THREE IMPROVE' if allthree else 'trade (not all three)'}")
        if not allthree:
            worse = []
            if sr <= 0:
                worse.append("SR")
            if ek >= 0:
                worse.append("empty km")
            if wt >= 0:
                worse.append("wait")
            print(f"           not improved: {', '.join(worse)}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"seeds": seeds, "split": args.split,
             "alpha": cfg.alpha, "results": res}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
