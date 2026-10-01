"""Interactive two-hotspot scenario: pick 2 zones, concentrate demand, compare.

Prompts for the two zones, the concentration share and the split, then runs
every policy on the identical set of windows and reports PAIRED differences
per window (see docs/eval-noise.md), plus a matched video per policy.

    uv run python scripts/hotspot_picker.py --checkpoint <ckpt>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector
from marl_fleet_rebalancing.visualization import TrajectoryRecorder, render_animation


def km(cents, a, b):
    lon = (cents[a][0] - cents[b][0]) * 111.0 * np.cos(np.radians(cents[a][1]))
    lat = (cents[a][1] - cents[b][1]) * 111.0
    return float(np.hypot(lon, lat))


def ask_int(label, lo, hi, default):
    while True:
        raw = input(f"{label} [{lo}-{hi}, default {default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  whole number, please")
            continue
        if lo <= v <= hi:
            return v
        print(f"  must be {lo}-{hi}")


def ask_float(label, lo, hi, default):
    while True:
        raw = input(f"{label} [{lo}-{hi}, default {default}]: ").strip()
        if not raw:
            return default
        try:
            v = float(raw)
        except ValueError:
            print("  number, please")
            continue
        if lo <= v <= hi:
            return v
        print(f"  must be {lo}-{hi}")


def episode(env, policy, seed, stochastic, record_frames=False):
    env.reset(seed=seed)
    rec = TrajectoryRecorder() if record_frames else None
    if rec is not None:
        rec.relocation_points = list(env.env.relocation_points)
        rec.target_zone = env._cfg.hotspot_zone_a
        rec.target_zone_pct = env._cfg.hotspot_pct
    last = -1.0
    reloc = total = 0
    served = cancelled = 0
    is_learner = hasattr(policy, "act")
    while env.agents:
        agent = env.agent_selection
        obs = obs_to_vector(env.observe(agent))
        mask = env.action_mask(agent)
        if is_learner:
            with torch.no_grad():
                act, _, _ = policy.act(
                    obs[None, :], mask[None, :], deterministic=not stochastic,
                )
            action = int(act[0])
        else:
            action = int(policy.select_action(agent, env, obs))
        if not mask[action]:
            action = int(np.flatnonzero(mask)[0])
        total += 1
        if action > env._max_pending:
            reloc += 1
        env.step(action)
        if rec is not None and env.env.time > last:
            rec.snapshot(env.env)
            last = env.env.time
    k = env.kpi_summary()
    served, cancelled = k["served"], k["cancelled"]
    return rec, {
        "sr": 100.0 * served / max(served + cancelled, 1.0),
        "wait": k["avg_wait"],
        "empty": k["empty_km"],
        "reloc_pct": 100.0 * reloc / max(total, 1),
        "served": served,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--out-dir", default="/home/prash/college/minor/logs/videos")
    ap.add_argument("--tag", default="hot2")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    base = NYCConfig()
    probe = RideGymAEC(base)
    cents = np.asarray(probe.env.relocation_points, float)
    ids = json.loads(Path(base.nyc_splits_dir, "manifest.json").read_text())["zone_ids"]
    nbr = {z: sorted(probe.env.region_neighbours[z]) for z in range(probe._num_zones)}

    # Baseline demand per zone, so the user can see which are already busy.
    probe.reset(seed=args.base_seed)
    orders = probe.env.order_generator.generate()
    counts = np.zeros(probe._num_zones)
    for o in orders:
        d = ((np.asarray([o.origin]) - cents[:, None, :]) ** 2).sum(-1)[0]
        counts[int(d.argmin())] += 1

    print("zones in the 16-zone study (env numbering)\n")
    print("  zone  taxi-id    lon       lat   base-orders  neighbours")
    for z in range(probe._num_zones):
        star = " << busy" if counts[z] >= counts.max() else ""
        print("  %4d  %7d  %8.4f %8.4f   %11.0f  %s%s"
              % (z, ids[z], cents[z][0], cents[z][1], counts[z], nbr[z][:6], star))

    print("\n--- two-hotspot scenario ---")
    a = ask_int("hotspot zone A", 0, probe._num_zones - 1, 8)
    b = ask_int("hotspot zone B", 0, probe._num_zones - 1, 13)
    if a == b:
        print("  A and B must differ; using B=%d" % (0 if a != 0 else 1))
        b = 0 if a != 0 else 1
    pct = ask_float("share of orders concentrated into A+B (%)", 10, 100, 60.0)
    split = ask_float("fraction of that share going to A (0.5 = even)", 0.0, 1.0, 0.5)
    orders_n = ask_int("orders per window", 40, 400, base.nyc_window_limit)

    cfg = NYCConfig()
    cfg.demand_shift_mode = "hotspot2"
    cfg.hotspot_zone_a = a
    cfg.hotspot_zone_b = b
    cfg.hotspot_pct = pct
    cfg.hotspot_split = split
    cfg.nyc_window_limit = orders_n

    shared = set(nbr[a]) & set(nbr[b])
    print("\n  A=%d  B=%d  A-B %.2f km  shared neighbours: %s"
          % (a, b, km(cents, a, b), sorted(shared) or "none"))
    if not shared:
        print("  NOTE: A and B share no neighbour, so a car cannot serve both")
        print("        by a single hop. This is the harder case.")

    env = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))
    greedy = GreedyZoneDispatcher()
    lp = ZoneLPRebalancer()

    policies = [
        ("greedy", greedy, True, False),
        ("lp", lp, True, False),
        ("rl-argmax", policy, False, True),
        ("rl-sampled", policy, True, True),
    ]
    rows: dict[str, list[dict]] = {}
    for name, pol, stochastic, is_rl in policies:
        out = []
        for i in range(args.episodes):
            if not is_rl:
                pol.reset(seed=args.base_seed + i)
            _rec, kpi = episode(env, pol, args.base_seed + i, stochastic)
            out.append(kpi)
        rows[name] = out

    seeds = [args.base_seed + i for i in range(args.episodes)]
    print("\nper-window service rate (%), paired on identical windows\n")
    print("seed   " + "".join(f"{n:>12}" for n in rows))
    for i, s in enumerate(seeds):
        print(f"{s}   " + "".join(f"{rows[n][i]['sr']:>12.1f}" for n in rows))

    lp_sr = np.array([r["sr"] for r in rows["lp"]])
    print(f"\n{'':22}" + "".join(f"{n:>12}" for n in ("SR mean", "vs LP", "won")))
    for name in rows:
        v = np.array([r["sr"] for r in rows[name]])
        d = v - lp_sr
        wins = int((d > 0).sum())
        print(f"  {name:20s}{v.mean():>12.2f}{d.mean():>+11.2f}pp  {wins}/{len(d)}")
    print("\nother KPIs (mean over windows)")
    print(f"  {'':20s}{'wait':>12}{'empty km':>12}{'reloc%':>12}")
    for name in rows:
        print("  %-20s%12.2f%12.0f%12.1f"
              % (name, np.mean([r["wait"] for r in rows[name]]),
                 np.mean([r["empty"] for r in rows[name]]),
                 np.mean([r["reloc_pct"] for r in rows[name]])))

    if args.no_video:
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print()
    for name, pol, stochastic, is_rl in policies:
        if not is_rl:
            pol.reset(seed=args.base_seed)
        rec, kpi = episode(env, pol, args.base_seed, stochastic, record_frames=True)
        if rec is None:
            continue
        path = out / f"nyc16_{args.tag}_{name}_seed{args.base_seed}.mp4"
        render_animation(
            rec, out_path=str(path), fps=10,
            description=(f"{name} - hotspots z{a}/z{b} at {pct:.0f}%, "
                         f"{orders_n} orders - SR {kpi['sr']:.1f}%"),
        )
        print(f"  -> {path}")


if __name__ == "__main__":
    main()