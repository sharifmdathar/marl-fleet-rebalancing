"""Export MP4 videos of the two-zone demand-shift competence test.

The competence test moves 100% of demand from zone A to a far zone B partway
through an episode. Service rate says whether a policy coped; a video shows
whether cars physically moved, which is the thing the reward fixes (D47-D51)
were supposed to change.

Reuses the same env, policy loading and shift construction as
`competence_two_zone.py` so the video and the numbers describe one scenario.
The env has no `render()`, so frames come from `ride_gym.visualize`'s
`TrajectoryRecorder` and are drawn by
`marl_fleet_rebalancing.visualization.render_animation`.

Videos are git-ignored (`*.mp4`); they are debugging artifacts, not results.

    uv run python scripts/export_shift_videos.py \
        --ippo runs/d51-ippo-s0/checkpoint_best.pt \
        --mappo runs/d51-mappo-s0/checkpoint_best.pt \
        --outdir results/videos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.visualize import TrajectoryRecorder

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector
from marl_fleet_rebalancing.visualization import render_animation

# Reuse the hub selection and shift guard from the competence test verbatim.
# A local copy of pick_far_pair got the BFS wrong and returned ADJACENT zones
# (A=0, B=1), which would make the scenario trivially easy and the video
# meaningless.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from competence_two_zone import pick_far_pair, verify_shift as _verify_shift


def build_shift_cfg(args) -> tuple[NYCConfig, int, int]:
    """Identical construction to competence_two_zone.py.

    `localhub` is the mechanism that works: every order originates at stage1_zone
    before the switch and stage2_zone after, with nothing elsewhere. `drift`
    cannot do this -- it only moves orders that already originate in the source
    zone, so a first attempt with pct=0.9 produced almost no shift and every
    candidate scored the same.
    """
    cfg = NYCConfig()
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
    print(f"hubs: A=zone {a}  B=zone {b}")
    print(f"  A neighbours: {sorted(j for j in env0.env.region_neighbours[a] if j != a)}")
    print(f"  B neighbours: {sorted(j for j in env0.env.region_neighbours[b] if j != b)}")
    return cfg, a, b


def run_and_record(cfg, policy, args, a, b, out_path: Path) -> dict:
    env = RideGymAEC(cfg)
    mp = env._max_pending
    rec = TrajectoryRecorder()
    rec.relocation_points = list(getattr(env.env, "relocation_points", []))

    if hasattr(policy, "reset") and not hasattr(policy, "act"):
        policy.reset(seed=args.base_seed)

    env.reset(seed=args.base_seed)
    rec.snapshot(env.env)

    reloc = [0, 0]
    disp = [0, 0]
    last_time = float(env.env.time)
    n_frames = 0

    while env.agents:
        ag = env.agent_selection
        m = env.action_mask(ag)
        obs = obs_to_vector(env.observe(ag))
        if hasattr(policy, "act"):
            with torch.no_grad():
                act, _, _ = policy.act(obs[None, :], m[None, :],
                                       deterministic=False)
            a_val = int(act[0])
        else:
            a_val = int(policy.select_action(ag, env, env.observe(ag)))
        if not m[a_val]:
            a_val = int(np.flatnonzero(m)[0])

        phase = 0 if env.env.time < args.switch_min else 1
        if a_val > mp:
            reloc[phase] += 1
        elif a_val > 0:
            disp[phase] += 1

        env.step(a_val)

        if float(env.env.time) > last_time:
            rec.snapshot(env.env)
            last_time = float(env.env.time)
            n_frames += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(rec, out_path=str(out_path), fps=args.fps)

    k = env.kpi_summary()
    served = sum(1 for o in env.env.orders.values()
                 if str(o.status).endswith("COMPLETED"))
    total = sum(1 for o in env.env.orders.values()
                if str(o.status).endswith(("COMPLETED", "CANCELLED")))
    return {
        "frames": n_frames,
        "reloc_phase1": reloc[0],
        "reloc_phase2": reloc[1],
        "disp_phase1": disp[0],
        "disp_phase2": disp[1],
        "served": served,
        "terminal_orders": total,
        "service_rate": served / max(total, 1),
        "empty_km": float(k["empty_km"]),
        "path": str(out_path),
    }


def verify_shift(cfg, a, b, switch_min, base_seed) -> tuple[bool, float, float]:
    """Confirm demand actually sat at the two hubs, before trusting the video.

    A first version of the competence test used `demand_shift_mode="drift"`,
    whose generator only moves orders that ALREADY originate in the source zone.
    It produced almost no shift, every candidate scored the same, and the whole
    comparison was meaningless. Refuse to render in that case.
    """
    from competence_two_zone import verify_shift as _vs

    return _vs(cfg, a, b, switch_min, base_seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--greedy", action="store_true",
                    help="also export the greedy dispatcher as a reference")
    ap.add_argument("--outdir", default="results/videos")
    ap.add_argument("--base-seed", type=int, default=7000)
    ap.add_argument("--switch-min", type=float, default=25.0)
    ap.add_argument("--pct", type=float, default=90.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    cfg, a, b = build_shift_cfg(args)
    outdir = Path(args.outdir)

    arms: list[tuple[str, object]] = []
    if args.greedy:
        from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher
        arms.append(("greedy", GreedyZoneDispatcher()))
    if args.ippo:
        learner, _m, _ = load_trained_policy(Path(args.ippo))
        arms.append(("ippo", learner))
    if args.mappo:
        learner, _m, _ = load_trained_policy(Path(args.mappo))
        arms.append(("mappo", learner))

    if not arms:
        print("no arms requested; pass --ippo/--mappo/--greedy")
        return

    ok, share_a, share_b = _verify_shift(cfg, a, b, args.switch_min, args.base_seed)
    print(f"\nSHIFT CHECK: early share at A={share_a:.2f}, late share at B={share_b:.2f}")
    if not ok:
        print("*** DEMAND DID NOT CONCENTRATE AT THE HUBS. These videos would")
        print("*** show a scenario that never happened. Refusing to render.")
        return

    print(f"\n{'arm':<8} {'SR':>7} {'empty km':>10} {'reloc p1':>10} "
          f"{'reloc p2':>10} {'frames':>7}  path")
    for name, policy in arms:
        out = outdir / f"shift_{name}.mp4"
        stats = run_and_record(cfg, policy, args, a, b, out)
        print(f"{name:<8} {100*stats['service_rate']:6.1f}% "
              f"{stats['empty_km']:10.1f} {stats['reloc_phase1']:10d} "
              f"{stats['reloc_phase2']:10d} {stats['frames']:7d}  {out}")

    print(f"\ndemand: 100% at zone {a} until minute {args.switch_min:g}, "
          f"then 100% at zone {b}")
    print("phase 1 = before the switch, phase 2 = after. "
          "reloc counts are relocation ACTIONS taken in each phase.")


if __name__ == "__main__":
    main()
