"""Is the gap to LP a reward problem or a learning problem?

Scores every arm on the SAME held-out windows (one episode per window, so the
whole split is covered) and reports the three KPIs next to the episode reward.

The reward is summed over all zones, so the per-zone credit split cancels and
what is left is the team objective. With the defaults (`--relocator-share 0
--targeting-bonus 0`) that is INTENT's R = gamma*served - beta*empty_km -
alpha*wait - cancel. Pass the training values to see the reward the policy was
actually trained on. Behaviour is identical either way: the policies are frozen
and act by argmax, so the reward config cannot change the trajectory.

Reading it:
  LP beats RL on INTENT R  and on the training reward -> RL has not found a
      policy the reward already prefers: a LEARNING problem.
  LP beats RL on INTENT R  but loses on the training reward -> the shaping pays
      for something the objective does not want: a REWARD problem.

`NAME+noreloc` re-runs an RL arm with its relocation actions masked off, which
shows what its relocations buy.

    uv run python scripts/diag_objective_gap.py --greedy --lp \
        --arm trained=runs/<run>/checkpoint.pt --noreloc trained
"""
from __future__ import annotations

import argparse
import copy
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

FIELDS = ("service_rate", "empty_km", "avg_wait", "cancel_rate", "reloc", "reward")


def run_episode(policy, cfg, seed: int, allow_reloc: bool = True) -> dict:
    env = RideGymAEC(cfg)
    if hasattr(policy, "reset") and not hasattr(policy, "act"):
        policy.reset(seed=seed)
    env.reset(seed=seed)

    # Call-through wrapper: records what the env computed, computes nothing itself.
    total = {"reward": 0.0}
    inner = env._compute_zone_rewards

    def recording(*args, **kwargs):
        out = inner(*args, **kwargs)
        total["reward"] += float(np.sum(out))
        return out

    env._compute_zone_rewards = recording

    mp = env._max_pending
    reloc = 0
    while env.agents:
        ag = env.agent_selection
        m = env.action_mask(ag)
        if not allow_reloc:
            m = m.copy()
            m[mp + 1:] = False
        if hasattr(policy, "act"):
            with torch.no_grad():
                a, _, _ = policy.act(
                    obs_to_vector(env.observe(ag))[None, :], m[None, :],
                    deterministic=True)
            a = int(a[0])
        else:
            a = int(policy.select_action(ag, env, env.observe(ag)))
        if not m[a]:
            raise RuntimeError(f"{ag} chose masked action {a}")
        if a > mp:
            reloc += 1
        env.step(a)

    k = env.kpi_summary()
    n = max(int(k["total"]), 1)
    return {
        "seed": seed,
        "service_rate": 100.0 * k["served"] / n,
        "cancel_rate": 100.0 * k["cancelled"] / n,
        "empty_km": float(k["empty_km"]),
        "avg_wait": float(k["avg_wait"]),
        "reloc": reloc,
        "reward": total["reward"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--episodes", type=int, default=12,
                    help="one per held-out window; 12 covers the val split")
    ap.add_argument("--base-seed", type=int, default=12000)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--lp", action="store_true")
    ap.add_argument("--lp-tuned", type=float, nargs=2, default=None,
                    metavar=("LOOKAHEAD", "PENALTY"),
                    help="add an LP arm with these lookahead_minutes and deficit_penalty_scale")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--noreloc", action="append", default=[], metavar="NAME",
                    help="also run this arm with relocation masked off")
    ap.add_argument("--relocator-share", type=float, default=0.0)
    ap.add_argument("--targeting-bonus", type=float, default=0.0)
    ap.add_argument("--reference", default="lp")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split
    cfg.relocator_share = args.relocator_share
    cfg.targeting_bonus = args.targeting_bonus
    seeds = [args.base_seed + i for i in range(args.episodes)]

    arms: list[tuple[str, object, bool]] = []
    arm_cfg: dict[str, NYCConfig] = {}
    if args.greedy:
        arms.append(("greedy", GreedyZoneDispatcher(), True))
    if args.lp:
        arms.append(("lp", ZoneLPRebalancer(), True))
    if args.lp_tuned:
        from marl_fleet_rebalancing.baselines.lp_baseline import LPBaselineConfig
        arms.append(("lp_tuned", ZoneLPRebalancer(LPBaselineConfig(
            lookahead_minutes=args.lp_tuned[0],
            deficit_penalty_scale=args.lp_tuned[1])), True))
    for spec in args.arm:
        name, _, path = spec.partition("=")
        learner, manifest, _ = load_trained_policy(Path(path))
        arms.append((name, learner, True))
        # Evaluate each policy under the action space it was trained with.
        own = copy.deepcopy(cfg)
        own.relocation_nearest_k = int(
            manifest.get("env_config", {}).get("relocation_nearest_k", 0))
        own.obs_inbound = bool(manifest.get("env_config", {}).get("obs_inbound", False))
        arm_cfg[name] = arm_cfg[name + "+noreloc"] = own
        if name in args.noreloc:
            arms.append((name + "+noreloc", learner, False))

    rows = {name: [run_episode(p, arm_cfg.get(name, cfg), s, allow) for s in seeds]
            for name, p, allow in arms}

    print(f"split={args.split}, {args.episodes} paired episodes "
          f"(seeds {seeds[0]}..{seeds[-1]}), argmax actions")
    print(f"reward: gamma={cfg.gamma} beta={cfg.beta} alpha={cfg.alpha} "
          f"cancel={cfg.cancel_penalty} relocator_share={cfg.relocator_share} "
          f"targeting_bonus={cfg.targeting_bonus}")
    print(f"\n{'arm':<20}{'SR%':>13}{'empty km':>15}{'wait min':>12}"
          f"{'cancel%':>10}{'reloc':>8}{'reward/ep':>14}")
    print("-" * 92)
    for name, _p, _a in arms:
        r = rows[name]
        mean = {f: float(np.mean([e[f] for e in r])) for f in FIELDS}
        sd = {f: float(np.std([e[f] for e in r], ddof=1)) for f in FIELDS}
        print(f"{name:<20}{mean['service_rate']:7.1f}±{sd['service_rate']:<5.1f}"
              f"{mean['empty_km']:9.1f}±{sd['empty_km']:<5.1f}"
              f"{mean['avg_wait']:8.2f}±{sd['avg_wait']:<4.2f}"
              f"{mean['cancel_rate']:9.1f}{mean['reloc']:8.1f}"
              f"{mean['reward']:9.1f}±{sd['reward']:<5.1f}")

    if args.reference in rows:
        ref = rows[args.reference]
        print(f"\nPaired difference vs {args.reference} (same window, same seed): "
              "mean ± standard error")
        for name, _p, _a in arms:
            if name == args.reference:
                continue
            line = f"  {name:<18}"
            for f, label in (("service_rate", "SR"), ("empty_km", "km"),
                             ("avg_wait", "wait"), ("reward", "reward")):
                d = np.array([a[f] - b[f] for a, b in zip(rows[name], ref)])
                se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
                line += f"{label} {d.mean():+7.2f}±{se:<5.2f} "
            print(line)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"split": args.split, "seeds": seeds,
             "relocator_share": cfg.relocator_share,
             "targeting_bonus": cfg.targeting_bonus,
             "arms": {n: str(getattr(p, "__class__", type(p)).__name__)
                      for n, p, _a in arms},
             "rows": rows}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
