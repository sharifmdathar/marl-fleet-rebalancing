"""Paired per-window comparison of every policy we have, on identical windows.

Answers the question the unpaired means cannot: is the learned policy actually
better than the baselines, and is the argmax/sampling gap the reason it looks
like it is not?
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


def run_episode(env, policy, seed, stochastic):
    """One episode. `policy` is an IPPO learner (has .act) or a scripted
    baseline (has .select_action)."""
    env.reset(seed=seed)
    reloc = 0
    total = 0
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
    k = env.kpi_summary()
    served, cancelled = k["served"], k["cancelled"]
    return {
        "sr": 100.0 * served / max(served + cancelled, 1.0),
        "wait": k["avg_wait"],
        "empty": k["empty_km"],
        "reloc_pct": 100.0 * reloc / max(total, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    from marl_fleet_rebalancing.baselines import ZoneLPRebalancer

    env = RideGymAEC(NYCConfig())
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))

    results: dict[str, list[dict]] = {}
    lp = ZoneLPRebalancer()
    for name in ("LP", "RL-argmax", "RL-sampled"):
        rows = []
        for i in range(args.episodes):
            seed = args.base_seed + i
            if name == "LP":
                lp.reset(seed=seed)
                rows.append(run_episode(env, lp, seed, stochastic=True))
            else:
                rows.append(
                    run_episode(env, policy, seed, stochastic=(name == "RL-sampled"))
                )
        results[name] = rows

    seeds = [args.base_seed + i for i in range(args.episodes)]
    print(f"per-window service rate (%), seeds {seeds[0]}-{seeds[-1]}\n")
    print("seed   " + "".join(f"{n:>12}" for n in results))
    for i, seed in enumerate(seeds):
        print(f"{seed}   " + "".join(f"{results[n][i]['sr']:>12.1f}" for n in results))
    print()

    print("%-28s %8s %8s %9s" % ("metric", "mean", "std", "vs LP"))
    for metric, unit in (("sr", "pp"), ("wait", "min"), ("empty", "km"), ("reloc_pct", "%")):
        lp_v = np.array([r[metric] for r in results["LP"]])
        print(f"{'LP ' + metric:28s} {lp_v.mean():>8.2f} {lp_v.std():>8.2f}")
        for name in ("RL-argmax", "RL-sampled"):
            v = np.array([r[metric] for r in results[name]])
            d = v - lp_v
            wins = int((d > 0).sum()) if metric in ("sr", "empty") else int((d < 0).sum())
            sign = "better" if metric in ("sr", "empty") else "better"
            print(
                f"{name + ' ' + metric:28s} {v.mean():>8.2f} {v.std():>8.2f} "
                f"{d.mean():>+7.2f}{unit}  {wins}/{len(d)} windows {sign}"
            )
        print()

    out = Path("runs/paired_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
