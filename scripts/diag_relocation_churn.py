"""Why does RL not beat LP on efficiency? Measure churn, don't guess.

Established (env 3.0, 12 paired windows):

    greedy            51.5% SR    72.6 km   70.9 SR/100km   reloc  0.0%
    LP                70.8% SR   159.6 km   44.4 SR/100km   reloc  3.3%
    RL sampled        81.3% SR   452.8 km   18.0 SR/100km   reloc  9.4%

RL relocates ~2.9x as often as LP and drives 2.8x the distance. This measures
WHERE those extra relocations go, via the per-zone relocation flow matrix:

    gross_legs      every relocate action executed
    churn_legs      min(inflow_z, outflow_z) summed over z -- a car sent INTO
                    zone z that is matched by zone z sending one of its own
                    cars OUT. The fleet ends where it started; only the
                    distance was spent.
    net_arrivals    gross - churn: legs that actually change the distribution

    wasted_target   legs whose destination had no unmet demand at all

    uv run python scripts/diag_relocation_churn.py --policy lp
    uv run python scripts/diag_relocation_churn.py --checkpoint runs/beta0.10-s0/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.baselines import ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def run(policy_kind, ckpt, episodes, base_seed):
    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    mp = env._max_pending
    n = env._num_zones

    learner = None
    pol = None
    if policy_kind == "checkpoint":
        learner, _m, _ = load_trained_policy(Path(ckpt))
    elif policy_kind == "lp":
        pol = ZoneLPRebalancer()

    gross = churn = wasted = 0
    served = cancelled = 0
    empty_km = 0.0
    per_ep = []

    for ep in range(episodes):
        if pol is not None:
            pol.reset(seed=base_seed + ep)
        env.reset(seed=base_seed + ep)
        inflow = Counter()
        outflow = Counter()
        ep_gross = 0
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

            if action > mp:
                target = action - mp - 1
                outflow[z] += 1
                inflow[target] += 1
                ep_gross += 1
                pend = float(env._pending_per_zone[target] * env._num_drivers)
                if round(pend) < 1:
                    wasted += 1

            env.step(action)

        c = sum(min(inflow[z], outflow[z]) for z in range(n))
        gross += ep_gross
        churn += c
        k = env.kpi_summary()
        served += int(k["served"])
        cancelled += int(k["cancelled"])
        empty_km += float(k["empty_km"])
        per_ep.append((ep_gross, c))

    tot = max(served + cancelled, 1)
    return {
        "gross": gross / episodes,
        "churn": churn / episodes,
        "net": (gross - churn) / episodes,
        "churn_pct": 100.0 * churn / max(gross, 1),
        "wasted_pct": 100.0 * wasted / max(gross, 1),
        "sr": 100.0 * served / tot,
        "empty_km": empty_km / episodes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="lp", choices=["lp", "checkpoint", "greedy"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    kind = "checkpoint" if args.checkpoint else args.policy
    r = run(kind, args.checkpoint, args.episodes, args.base_seed)
    label = args.checkpoint or args.policy

    print("\nrelocation flow decomposition (%d episodes, env 3.0)" % args.episodes)
    print("  policy                    %s" % label)
    print("  gross legs/ep             %8.1f" % r["gross"])
    print("  churn legs/ep             %8.1f   (%.1f%% of gross)"
          % (r["churn"], r["churn_pct"]))
    print("  net-changing legs/ep      %8.1f" % r["net"])
    print("  legs to a demand-free zone %7.1f%%" % r["wasted_pct"])
    print("  service rate              %7.1f%%" % r["sr"])
    print("  empty km/ep               %8.1f" % r["empty_km"])
    print("  SR per 100 empty km       %8.1f"
          % (100 * r["sr"] / max(r["empty_km"], 1e-9)))
    print("  km per net-changing leg   %8.2f"
          % (r["empty_km"] / max(r["net"], 1e-9)))
    print("  km per gross leg          %8.2f"
          % (r["empty_km"] / max(r["gross"], 1e-9)))


if __name__ == "__main__":
    main()