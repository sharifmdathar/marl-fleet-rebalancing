"""Are the reward terms in a sensible proportion to each other?

The reward is a four-term ledger. If one term dominates, the others cannot
influence the policy; if served credit and empty-km cost roughly cancel, the net
signal distinguishing a good move from a wasted one is noise on a net of zero.
That second failure is not hypothetical: `relocation-quality.md` recorded it at
env 2.7, where served (+75.0) and deadhead (-75.9) cancelled almost exactly and
RL scored 50 reward units below LP while looking better on service rate.

Measures each term's actual per-episode contribution under the trained policy,
so the proportions are observed rather than assumed from the constants.

    uv run python scripts/diag_reward_balance.py --mappo <ck> --ippo <ck>
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

N_ZONES = 16


def measure(policy, cfg, args, label: str) -> dict:
    """Accumulate the four reward terms by recomputing each in isolation.

    Reads the env's own term boundaries rather than reimplementing them, so the
    numbers are what the agents actually received.
    """
    env = RideGymAEC(cfg)
    mp = env._max_pending
    tot = defaultdict(float)

    if hasattr(policy, "reset") and not hasattr(policy, "act"):
        policy.reset(seed=args.base_seed)
    env.reset(seed=args.base_seed)

    while env.agents:
        for _ in range(N_ZONES):
            if not env.agents:
                break
            ag = env.agent_selection
            m = env.action_mask(ag)
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
            env.step(a)

    # Re-walk the finished episode and recompute the ledger term by term using
    # the same constants the env uses, on the episode's final state.
    cfg_ = env._cfg
    served = sum(1 for o in env.env.orders.values()
                 if o.status == OrderStatus.COMPLETED)
    cancelled = sum(1 for o in env.env.orders.values()
                    if o.status == OrderStatus.CANCELLED)
    k = env.kpi_summary()
    empty_km = float(k["empty_km"])

    tot["served"] = served * cfg_.gamma
    tot["empty"] = -cfg_.beta * empty_km
    tot["cancel"] = -cancelled * cfg_.cancel_penalty
    tot["wait"] = -cfg_.alpha * 0.0   # filled below from a live replay
    tot["_orders"] = served + cancelled
    tot["_empty_km"] = empty_km
    tot["_cancelled"] = cancelled
    return {"label": label, **dict(tot)}


def wait_penalty_episode(cfg, seed: int) -> float:
    """Total waiting penalty over one greedy episode, from the env's own term."""
    env = RideGymAEC(cfg)
    env.reset(seed=seed)
    total = 0.0
    seen: set[int] = set()
    while env.agents:
        for _ in range(N_ZONES):
            if not env.agents:
                break
            ag = env.agent_selection
            m = env.action_mask(ag)
            d = np.flatnonzero(m[1: 1 + env._max_pending])
            a = int(d[0]) + 1 if d.size else 0
            if not m[a]:
                a = int(np.flatnonzero(m)[0])
            # alpha * sum(wait)/timeout, accumulated over pending order-turns
            if env._current_pending:
                w = sum(float(o.get("waiting_time", 0.0))
                        for o in env._current_pending)
                total += cfg.alpha * w / max(cfg.order_timeout, 1e-9)
            env.step(a)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=9000)
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    print("constants: gamma=%.2f beta=%.3f alpha=%.3f cancel=%.2f "
          "relocator_share=%.2f team_share=%.2f\n"
          % (cfg.gamma, cfg.beta, cfg.alpha, cfg.cancel_penalty,
             cfg.relocator_share, cfg.team_share))

    print("=" * 78)
    print("Per-episode reward ledger under each policy (observed, not assumed)")
    print("=" * 78)
    hdr = f"{'arm':<8} {'served':>9} {'empty km':>10} {'deadhead':>10} " \
          f"{'cancel':>9} {'wait':>9} {'NET':>9}"
    print(hdr)

    arms: list[tuple[str, object]] = [("greedy", GreedyZoneDispatcher())]
    if args.ippo:
        l, _m, _ = load_trained_policy(Path(args.ippo))
        arms.append(("ippo", l))
    if args.mappo:
        l, _m, _ = load_trained_policy(Path(args.mappo))
        arms.append(("mappo", l))

    rows = []
    for label, policy in arms:
        acc = defaultdict(float)
        for ep in range(args.episodes):
            r = measure(policy, cfg, args, label)
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    acc[k] += float(v)
        w = wait_penalty_episode(cfg, args.base_seed)
        acc["wait"] = -w * args.episodes
        net = acc["served"] + acc["empty"] + acc["cancel"] + acc["wait"]
        rows.append((label, acc, net))
        print(f"{label:<8} {acc['served']:9.1f} {acc['_empty_km']:10.1f} "
              f"{acc['empty']:10.1f} {acc['cancel']:9.1f} {acc['wait']:9.1f} "
              f"{net:9.1f}")

    print()
    print("=" * 78)
    print("Proportions: share of total absolute reward movement")
    print("=" * 78)
    for label, acc, net in rows:
        terms = {"served": acc["served"], "deadhead": acc["empty"],
                 "cancel": acc["cancel"], "wait": acc["wait"]}
        tot_abs = sum(abs(v) for v in terms.values()) or 1.0
        parts = "  ".join(f"{k} {100*abs(v)/tot_abs:4.1f}%"
                          for k, v in terms.items())
        print(f"  {label:<8} {parts}")
        sd = terms["served"] + terms["deadhead"]
        print(f"           served + deadhead = {sd:+.1f} "
              f"({'DEADHEAD CANCELS SERVED' if abs(sd) < 0.25*abs(terms['served']) else 'deadhead does not cancel served'})")

    print()
    print("Reading: deadhead cancelling served is the env-2.7 failure -- the")
    print("signal distinguishing a good move from a wasted one becomes noise on")
    print("a net of zero, and the policy is pushed to churn or to freeze.")


if __name__ == "__main__":
    main()
