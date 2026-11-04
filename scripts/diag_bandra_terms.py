"""Measure reward-term shares and relocation quality on Bandra.

The report's reward-share and targeting numbers were measured on NYC16, but
the Bandra policies train with different constants (beta=0.1, gamma=5.0) and a
different reward split, so they do not carry over. This measures the Bandra
analogues directly from a frozen checkpoint:

  - share of reward movement from served credit, relocation bonus, empty-km
    cost, waiting penalty and cancellation penalty;
  - relocation precision against the best legal target (and chance);
  - P(credit fires | relocation), the conditional term in the break-even;
  - empty km per relocation (cost basis) and mean relocation-leg distance.

Usage:
    uv run python scripts/diag_bandra_terms.py \
        --checkpoint runs/ippo-bandra-200-s0/checkpoint.pt --episodes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def run_episode(env, policy, seed: int) -> dict:
    from ride_gym.enums import OrderStatus

    cfg = env._cfg
    env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed=seed)

    n = dict(reloc=0, dispatch=0, fires=0, completions=0, cancellations=0)
    served = bonus = clawback = wait = cancel = 0.0
    empty_km = 0.0
    legs: list[float] = []
    leg_m: dict[int, float] = {}
    status: dict[int, str] = {}
    lvl = dict(n_legal=0, best_chosen=0, chance=0.0, n_targets=0,
               hit_any=0, global_best=0, d_pick=0.0, d_best=0.0)

    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        nonlocal served, bonus, clawback, wait, cancel, empty_km
        seen = set(env._seen_completed_ids)

        for did, ev in (events or {}).items():
            if ev.get("is_empty_move", False):
                km = float(ev.get("distance_moved", 0.0)) / 1000.0
                empty_km += km
                if status.get(did) == "relocating":
                    leg_m[did] = leg_m.get(did, 0.0) + km

        # Served credit and count are read without consuming the delta.
        new_completed = env._newly_completed_ids()
        n["completions"] += len(new_completed)
        served += cfg.gamma * len(new_completed)
        env._seen_completed_ids = seen

        # Relocator credit fires at dispatch.
        for _oid, did, zone_idx in (dispatched_this_turn or ()):
            relocator = env._driver_relocator.get(did)
            if relocator is not None and relocator != zone_idx \
                    and 0 <= relocator < env._num_zones:
                paid = cfg.relocator_share * cfg.gamma
                if paid > 0.0:
                    n["fires"] += 1
                    # In split mode the bonus is carved out of gamma, not added.
                    if not cfg.relocator_share_is_split:
                        bonus += paid

        # Cancelled this turn: clawback of relocation credit plus penalty.
        cancelled_now = {
            oid for oid, o in env.env.orders.items()
            if o.status == OrderStatus.CANCELLED
        }
        new_cancelled = cancelled_now - getattr(env, "_seen_cancelled_ids", set())
        n["cancellations"] += len(new_cancelled)
        for oid in new_cancelled:
            paid = env._order_relocator_paid.get(oid)
            if paid is not None and not cfg.relocator_share_is_split:
                clawback += paid[1]
        cancel += cfg.cancel_penalty * len(new_cancelled)

        # Waiting penalty for the turn.
        for o in getattr(env, "_current_pending", []):
            wait += cfg.alpha * float(o.get("waiting_time", 0.0)) / max(cfg.order_timeout, 1e-9)

        out = orig(events, dispatched_this_turn)

        for did in range(env._num_drivers):
            cur = env._last_obs[did]["self"].get("status")
            if status.get(did) == "relocating" and cur != "relocating":
                legs.append(leg_m.pop(did, 0.0))
            status[did] = cur
        return out

    env._compute_zone_rewards = traced

    while env.agents:
        agent = env.agent_selection
        zone = env._zone_int_id(agent)
        mask = env.action_mask(agent)
        pending = (env._pending_per_zone * env._num_drivers).copy()

        obs = obs_to_vector(env.observe(agent))
        with torch.no_grad():
            act, _, _ = policy.act(obs[None, :], mask[None, :], deterministic=True)
        action = int(act[0])
        if not mask[action]:
            action = int(np.flatnonzero(mask)[0])

        if action > env._max_pending:
            n["reloc"] += 1
            target = action - env._max_pending - 1
            legal = list(env._legal_relocate_targets(zone))
            if legal:
                lvl["n_legal"] += 1
                lvl["n_targets"] += len(legal)
                lvl["chance"] += 1.0 / len(legal)
                best = max(legal, key=lambda j: (pending[j], -j))
                lvl["d_pick"] += float(pending[target])
                lvl["d_best"] += float(pending[best])
                if target == best:
                    lvl["best_chosen"] += 1
                if pending[target] > 0:
                    lvl["hit_any"] += 1
                if target == int(np.argmax(pending)):
                    lvl["global_best"] += 1
        elif action > 0:
            n["dispatch"] += 1

        env.step(action)

    env._compute_zone_rewards = orig
    for did, open_leg in list(leg_m.items()):
        legs.append(open_leg)

    terms = {
        "served": served,
        "reloc_bonus": bonus,
        "empty_cost": cfg.beta * empty_km,
        "wait": wait,
        "cancel": cancel,
        "clawback": clawback,
    }
    total = sum(abs(v) for v in terms.values())
    shares = {k: (v / total if total > 0 else 0.0) for k, v in terms.items()}

    kpi = env.kpi_summary()
    sum_legs = float(np.sum(legs)) if legs else 0.0
    pickup_km = max(empty_km - sum_legs, 0.0)
    return {
        "seed": seed,
        "service_rate": 100.0 * kpi["served"] / max(kpi["total"], 1.0),
        "empty_km": float(kpi["empty_km"]),
        "relocs": n["reloc"],
        "dispatches": n["dispatch"],
        "completions": n["completions"],
        "cancellations": n["cancellations"],
        "credit_fires": n["fires"],
        "p_fire": n["fires"] / max(n["reloc"], 1),
        "km_per_reloc": empty_km / max(n["reloc"], 1),
        "mean_leg_km": float(np.mean(legs)) if legs else 0.0,
        "sum_leg_km": sum_legs,
        "pickup_km": pickup_km,
        "pickup_km_per_completion": pickup_km / max(n["completions"], 1),
        "km_total_per_reloc": (sum_legs + pickup_km) / max(n["reloc"], 1),
        "n_legs": len(legs),
        **{f"share_{k}": v for k, v in shares.items()},
        **{f"term_{k}": v for k, v in terms.items()},
        "precision": lvl["best_chosen"] / max(lvl["n_legal"], 1),
        "chance": lvl["chance"] / max(lvl["n_legal"], 1),
        "mean_targets": lvl["n_targets"] / max(lvl["n_legal"], 1),
        "hit_any_demand": lvl["hit_any"] / max(lvl["n_legal"], 1),
        "global_best": lvl["global_best"] / max(lvl["n_legal"], 1),
        "demand_ratio": (lvl["d_pick"] / lvl["d_best"]) if lvl["d_best"] > 0 else 0.0,
        "n_target": lvl["n_legal"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    policy, _manifest, env = load_trained_policy(Path(args.checkpoint))
    rows = [run_episode(env, policy, args.base_seed + i) for i in range(args.episodes)]

    print(f"checkpoint: {args.checkpoint}")
    print(f"config: alpha={env._cfg.alpha} beta={env._cfg.beta} "
          f"gamma={env._cfg.gamma} cancel={env._cfg.cancel_penalty} "
          f"share={env._cfg.relocator_share} split={env._cfg.relocator_share_is_split}")
    print(f"episodes: {args.episodes}, seeds {args.base_seed}..{args.base_seed + args.episodes - 1}\n")

    keys = ["service_rate", "empty_km", "relocs", "p_fire", "km_per_reloc",
            "mean_leg_km", "pickup_km_per_completion", "km_total_per_reloc",
            "mean_targets", "precision", "chance",
            "global_best", "hit_any_demand", "demand_ratio",
            "share_served", "share_reloc_bonus", "share_empty_cost",
            "share_wait", "share_cancel"]
    print(f"{'metric':<22}{'mean':>10}{'std':>10}")
    for k in keys:
        vals = [float(r[k]) for r in rows]
        print(f"{k:<22}{np.mean(vals):>10.3f}{np.std(vals, ddof=1):>10.3f}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": args.checkpoint, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()