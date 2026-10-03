"""Relocation payback ledger: does a relocation ever pay, and does it pay in time?

The oracle-forecast probe answers *whether* a policy can aim at future demand.
It cannot answer whether aiming would have been worth it. Relocation is charged
`-beta * empty_km` on the turn it moves and paid `relocator_share * gamma /
len(chain)` = 0.45 only when the moved car is dispatched AND completes a ride --
many turns later, and only if it ever happens. This script prices each
individual relocation leg so those two facts separate:

    "relocation never pays"      vs    "relocation pays, but too late to matter"

which the aggregate precision number alone cannot distinguish.

Per leg:
    cost              beta * empty_km charged for that move (and, over the leg's
                      life, every subsequent empty move by the same car)
    turns_to_payback  turns from the move until that car's dispatched order
                      completes; None if it never completes one
    never_paid_share  fraction of legs with no payback inside the episode

Discipline (see AGENTS.md and scripts/diag_relocation_conversion.py):
  * the env's own bookkeeping is never used to observe these facts. Cost is read
    from a SNAPSHOT of the `events` dict taken before `_compute_zone_rewards`
    consumes it, and completion is detected by polling order statuses. Calling
    `_newly_completed_ids()` would mutate `_seen_completed_ids` and has already
    produced one wrong report in this repo.
  * legs are identified by diffing `_driver_relocation_target` across the turn,
    so the driver actually picked by `_flush_turn`'s idle queue is the one
    recorded -- not an assumed one.

    uv run python scripts/diag_relocation_roi.py --episodes 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

from ride_gym.enums import OrderStatus

_M_PER_KM = 1000.0


def traced_ledger(env: RideGymAEC) -> dict:
    """Wrap `_compute_zone_rewards` to snapshot the events dict it is handed.

    The snapshot must be taken on entry: `_compute_zone_rewards` is what charges
    the empty-distance penalty out of these events, so afterwards they are
    spent. Nothing here mutates env state.

    The snapshot is consumed exactly ONCE. `_compute_zone_rewards` fires once
    per TURN, but the AEC loop calls `env.step()` once per AGENT -- 16 agents
    per turn. Leaving the snapshot readable across those 16 steps would
    charge the same turn's empty distance 16 times, which is what inflated
    the first ledger's cost by that factor. `snap_id` marks the snapshot's
    generation so the caller can tell a fresh turn from a re-read.
    """
    state: dict = {"turn": 0, "snap": {}, "snap_id": 0, "legs": [], "open": {}}
    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        state["snap"] = {int(did): dict(ev) for did, ev in events.items()}
        state["snap_id"] += 1
        return orig(events, dispatched_this_turn)

    env._compute_zone_rewards = traced
    return state


def run_episode(env, policy, seed, ledger) -> dict:
    """Run one episode and return its leg records plus the served count."""
    env.reset(seed=seed)
    ledger["turn"] = 0
    ledger["snap"] = {}
    ledger["snap_id"] = 0
    ledger["legs"] = []
    last_snap_id = -1
    # did -> legs still awaiting payback, oldest first. A car can be re-tasked
    # before it pays, so this is a queue, not a single slot.
    ledger["open"] = {}
    seen_completed: set[int] = set()
    beta = float(env._cfg.beta)
    norm_km = float(getattr(env._cfg, "empty_distance_norm_km", 1.0) or 1.0)
    served = 0
    empty_km_total = 0.0

    while env.agents:
        ag = env.agent_selection
        m = env.action_mask(ag)
        vec = obs_to_vector(env.observe(ag))
        with torch.no_grad():
            act, _, _ = policy.act(vec[None, :], m[None, :], deterministic=False)
        a = int(act[0])
        if not m[a]:
            a = int(np.flatnonzero(m)[0])

        before = dict(getattr(env, "_driver_relocation_target", {}) or {})
        env.step(a)
        after = dict(getattr(env, "_driver_relocation_target", {}) or {})

        # New legs: a driver gained a target, or changed it (a relay hop).
        # `_driver_relocation_target` is only mutated at turn flush, so this
        # diff fires once per turn, not once per agent step.
        for did, tgt in after.items():
            if before.get(did) != tgt:
                leg = {
                    "driver": int(did),
                    "target": int(tgt),
                    "turn": ledger["turn"],
                    "cost": 0.0,
                    "paid_turn": None,
                }
                ledger["legs"].append(leg)
                ledger["open"].setdefault(int(did), []).append(leg)

        seen_id = ledger["snap_id"]
        if seen_id == last_snap_id:
            continue
        last_snap_id = seen_id

        # Charge ONLY genuine relocation travel. A leg must stop absorbing cost
        # the moment its car is dispatched, otherwise it silently picks up the
        # pickup deadhead of a ride it had nothing to do with -- which is not a
        # relocation cost and is not what the env charges the relocator for.
        for did, ev in ledger["snap"].items():
            if not ev.get("is_empty_move", False):
                continue
            km = float(ev.get("distance_moved", 0.0)) / _M_PER_KM / max(norm_km, 1e-9)
            empty_km_total += km
            d = (env._last_obs or {}).get(int(did))
            if d is None or d["self"].get("status") != "relocating":
                continue
            pending = ledger["open"].get(int(did))
            if pending:
                pending[-1]["cost"] += beta * km

        # Payback: an order this car was assigned has COMPLETED. Polled, never
        # read out of the env's consumed completion bookkeeping.
        for oid, o in env.env.orders.items():
            if o.status != OrderStatus.COMPLETED or oid in seen_completed:
                continue
            seen_completed.add(oid)
            served += 1
            did = getattr(o, "assigned_driver", None)
            if did is None:
                continue
            queue = ledger["open"].get(int(did))
            if queue:
                queue.pop(0)["paid_turn"] = ledger["turn"]

        ledger["turn"] += 1

    return {
        "seed": seed,
        "served": served,
        "legs": ledger["legs"],
        "empty_km_total": empty_km_total,
        # The env's own tally of ALL empty distance. Cross-check: my per-turn
        # sum must match it, or the snapshot is being read more (or less) often
        # than once per turn and every cost figure here is wrong.
        "env_empty_km": float(env._episode_empty_m / _M_PER_KM),
        "horizon_turns": ledger["turn"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=7000)
    ap.add_argument("--checkpoint", default="runs/d43-ippo-s0/checkpoint_best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="results/relocation_roi/relocation_roi.json")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    policy, manifest, _ = load_trained_policy(ckpt)
    print(f"relocation payback ledger -- {ckpt.parent.name}")
    print(f"  beta={manifest['env_config']['beta']} "
          f"relocator_share={manifest['env_config']['relocator_share']} "
          f"gamma={manifest['env_config']['gamma']}")
    print(f"  episode turns={manifest['episode_turns']} drivers={manifest['num_drivers']}")

    cfg = NYCConfig(**{k: v for k, v in manifest["env_config"].items()
                       if hasattr(NYCConfig, k)})
    cfg.nyc_split = args.split
    env = RideGymAEC(cfg)
    ledger = traced_ledger(env)

    per_ep = []
    all_legs: list[dict] = []
    horizon = 0
    for i in range(args.episodes):
        ep = run_episode(env, policy, args.base_seed + i, ledger)
        legs = ep["legs"]
        paid = [g for g in legs if g["paid_turn"] is not None]
        all_legs.extend(legs)
        horizon = ep["horizon_turns"]
        per_ep.append({
            "seed": ep["seed"],
            "served": ep["served"],
            "legs": len(legs),
            "legs_paid": len(paid),
            "never_paid": len(legs) - len(paid),
            "cost_sum": sum(g["cost"] for g in legs),
            "empty_km_ledger": ep["empty_km_total"],
            "empty_km_env": ep["env_empty_km"],
        })
        print(f"  ep {ep['seed']}: served={ep['served']:3d} legs={len(legs):3d} "
              f"paid={len(paid):3d} never_paid={len(legs) - len(paid):3d} "
              f"cost={sum(g['cost'] for g in legs):.3f} "
              f"empty_km ledger={ep['empty_km_total']:.1f} "
              f"env={ep['env_empty_km']:.1f}")
        # Independent check on the snapshot discipline: the ledger's total
        # empty km must reproduce the env's own tally for the same episode.
        drift = abs(ep["empty_km_total"] - ep["env_empty_km"])
        if drift > 0.5:
            raise AssertionError(
                f"empty-km accounting drifted by {drift:.2f} km "
                f"(ledger {ep['empty_km_total']:.1f} vs env {ep['env_empty_km']:.1f}); "
                "the per-turn snapshot is not being read exactly once per turn"
            )

    legs_n = max(len(all_legs), 1)
    paid_legs = [g for g in all_legs if g["paid_turn"] is not None]
    ttp = [g["paid_turn"] - g["turn"] for g in paid_legs]
    mean_ttp = float(np.mean(ttp)) if ttp else float("nan")
    never_share = (legs_n - len(paid_legs)) / legs_n
    # A leg pays relocator_share * gamma; mean cost per leg says whether the
    # ledger is net positive or negative.
    pay_per_leg = float(env._cfg.relocator_share) * float(env._cfg.gamma)
    mean_cost = sum(g["cost"] for g in all_legs) / legs_n
    net_per_leg = pay_per_leg * (len(paid_legs) / legs_n) - mean_cost

    print(f"\n  relocation legs:            {legs_n}")
    print(f"  legs that ever paid back:   {len(paid_legs)} "
          f"({100.0 * len(paid_legs) / legs_n:.1f}%)")
    print(f"  never_paid_share:           {never_share:.3f}")
    print(f"  mean turns_to_payback:      {mean_ttp:.1f} (horizon {horizon} turns)")
    print(f"  mean cost per leg:          {mean_cost:.4f}  (paid per leg = {pay_per_leg})")
    print(f"  net per leg (paid - cost):  {net_per_leg:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "relocation_roi",
        "checkpoint": str(ckpt),
        "episodes": args.episodes,
        "legs": len(all_legs),
        "legs_paid": len(paid_legs),
        "never_paid_share": never_share,
        "mean_turns_to_payback": None if np.isnan(mean_ttp) else mean_ttp,
        "pay_per_leg": pay_per_leg,
        "mean_cost_per_leg": mean_cost,
        "net_per_leg": net_per_leg,
        "turns_to_payback_hist": {
            str(b): int(sum(1 for t in ttp if lo <= t <= hi))
            for lo, hi in ((0, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 6))
            for b in [f"{lo}-{hi if hi < 10 ** 6 else '+'}"]
        },
        "per_episode": per_ep,
    }, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()