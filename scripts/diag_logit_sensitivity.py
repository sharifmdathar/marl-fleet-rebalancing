"""Measure the network's sensitivity to each demand channel, directly.

An earlier probe overwrote `all_zone_state` column 2 (pending) and left column 3
(deficit = pending - idle) at its original value, so it tested a state the env
could never produce. It concluded "the output is invariant to demand", which may
have been the probe's fault rather than the network's.

This measures properly, by perturbing ONE input dimension at a time and
reporting the resulting change in the relocation logits:

  * gradient-style sensitivity: d(logit_j) / d(input_i) for every relocation
    action j and every demand-bearing input i, by finite difference
  * rank agreement: does the argmax relocation action change when demand moves
    to a different zone, with ALL demand columns updated consistently?

Finite differences on the real network, no sampling, no env randomness, so the
only thing that varies is the input.

    uv run python scripts/diag_logit_sensitivity.py \
        --mappo runs/d51-mappo-s0/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

A0 = 6
NZ = 16
MP = 16
DH0 = 214


def actor_logits(learner, v: np.ndarray) -> np.ndarray:
    """Logits over all actions for one observation.

    The attention encoder's `forward(obs, action_mask)` returns
    `(logits, value)`; that is the audited path, the same one `act()` uses.
    """
    net = getattr(learner, "actor", None)
    if net is None:
        net = getattr(learner, "policy", None)
    if net is None:
        raise SystemExit("could not locate the actor network on the learner")
    with torch.no_grad():
        dev = next(net.parameters()).device
        out = net(torch.as_tensor(v, dtype=torch.float32, device=dev)[None, :],
                  None)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().cpu().numpy().astype(float).ravel()


def reloc_slice(a: np.ndarray) -> np.ndarray:
    return a[1 + MP: 1 + MP + NZ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--eps", type=float, default=0.25)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    env.reset(seed=args.base_seed)
    MP = env._max_pending

    # Find a zone-step where relocation is legal, so its logits are meaningful.
    target = None
    for _ in range(4000):
        if not env.agents:
            break
        ag = env.agent_selection
        zi = env._zone_int_id(ag)
        if env._zone_idle_counts[zi] > 0 and env._legal_relocate_targets(zi):
            target = (ag, zi)
            break
        m = env.action_mask(ag)
        d = np.flatnonzero(m[1: 1 + MP])
        a = int(d[0]) + 1 if d.size else 0
        if not m[a]:
            a = int(np.flatnonzero(m)[0])
        env.step(a)
    if target is None:
        print("no legal relocation opportunity found")
        return
    ag, zi = target
    v0 = obs_to_vector(env.observe(ag)).copy()
    legal = set(env._legal_relocate_targets(zi))
    print(f"zone {zi}: {len(legal)} legal relocation targets")
    print(f"probe eps = {args.eps}\n")

    def write_demand(v: np.ndarray, hot: int) -> np.ndarray:
        """Put all pending on `hot`, updating BOTH derived columns."""
        w = v.copy()
        for z in range(NZ):
            p = 1.0 if z == hot else 0.0
            w[A0 + z * 4 + 2] = p                 # pending
            w[A0 + z * 4 + 3] = p                 # deficit (pending - idle)
        return w

    for label, ck in (("IPPO", args.ippo), ("MAPPO", args.mappo)):
        if not ck:
            continue
        learner, _m, _ = load_trained_policy(Path(ck))
        base = actor_logits(learner, v0)
        rb = reloc_slice(base)

        print("=" * 72)
        print(f"{label}: finite-difference sensitivity of the relocate logits")
        print("=" * 72)

        groups = {
            "all_zone_state col2 (pending)":
                [A0 + z * 4 + 2 for z in range(NZ)],
            "all_zone_state col3 (deficit)":
                [A0 + z * 4 + 3 for z in range(NZ)],
            "demand_hist (64 dims)":
                list(range(DH0, DH0 + 64)),
            "neighbor_mask (16 dims)":
                list(range(198, 214)),
        }
        for gname, dims in groups.items():
            grad = np.zeros(NZ)
            for i in dims:
                vp = v0.copy(); vp[i] += args.eps
                vm = v0.copy(); vm[i] -= args.eps
                rp = reloc_slice(actor_logits(learner, vp))
                rm = reloc_slice(actor_logits(learner, vm))
                grad += (rp - rm) / (2 * args.eps)
            denom = np.abs(grad).sum() or 1.0
            share = 100.0 * np.abs(grad).sum() / (
                np.abs(grad).sum() + 1e-12)
            print(f"  {gname:<32} L1 sensitivity {np.abs(grad).sum():9.4f} "
                  f" max|d/dx| {np.abs(grad).max():8.4f}")
        print()

        # Rank agreement: does the argmax target follow the demand?
        print("  argmax relocation target as demand moves (all cols updated):")
        picks = []
        for hot in range(NZ):
            rl = reloc_slice(actor_logits(learner, write_demand(v0, hot)))
            if legal:
                masked = np.array([rl[z] if z in legal else -1e9 for z in range(NZ)])
                picks.append(int(np.argmax(masked)))
            else:
                picks.append(int(np.argmax(rl)))
        for hot in range(min(NZ, 8)):
            print(f"    all demand in zone {hot:2d} -> picks zone {picks[hot]:2d}")
        hit = sum(1 for hot in range(NZ) if picks[hot] == hot)
        print(f"  picked the zone that actually holds the demand: {hit}/{NZ} "
              f"({100*hit/NZ:.0f}%, chance {100/NZ:.1f}%)")
        print()


if __name__ == "__main__":
    main()
