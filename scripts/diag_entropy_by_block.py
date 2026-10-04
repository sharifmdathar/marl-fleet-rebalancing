"""Is the relocation sub-distribution actually flat, or is the mean hiding it?

The training log reports mean entropy over ALL agent decisions, including noop
and dispatch. The D60 finding was that the relocation choice specifically is
near-uniform across ~10 legal targets (top choice 14.2%). Those two facts are
only compatible if the relocate decisions are a small, flat minority of the
mean -- so measure entropy PER ACTION BLOCK rather than overall.

Splits each decision's legal-action set into:
  * relocate actions only (the ones D60 is about)
  * everything else (noop + dispatch)

and reports the top-1 probability and entropy for each. Then does the same for
two checkpoints trained at different entropy_coef, which is the actual question:
does lowering the coefficient sharpen the relocate choice at all?

    uv run python scripts/diag_entropy_by_block.py \
        --ckpt runs/ec0.001-mappo-s0/checkpoint_best.pt:0.001 \
        --ckpt runs/d59-mappo-s0/checkpoint_best.pt:0.01
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


def stats(p: np.ndarray) -> tuple[float, float]:
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return float("nan"), float("nan")
    s = p.sum()
    q = p / s if s > 0 else p
    ent = float(-(q * np.log(q)).sum())
    return float(q.max()), ent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                    metavar="PATH:COEF", help="checkpoint:entropy_coef label")
    ap.add_argument("--base-seed", type=int, default=8100)
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    mp = cfg.max_pending

    for spec in args.ckpt:
        path, _, coef = spec.rpartition(":")
        learner, _m, _ = load_trained_policy(Path(path))
        net = (getattr(learner, "actor", None)
               or getattr(learner, "policy")).eval()
        dev = next(net.parameters()).device

        env = RideGymAEC(cfg)
        env.reset(seed=args.base_seed)
        reloc_top, reloc_ent, reloc_n = [], [], []
        other_top, other_ent, other_n = [], [], []
        seen = 0

        while env.agents and seen < args.max:
            for _ in range(env._num_zones):
                if not env.agents:
                    break
                ag = env.agent_selection
                zi = env._zone_int_id(ag)
                m = np.asarray(env.action_mask(ag), dtype=bool)
                legal = env._legal_relocate_targets(zi)
                if legal:
                    with torch.no_grad():
                        lg, _ = net(
                            torch.as_tensor(obs_to_vector(env.observe(ag)),
                                            dtype=torch.float32,
                                            device=dev)[None, :], None)
                    lg = lg[0].cpu().numpy().astype(float)
                    base = 1 + mp
                    r = lg[base:base + env._num_zones]
                    rmask = m[base:base + env._num_zones]
                    if rmask.any():
                        pr = np.where(rmask, np.exp(r - r[rmask].max()), 0.0)
                        t, e = stats(pr)
                        reloc_top.append(t)
                        reloc_ent.append(e)
                        reloc_n.append(int(rmask.sum()))
                    others = np.where(m[:base], np.exp(
                        lg[:base] - lg[:base][m[:base]].max()), 0.0)
                    t2, e2 = stats(others)
                    other_top.append(t2)
                    other_ent.append(e2)
                    other_n.append(int(m[:base].sum()))
                    seen += 1
                d = np.flatnonzero(m[1:1 + mp])
                a = int(d[0]) + 1 if d.size else 0
                if not m[a]:
                    a = int(np.flatnonzero(m)[0])
                env.step(a)

        print("=" * 72)
        print(f"entropy_coef {coef}  ({Path(path).parent.name})")
        print("=" * 72)
        print(f"  relocate block  : {len(reloc_top):4d} decisions")
        if reloc_top:
            print(f"    mean top-1 probability : {100*np.mean(reloc_top):5.1f}%")
            print(f"    mean entropy           : {np.mean(reloc_ent):5.3f} nats")
            print(f"    mean legal targets     : {np.mean(reloc_n):5.2f}")
            mx = max(np.log(reloc_n)) if reloc_n else 0
            print(f"    max possible entropy   : {mx:5.3f} nats")
            print(f"    -> entropy as %% of max : "
                  f"{100*np.mean(reloc_ent)/max(mx,1e-9):5.1f}%")
        print(f"  dispatch/noop block: {len(other_top):4d} decisions")
        if other_top:
            print(f"    mean top-1 probability : {100*np.mean(other_top):5.1f}%")
            print(f"    mean entropy           : {np.mean(other_ent):5.3f} nats")
            print(f"    mean legal actions     : {np.mean(other_n):5.2f}")
        print()


if __name__ == "__main__":
    main()
