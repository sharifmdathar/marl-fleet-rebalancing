"""Is the raw-feature path starved-but-correct, or starved-and-harmful?

D54's decomposition showed the destination is decided by the key/query term
alone (100% argmax agreement, 61% of logit variation), while the raw-feature
term contributes under 6% and agrees with the full head 87% of the time.

That motivates an MLP for the raw path -- but only under one hypothesis:

  (A) CAPACITY. The raw path is a Linear(8, 1) and cannot express what the
      learnability probe expressed with an MLP, so widening it should help.
  (B) OPTIMISATION. The raw path is already good enough and the optimiser
      simply preferred key/query, so widening it should change nothing.

These make opposite predictions, so test them before changing anything. Measures,
over real relocation opportunities, how needy the chosen destination is when
each term decides ALONE:

    demand_ratio(term) = mean demand at chosen target / mean demand at best target

If raw-only scores as well as or better than the full head, the raw path is not
the bottleneck and an MLP is the wrong change. If full-head beats raw-only, key/query
is adding noise on top of a correct raw signal.

    uv run python scripts/diag_raw_path_quality.py \
        --mappo runs/d54-mappo-s0/checkpoint_best.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_dest_head_decomposition import terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--turns", type=int, default=400)
    ap.add_argument("--random", action="store_true",
                    help="also score a random legal target as the floor")
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    env.reset(seed=args.base_seed)
    mp = env._max_pending
    nz = env._num_zones
    hl = env._hist_len

    obs_list, legal_list, pend_list = [], [], []
    while env.agents and len(obs_list) < args.turns:
        for _ in range(nz):
            if not env.agents:
                break
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            legal = env._legal_relocate_targets(zi)
            if legal:
                obs_list.append(obs_to_vector(env.observe(ag)).copy())
                legal_list.append(np.array(sorted(legal)))
                pend_list.append(np.asarray(env._pending_per_zone, dtype=float)
                                 * max(env._num_drivers, 1))
            m = env.action_mask(ag)
            d = np.flatnonzero(m[1: 1 + mp])
            a = int(d[0]) + 1 if d.size else 0
            if not m[a]:
                a = int(np.flatnonzero(m)[0])
            env.step(a)

    print(f"relocation opportunities: {len(obs_list)}\n")

    def score(picks, pends, legals):
        """Mean demand at chosen target / mean demand at best target."""
        num = den = 0.0
        top1 = 0
        for p, pend, lg in zip(picks, pends, legals):
            best = lg[int(np.argmax(pend[lg]))]
            num += float(pend[p])
            den += float(pend[best])
            top1 += int(p == best)
        return num / max(den, 1e-9), 100.0 * top1 / max(len(picks), 1)

    rng = np.random.default_rng(0)
    rnd_ratio, rnd_top1 = score(
        [int(lg[rng.integers(len(lg))]) for lg in legal_list],
        pend_list, legal_list)

    print(f"{'selector':<28} {'demand ratio':>13} {'top-1':>8}")
    print(f"{'random legal target':<28} {rnd_ratio:12.2f} {rnd_top1:7.1f}%")

    for label, ck in (("IPPO", args.ippo), ("MAPPO", args.mappo)):
        if not ck:
            continue
        learner, _m, _ = load_trained_policy(Path(ck))
        net = (getattr(learner, "actor", None)
               or getattr(learner, "policy")).eval()

        picks = {"full": [], "key/query": [], "raw bias": [], "travel": []}
        for v, lg in zip(obs_list, legal_list):
            kq, b, t = terms(net, v, nz, hl)
            mask = np.isin(np.arange(nz), lg)
            for name, score_vec in (
                ("full", kq + b + t), ("key/query", kq),
                ("raw bias", b), ("travel", t),
            ):
                masked = np.where(mask, score_vec, -1e9)
                picks[name].append(int(np.argmax(masked)))
        print()
        ratios = {}
        for name, p in picks.items():
            r, t1 = score(p, pend_list, legal_list)
            ratios[name] = r
            print(f"{label + ' ' + name:<28} {r:12.2f} {t1:7.1f}%")
        print()
        print(f"  {label}: full head vs raw path alone = "
              f"{ratios['full']:.2f} vs {ratios['raw bias']:.2f}")
        if ratios["raw bias"] >= ratios["full"] - 1e-9:
            print("  -> the RAW path is already as good as the full head. It is")
            print("     not capacity-limited, so widening it is the WRONG change;")
            print("     the optimiser simply preferred key/query and the head is")
            print("     already extracting what the features offer.")
        else:
            print("  -> key/query is ADDING NOISE on top of a better raw signal.")
            print("     Widening the raw path is justified, and reducing key/query")
            print("     may be justified too.")


if __name__ == "__main__":
    main()
