"""Decompose the destination head: which term actually decides the target?

D54 raised decision-time precision from ~6% (below chance) to 13-16% (above
~8% chance). But the mean unmet demand at the chosen target is still only 0.34 of
the best available, and a supervised probe reaches 87% on the same features. So
the head works better than before and is still far from what is learnable.

This attributes each destination logit to its three terms, so the dominant term
is identified rather than guessed:

    dest[j] = key_query[j]  +  raw_bias[j]  +  travel[j]

Reported per term: its spread across destinations (does it vary at all?), its
share of the total variation, and the destination it alone would pick. If one
term dominates the argmax, the others are decoration.

Also reports what the head would pick with each term ablated, so a term that
looks small in magnitude but still flips the argmax is visible.

    uv run python scripts/diag_dest_head_decomposition.py \
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

A0 = 6


def terms(net, v: np.ndarray, n_zones: int, hist_len: int):
    """Recompute the three destination-logit terms for one observation."""
    dev = next(net.parameters()).device
    obs = torch.as_tensor(v, dtype=torch.float32, device=dev)[None, :]

    def sl(name):
        lo, hi = net._offsets[name]
        return obs[..., lo:hi]

    b = obs.shape[0]
    zone_loc = sl("zone_location")
    zone_state = sl("zone_state")
    azs = sl("all_zone_state").view(b, n_zones, 4)
    dh = sl("demand_hist").view(b, n_zones, hist_len)
    nbr = sl("neighbor_mask")
    self_ctx = torch.cat([zone_loc, zone_state], dim=-1)

    zone_feat = torch.cat([azs, dh], dim=-1)
    zone_emb = net.zone_embed(zone_feat)

    keys = net.dest_key(zone_emb)
    q = net.self_query(self_ctx).unsqueeze(1)
    t_kq = (keys * q).sum(-1) / net._dest_scale

    t_bias = net.dest_bias(torch.cat([azs, dh], dim=-1)).squeeze(-1)

    cents = net.zone_centroid(net.zone_idx.weight)
    geo = torch.stack([
        cents[:, 0].unsqueeze(0) - zone_loc[:, 0:1],
        cents[:, 1].unsqueeze(0) - zone_loc[:, 1:2],
    ], dim=-1)
    t_travel = net.travel_cost(geo).squeeze(-1)

    return (t_kq[0].detach().cpu().numpy(),
            t_bias[0].detach().cpu().numpy(),
            t_travel[0].detach().cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--turns", type=int, default=60)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    env.reset(seed=args.base_seed)
    mp = env._max_pending
    nz = env._num_zones
    hl = env._hist_len

    # Collect real observations from zone-turns where relocation is legal.
    obs_list = []
    legal_list = []
    while env.agents and len(obs_list) < args.turns:
        for _ in range(nz):
            if not env.agents:
                break
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            legal = env._legal_relocate_targets(zi)
            if legal:
                obs_list.append(obs_to_vector(env.observe(ag)).copy())
                legal_list.append(set(legal))
            m = env.action_mask(ag)
            d = np.flatnonzero(m[1: 1 + mp])
            a = int(d[0]) + 1 if d.size else 0
            if not m[a]:
                a = int(np.flatnonzero(m)[0])
            env.step(a)

    print(f"collected {len(obs_list)} relocation-opportunity observations\n")

    for label, ck in (("IPPO", args.ippo), ("MAPPO", args.mappo)):
        if not ck:
            continue
        learner, _m, _ = load_trained_policy(Path(ck))
        net = (getattr(learner, "actor", None)
              or getattr(learner, "policy")).eval()

        sp_kq, sp_b, sp_t = [], [], []
        full_pick, kq_pick, b_pick, t_pick = [], [], [], []
        agree_kq = 0

        for v, legal in zip(obs_list, legal_list):
            kq, b, t = terms(net, v, nz, hl)
            mask = np.array([z in legal for z in range(nz)])

            def pick(x):
                y = np.where(mask, x, -1e9)
                return int(np.argmax(y))

            full = pick(kq + b + t)
            pk, pb, pt = pick(kq), pick(b), pick(t)
            full_pick.append(full)
            kq_pick.append(pk)
            b_pick.append(pb)
            t_pick.append(pt)
            agree_kq += (pk == full)

            tot = np.abs(kq + b + t).std() or 1e-9
            sp_kq.append(np.abs(kq[mask]).std() / tot)
            sp_b.append(np.abs(b[mask]).std() / tot)
            sp_t.append(np.abs(t[mask]).std() / tot)

        print("=" * 74)
        print(f"{label}: which term decides the destination?")
        print("=" * 74)
        print(f"  share of total logit variation (mean over {len(obs_list)} obs)")
        print(f"    key/query  {np.mean(sp_kq)*100:5.1f}%")
        print(f"    raw bias   {np.mean(sp_b)*100:5.1f}%")
        print(f"    travel     {np.mean(sp_t)*100:5.1f}%")
        print()
        print(f"  argmax agreement with the full head:")
        print(f"    key/query alone : {100*agree_kq/len(obs_list):5.1f}%")
        print(f"    raw bias alone  : "
              f"{100*np.mean([p==f for p,f in zip(b_pick, full_pick)]):5.1f}%")
        print(f"    travel alone    : "
              f"{100*np.mean([p==f for p,f in zip(t_pick, full_pick)]):5.1f}%")
        print()
        dom = max(range(3), key=lambda i: np.mean([sp_kq, sp_b, sp_t][i]))
        name = ["key/query", "raw bias", "travel"][dom]
        print(f"  dominant term: {name}")
        if dom == 0:
            print("    The key/query path decides the destination. It is built from")
            print("    a Tanh zone embedding attended by the agent's own state, so it")
            print("    CAN carry demand -- but only through the embedding. The direct")
            print("    raw-feature bias, which the learnability probe showed is the")
            print("    sufficient signal, is contributing little.")
        elif dom == 1:
            print("    The direct raw-feature term dominates, which is the intended")
            print("    design: demand and car counts reach the logit unmediated.")
        else:
            print("    Travel cost dominates. The head is mostly choosing a NEAR")
            print("    zone rather than a NEEDY one, which would cap precision well")
            print("    below chance-level improvements.")
        print()


if __name__ == "__main__":
    main()
