#!/usr/bin/env python3
"""Dump one real zone-turn of the shipped attention policy, tensor by tensor.

Purpose. The deck claims to show "what the layers see and what happens".
Rather than draw an invented example, this replays the real MAPPO checkpoint on
the real environment at the demo operating point, finds a turn where a zone has
waiting riders, an idle car AND at least one legal relocation, and prints every
intermediate tensor for that one decision.

Read-only: builds the env, loads a checkpoint, steps it. Changes no project code
and writes nothing except its own JSON dump.

    uv run python scripts/probe_one_zone_turn.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d83-chain-mappo-s1/checkpoint_it0080.pt")
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--drivers", type=int, default=55)
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--out", default="results/probe_zone_turn.json")
    ap.add_argument("--prefer", choices=["relocate", "dispatch", "any"],
                    default="relocate",
                    help="only accept a turn whose argmax is of this kind")
    ap.add_argument("--min-pending", type=int, default=2)
    ap.add_argument("--min-targets", type=int, default=3)
    args = ap.parse_args()

    cfg = replace(NYCConfig(), num_drivers=args.drivers,
                  nyc_window_limit=args.cap, max_pending=16,
                  obs_inbound=True, beta=0.12, order_timeout=args.timeout)
    env = RideGymAEC(cfg)
    algo, _, _ = load_trained_policy(Path(args.ckpt))
    # MAPPO exposes the encoder as .actor, IPPO as .policy
    enc = getattr(algo, "actor", None) or getattr(algo, "policy")
    P = cfg.max_pending                    # 16
    Z = env.unwrapped._num_zones           # 16, from the data manifest

    # the trainer copies the real centroids into this buffer (train.py:681)
    if hasattr(enc, "zone_xy"):
        with torch.no_grad():
            enc.zone_xy.copy_(torch.as_tensor(
                env.unwrapped._zone_centroids_norm, dtype=torch.float32))

    env.reset(seed=args.seed)
    found = None

    for turn in range(120):
        for _ in env.agent_iter():
            obs, _, term, trunc, info = env.last()
            agent = env.agent_selection
            if term or trunc:
                env.step(None)
                continue

            mask = np.asarray(env.action_mask(agent), dtype=bool)
            zone_i = int(env.unwrapped._zone_int_id(agent))   # "zone_9" -> 9
            n_disp = int(mask[1:1 + P].sum())
            n_reloc = int(mask[1 + P:].sum())

            # candidate turns: riders waiting, a car free to move, and
            # somewhere legal to send it
            if (found is None and n_disp > 0 and n_reloc >= args.min_targets
                    and float(obs_to_vector(obs)[166:182].sum()) >= args.min_pending):
                vec = obs_to_vector(obs)
                dev = next(enc.parameters()).device
                t = torch.as_tensor(vec, dtype=torch.float32, device=dev).unsqueeze(0)
                mt = torch.as_tensor(mask, device=dev)

                with torch.no_grad():
                    raw_logits, value = enc(t)                 # unmasked
                    masked_logits, _ = enc(t, mt)

                # --- the four terms of the relocation head, recomputed
                with torch.no_grad():
                    b = 1
                    zl = t[..., 0:2]
                    zs = t[..., 2:6]
                    sc = torch.cat([zl, zs], dim=-1)
                    azs = t[..., 6:70].view(b, Z, 4)
                    dh = t[..., 214:278].view(b, Z, 4)
                    zone_emb = enc.zone_embed(torch.cat([azs, dh], dim=-1))
                    keys = enc.dest_key(zone_emb)
                    q = enc.self_query(sc).unsqueeze(1)
                    term_a = (keys * q).sum(-1) / enc._dest_scale
                    per_dest = [azs, dh]
                    if getattr(enc, "_inbound", False):
                        per_dest.append(t[..., 279:295].unsqueeze(-1))
                    term_b = enc.dest_bias(torch.cat(per_dest, dim=-1)).squeeze(-1)
                    cents = enc.zone_centroid(enc.zone_idx.weight)
                    geo = torch.stack([
                        cents[:, 0].unsqueeze(0) - zl[:, 0:1],
                        cents[:, 1].unsqueeze(0) - zl[:, 1:2]], dim=-1)
                    term_c = enc.travel_cost(geo).squeeze(-1)
                    dist = (enc.zone_xy.unsqueeze(0) - zl.unsqueeze(1)).norm(dim=-1)
                    term_d = enc.dist_cost(
                        (dist * enc.dist_input_scale).unsqueeze(-1)).squeeze(-1)

                    # attention internals
                    po = t[..., 70:166].view(b, P, 6)
                    pp = t[..., 182:198]
                    pm = t[..., 166:182]
                    ofeat = torch.cat([po, pp.unsqueeze(-1)], dim=-1)
                    oemb = enc.order_embed(ofeat)
                    oq = enc.order_query(sc).unsqueeze(1)
                    olg = (oq * oemb).sum(-1) / enc._attn_scale
                    olg = olg.masked_fill(pm < 0.5, -1e9)
                    oattn = torch.softmax(olg, -1) * (pm.sum(-1, keepdim=True) > 0)
                    opool = (oattn.unsqueeze(-1) * oemb).sum(1)

                    nm = t[..., 198:214]
                    zlg = (enc.zone_query(sc).unsqueeze(1) * zone_emb).sum(-1) / enc._attn_scale
                    zattn = torch.softmax(zlg + enc.neighbor_bias * nm, -1)
                    zpool = (zattn.unsqueeze(-1) * zone_emb).sum(1)

                    ctx = torch.cat([zl, zs, opool, zpool, t[..., 278:279]], dim=-1)
                    h = enc.trunk(ctx)

                logits = raw_logits[0].cpu().numpy()
                probs_t = torch.softmax(masked_logits[0], dim=-1).cpu().numpy()
                chosen_i = int(np.argmax(masked_logits[0].cpu().numpy()))

                # when asked for a relocation example, keep scanning until a
                # turn actually ends in one
                if args.prefer == "relocate" and not (P + 1 <= chosen_i <= P + Z):
                    env.step(None)
                    continue
                if args.prefer == "dispatch" and not (1 <= chosen_i <= P):
                    env.step(None)
                    continue

                found = {
                    "ckpt": args.ckpt, "turn": turn, "agent": agent, "zone_index": zone_i,
                    "minute": round(turn * cfg.dt, 2),
                    "obs_len": int(vec.shape[0]),
                    "obs_layout": {k: list(v) for k, v in enc._offsets.items()},
                    "zone_location": vec[0:2].tolist(),
                    "zone_state": vec[2:6].tolist(),
                    "all_zone_state": vec[6:70].reshape(Z, 4).tolist(),
                    "pending_mask": vec[166:182].tolist(),
                    "n_pending": int(vec[166:182].sum()),
                    "pending_orders_slot0": vec[70:76].tolist(),
                    "neighbor_mask": vec[198:214].tolist(),
                    "n_neighbor_mask": int(vec[198:214].sum()),
                    "demand_hist_own": vec[214:218].tolist(),
                    "time_frac": float(vec[278]),
                    "inbound_own": float(vec[279 + zone_i]) if vec.shape[0] > 279 else None,
                    "order_attention": oattn[0].tolist(),
                    "zone_attention": zattn[0].tolist(),
                    "ctx_dim": int(ctx.shape[-1]),
                    "h_norm": float(h.norm()),
                    "reloc_terms": {
                        "A_key_query": term_a[0].tolist(),
                        "B_raw_bias": term_b[0].tolist(),
                        "C_travel_offset": term_c[0].tolist(),
                        "D_real_distance": term_d[0].tolist(),
                        "dist_norm": dist[0].tolist(),
                        "neighbor_bias": float(enc.neighbor_bias),
                    },
                    "logits": logits.tolist(),
                    "probs": probs_t.tolist(),
                    "action_mask": mask.tolist(),
                    "value": float(value.reshape(-1)[0]),
                    "n_legal_dispatch": n_disp,
                    "n_legal_reloc": n_reloc,
                    "chosen_action": chosen_i,
                }
                break
            env.step(None)
        if found is not None:
            break

    if found is None:
        print("no turn matched the filter; try another seed")
        return

    a = found["chosen_action"]
    found["action_meaning"] = (
        "no zone action (wait)" if a == 0 else
        f"dispatch ranked candidate order #{a}" if a <= P else
        f"relocate one idle car to zone index {a - 1 - P}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(found, indent=2))

    print(f"turn {found['turn']}  minute {found['minute']}  "
          f"agent {found['agent']} (index {found['zone_index']})")
    print(f"obs {found['obs_len']} dims | {found['n_pending']} pending riders | "
          f"{found['n_neighbor_mask']} legal targets")
    print(f"legal: {found['n_legal_dispatch']}/16 dispatch, "
          f"{found['n_legal_reloc']}/16 relocate")
    print(f"context vector {found['ctx_dim']} -> h 256, |h| = {found['h_norm']:.1f}")
    print("\ntop 6 legal actions:")
    lg, pr, m = found["logits"], found["probs"], found["action_mask"]
    for i in np.argsort(-np.where(m, lg, -np.inf))[:6]:
        print(f"   {i:2d}  logit {lg[i]:+7.3f}   p {pr[i]:.3f}")
    print(f"\nchosen action {a}: {found['action_meaning']}")
    print(f"critic value {found['value']:+.3f}")
    print(f"\nrelocation head terms for the top 4 legal destinations:")
    T = found["reloc_terms"]
    legal_dest = [j for j in range(16) if m[1 + P + j]]
    for j in legal_dest[:4]:
        print(f"   zone {j:2d}  A {T['A_key_query'][j]:+6.2f}  B {T['B_raw_bias'][j]:+6.2f}"
              f"  C {T['C_travel_offset'][j]:+6.2f}  D {T['D_real_distance'][j]:+6.2f}"
              f"  = {lg[1+P+j]:+6.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()