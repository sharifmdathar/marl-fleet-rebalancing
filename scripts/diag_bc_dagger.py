"""Close the BC clone's covariate shift with DAgger, on the LP as teacher.

D99 measured the gap this addresses: the clone scores the LP's service rate and
deadhead (67.5% / 311 km vs 67.9% / 286 km) while matching the LP's exact
relocation target on only 36% of held-out LP states. Teacher-forced accuracy on
the LP's own trajectory answers "can the network fit the labels"; it does not
answer "does the clone stay where the labels are valid". DAgger answers the
second one: roll the clone out, label ITS states with the LP, retrain.

The reported number to watch is `reloc_target` on the ROUND's rollout states
(`on_policy`), not on the cached LP tape (`off_policy`). The first measures the
clone's targeting where it actually goes; the second was already known.

This is a diagnostic and a warm-start donor, never a result: the teacher is the
baseline itself, so a clone can at best reproduce it.

    uv run python scripts/diag_bc_dagger.py --rounds 4
    uv run python scripts/diag_objective_gap.py --lp \
        --arm dagger=runs/nyc16-dagger/checkpoint.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_bc_lp import class_report, masked_logits  # shared with the BC probe

from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.baselines import ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import (
    ZoneActorCritic,
    obs_to_vector,
)


def load_actor(path: Path) -> ZoneActorCritic:
    w = torch.load(path, map_location="cpu", weights_only=True)
    actor = w.get("actor", w)
    net = ZoneActorCritic(actor["trunk.0.weight"].shape[1],
                          actor["actor.weight"].shape[0],
                          actor["trunk.0.weight"].shape[0])
    net.load_state_dict(actor)
    return net


@torch.no_grad()
def rollout(net: ZoneActorCritic, split: str, episodes: int, base_seed: int,
            label: bool) -> dict:
    """Run `net` and record every state; optionally label it with the LP."""
    cfg = NYCConfig()
    cfg.nyc_split = split
    lp = ZoneLPRebalancer() if label else None
    obs_all, mask_all, act_all = [], [], []
    t0 = time.time()
    for ep in range(episodes):
        seed = base_seed + ep
        env = RideGymAEC(cfg)
        env.reset(seed=seed)
        if lp is not None:
            lp.reset(seed=seed)
        while env.agents:
            ag = env.agent_selection
            m = np.asarray(env.action_mask(ag))
            o = obs_to_vector(env.observe(ag))
            logits = masked_logits(net, o[None], m[None])
            a = int(logits.argmax(-1)[0])
            if label:
                lab = int(lp.select_action(ag, env, env.observe(ag)))
                if not m[lab]:
                    raise RuntimeError(f"LP label {lab} is masked for {ag}")
                obs_all.append(o)
                mask_all.append(m.astype(np.uint8))
                act_all.append(lab)
            if not m[a]:
                raise RuntimeError(f"clone chose masked action {a}")
            env.step(a)
        if (ep + 1) % 6 == 0 or ep == episodes - 1:
            print(f"[dagger] {split} rollout {ep + 1}/{episodes} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not label:
        return {}
    return {"obs": np.stack(obs_all), "mask": np.stack(mask_all),
            "act": np.asarray(act_all, dtype=np.int16)}


@torch.no_grad()
def agreement(net: ZoneActorCritic, d: dict, max_pending: int,
              batch: int = 4096) -> dict:
    """Top-1 agreement with the labels, split by what the teacher chose."""
    n = len(d["act"])
    pred = np.empty(n, dtype=np.int64)
    for i in range(0, n, batch):
        pred[i:i + batch] = masked_logits(
            net, d["obs"][i:i + batch], d["mask"][i:i + batch]).argmax(-1).numpy()
    act = d["act"]
    is_reloc = act > max_pending
    hit = pred == act
    return {
        "overall": 100 * float(hit.mean()),
        "reloc_target": 100 * float(hit[is_reloc].mean()) if is_reloc.any() else float("nan"),
        "reloc_rate_teacher": 100 * float(is_reloc.mean()),
        "reloc_rate_clone": 100 * float((pred > max_pending).mean()),
        "n": int(n),
    }


def concat(*parts: dict) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def train_rounds(net: ZoneActorCritic, d: dict, max_pending: int, epochs: int,
                 lr: float, batch_size: int, reloc_weight: float, seed: int) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.from_numpy(d["obs"].astype(np.float32))
    mask = torch.from_numpy(d["mask"].astype(bool))
    act = torch.from_numpy(d["act"].astype(np.int64))
    weight = torch.where(act > max_pending,
                         torch.tensor(float(reloc_weight)), torch.tensor(1.0))
    n = len(act)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, _ = net(obs[idx], mask[idx])
            ce = torch.nn.functional.cross_entropy(logits, act[idx], reduction="none")
            loss = (ce * weight[idx]).sum() / weight[idx].sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        print(f"[dagger] epoch {epoch + 1}/{epochs} loss {total / count:.4f} "
              f"n={n}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/nyc16-bc-lp-rw5/checkpoint.pt")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=12,
                    help="clone rollouts per round, train split")
    ap.add_argument("--epochs", type=int, default=3, help="epochs per round")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--reloc-weight", type=float, default=5.0)
    ap.add_argument("--rollout-base-seed", type=int, default=1000,
                    help="disjoint from the BC tape seeds (0..59)")
    ap.add_argument("--val-base-seed", type=int, default=12000)
    ap.add_argument("--val-episodes", type=int, default=12)
    ap.add_argument("--data-dir", default="runs/bc-lp")
    ap.add_argument("--out-dir", default="runs/nyc16-dagger")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data_dir)
    d0 = {k: v for k, v in np.load(data / "train_60ep.npz").items()
          if k in ("obs", "mask", "act")}
    d_tape_val = {k: v for k, v in np.load(data / f"val_{args.val_episodes}ep.npz").items()
                  if k in ("obs", "mask", "act")}
    max_pending = (d0["mask"].shape[1] - 1) // 2
    print(f"[dagger] max_pending={max_pending} init={args.init}")
    print(f"[dagger] tape: {len(d0['act'])} decisions, "
          f"val tape: {len(d_tape_val['act'])}")

    net = load_actor(Path(args.init))
    train_parts = [d0]
    history = []

    def evaluate(round_idx: int) -> dict:
        off = agreement(net, d_tape_val, max_pending)
        on = rollout(net, "val", args.val_episodes, args.val_base_seed, label=True)
        on_acc = agreement(net, on, max_pending)
        rec = {"round": round_idx, "off_policy": off, "on_policy": on_acc,
               "n_train": int(sum(len(p["act"]) for p in train_parts))}
        history.append(rec)
        print(f"[dagger] R{round_idx} off-policy reloc_target "
              f"{off['reloc_target']:.1f}% | ON-POLICY reloc_target "
              f"{on_acc['reloc_target']:.1f}% (clone reloc {on_acc['reloc_rate_clone']:.1f}% "
              f"vs teacher {on_acc['reloc_rate_teacher']:.1f}%) | n={rec['n_train']}",
              flush=True)
        return rec

    evaluate(0)
    for r in range(1, args.rounds + 1):
        new = rollout(net, "train", args.episodes,
                      args.rollout_base_seed + (r - 1) * args.episodes, label=True)
        print(f"[dagger] R{r} collected {len(new['act'])} decisions, "
              f"class mix {class_report(new['act'], max_pending)}", flush=True)
        train_parts.append(new)
        d = concat(*train_parts)
        train_rounds(net, d, max_pending, args.epochs, args.lr, args.batch_size,
                     args.reloc_weight, seed=args.seed + r)
        evaluate(r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Same D42 critic layout train.py builds, so `train --resume` can load it.
    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    env.reset(seed=0)
    sample = env.observe(env.possible_agents[0])
    learner = MAPPO(
        observation_dim=int(d0["obs"].shape[1]),
        state_dim=len(env.state()) + int(d0["obs"].shape[1]),
        action_dim=int(d0["mask"].shape[1]),
        config=MAPPOConfig(hidden_dim=int(net.trunk[0].out_features), arch="mlp"),
        num_zones=16, max_pending=max_pending,
        hist_len=int(sample["demand_hist"].shape[1]),
    )
    learner.actor.load_state_dict(net.state_dict())
    torch.save(learner.state_dict(), out_dir / "checkpoint.pt")

    ref = json.loads(Path("runs/nyc16-h24-obs-s0/manifest.json").read_text())
    manifest = copy.deepcopy(ref)
    manifest.update({
        "scenario_id": "DAgger-lp-nyc",
        "algo": "mappo",
        "iterations": 0,
        "trained_by": "DAgger on ZoneLPRebalancer, no PPO",
        "dagger": {
            "init": args.init, "rounds": args.rounds,
            "episodes_per_round": args.episodes, "epochs_per_round": args.epochs,
            "reloc_weight": args.reloc_weight, "lr": args.lr,
            "rollout_seeds": [args.rollout_base_seed,
                              args.rollout_base_seed
                              + args.rounds * args.episodes - 1],
            "history": history,
        },
    })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    out = Path("results/bc_lp_dagger.json")
    out.write_text(json.dumps({"history": history}, indent=2, default=float))
    print(f"[dagger] wrote {out_dir}/checkpoint.pt and {out}")


if __name__ == "__main__":
    main()
