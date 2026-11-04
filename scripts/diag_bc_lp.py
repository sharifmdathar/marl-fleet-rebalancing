"""Can the actor architecture express the LP's policy at all?

Supervised fit of the tuned LP's actions into the SAME policy class the RL arm
uses `(flat MLP 615 -> 256 -> 256 -> 33, shared across zones, argmax at eval)`.
No PPO, no reward, no exploration: just cross-entropy on demonstrations.

Why this is the discriminating experiment (D98 left it open):

  BC ~= LP   -> the observation and the per-zone 33-way head CAN express the
                LP's behaviour, so the RL gap is optimisation / credit
                assignment, and a warm start should close it.
  BC << LP   -> the head cannot express it (the LP's plan is a coupled 16x16
                transfer matrix; a per-zone argmax cannot represent "the 4th
                car to zone 7"), so the interface is the binding constraint.

Two accuracies are reported, because they answer different questions:

  in-sample (train windows)   : fit quality.
  teacher-forced (val windows): the representation ceiling. Scored on the LP's
      OWN trajectory, so it is not contaminated by covariate shift. High
      teacher-forced accuracy with bad closed-loop KPIs means the clone drifted
      off the demonstration distribution (recoverable with DAgger), not that
      the head is too weak.

The written run dir is a normal checkpoint, so `scripts/diag_objective_gap.py`
scores it on the same paired windows as the LP and the RL seeds.

    uv run python scripts/diag_bc_lp.py --episodes 40
    uv run python scripts/diag_objective_gap.py --lp \
        --arm bc_lp=runs/nyc16-bc-lp/checkpoint.pt \
        --arm rl_s0=runs/nyc16-h24-obs-s0/checkpoint.pt
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

from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.baselines import ZoneLPRebalancer
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import (
    ZoneActorCritic,
    obs_dim,
    obs_to_vector,
)


def collect(split: str, episodes: int, base_seed: int, cache: Path) -> dict:
    """Roll the LP out and record (obs, action mask, action) at every decision."""
    if cache.exists():
        z = np.load(cache)
        print(f"[bc] reusing {cache} ({len(z['act'])} decisions)")
        return {k: z[k] for k in z.files}

    cfg = NYCConfig()
    cfg.nyc_split = split
    lp = ZoneLPRebalancer()

    obs_all, mask_all, act_all, win_all = [], [], [], []
    t0 = time.time()
    for ep in range(episodes):
        seed = base_seed + ep
        env = RideGymAEC(cfg)
        lp.reset(seed=seed)
        env.reset(seed=seed)
        gen = getattr(env, "_order_generator", None)
        window = getattr(gen, "last_window_index", -1)
        while env.agents:
            ag = env.agent_selection
            mask = env.action_mask(ag)
            obs = obs_to_vector(env.observe(ag))
            action = int(lp.select_action(ag, env, env.observe(ag)))
            if not mask[action]:
                raise RuntimeError(f"{ag} chose masked action {action}")
            obs_all.append(obs)
            mask_all.append(np.asarray(mask, dtype=np.uint8))
            act_all.append(action)
            win_all.append(window)
            env.step(action)
        if (ep + 1) % 5 == 0 or ep == episodes - 1:
            print(f"[bc] {split}: {ep + 1}/{episodes} episodes, "
                  f"{len(act_all)} decisions, {time.time() - t0:.0f}s")

    out = {
        "obs": np.stack(obs_all),
        "mask": np.stack(mask_all),
        "act": np.asarray(act_all, dtype=np.int16),
        "window": np.asarray(win_all, dtype=np.int16),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"[bc] wrote {cache}")
    return out


def class_report(act: np.ndarray, max_pending: int) -> dict:
    n = len(act)
    dispatch = int(np.sum((act >= 1) & (act <= max_pending)))
    reloc = int(np.sum(act > max_pending))
    noop = int(np.sum(act == 0))
    return {"decisions": n, "noop": noop, "dispatch": dispatch, "reloc": reloc,
            "noop_pct": 100 * noop / max(n, 1),
            "dispatch_pct": 100 * dispatch / max(n, 1),
            "reloc_pct": 100 * reloc / max(n, 1)}


def masked_logits(net: ZoneActorCritic, obs: np.ndarray,
                  mask: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs.astype(np.float32))
    m = torch.from_numpy(mask.astype(bool))
    logits, _ = net(x, m)
    return logits


@torch.no_grad()
def accuracy(net: ZoneActorCritic, d: dict, max_pending: int,
             batch: int = 4096) -> dict:
    """Top-1 accuracy overall, plus behaviour split by what the LP chose."""
    n = len(d["act"])
    pred = np.empty(n, dtype=np.int64)
    for i in range(0, n, batch):
        logits = masked_logits(net, d["obs"][i:i + batch], d["mask"][i:i + batch])
        pred[i:i + batch] = logits.argmax(-1).numpy()
    act = d["act"]
    hit = pred == act
    is_disp = (act >= 1) & (act <= max_pending)
    is_reloc = act > max_pending
    # How often the clone relocates when the LP relocated, and exactly where.
    reloc_as_reloc = is_reloc & (pred > max_pending)
    return {
        "overall": 100 * float(hit.mean()),
        "dispatch_only": 100 * float(hit[is_disp].mean()) if is_disp.any() else float("nan"),
        "reloc_only": 100 * float(hit[is_reloc].mean()) if is_reloc.any() else float("nan"),
        "reloc_target": (100 * float((pred[is_reloc] == act[is_reloc]).mean())
                         if is_reloc.any() else float("nan")),
        "reloc_precision": (100 * float(hit[is_reloc][reloc_as_reloc[is_reloc]].mean())
                            if reloc_as_reloc.any() else float("nan")),
        "pred_reloc_pct": 100 * float((pred > max_pending).mean()),
    }


def train_bc(d_train: dict, obs_size: int, action_dim: int, hidden_dim: int,
             max_pending: int, epochs: int, lr: float, batch_size: int,
             reloc_weight: float, seed: int) -> ZoneActorCritic:
    torch.manual_seed(seed)
    net = ZoneActorCritic(obs_size, action_dim, hidden_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.from_numpy(d_train["obs"].astype(np.float32))
    mask = torch.from_numpy(d_train["mask"].astype(bool))
    act = torch.from_numpy(d_train["act"].astype(np.int64))
    weight = torch.where(act > max_pending,
                         torch.tensor(float(reloc_weight)),
                         torch.tensor(1.0))
    n = len(act)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, _ = net(obs[idx], mask[idx])
            ce = torch.nn.functional.cross_entropy(
                logits, act[idx], reduction="none")
            loss = (ce * weight[idx]).sum() / weight[idx].sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        print(f"[bc] epoch {epoch + 1}/{epochs}  loss {total / count:.4f}")
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40,
                    help="train-split LP episodes to imitate")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--val-base-seed", type=int, default=12000,
                    help="must match diag_objective_gap's --base-seed")
    ap.add_argument("--val-episodes", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--reloc-weight", type=float, nargs="+", default=[1.0, 5.0],
                    help="one run dir per value; upweights relocate decisions")
    ap.add_argument("--data-dir", default="runs/bc-lp")
    ap.add_argument("--out-dir", default="runs/nyc16-bc-lp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data_dir)
    d_train = collect("train", args.episodes, args.base_seed,
                      data / f"train_{args.episodes}ep.npz")
    d_val = collect("val", args.val_episodes, args.val_base_seed,
                    data / f"val_{args.val_episodes}ep.npz")

    action_dim = int(d_train["mask"].shape[1])
    # 1 no-op + max_pending dispatch slots + one relocation action per zone.
    max_pending = (action_dim - 1) // 2
    obs_size = int(d_train["obs"].shape[1])
    print(f"[bc] obs_dim={obs_size} action_dim={action_dim} "
          f"max_pending={max_pending}")
    print(f"[bc] train windows seen: {len(np.unique(d_train['window']))}, "
          f"val windows: {len(np.unique(d_val['window']))}")
    print(f"[bc] class mix, train: {class_report(d_train['act'], max_pending)}")
    print(f"[bc] class mix, val:   {class_report(d_val['act'], max_pending)}")

    # Reference manifest: the RL arm this probe is compared against. Copied so
    # the BC checkpoint is evaluated under the exact same env config (D97
    # obs_inbound + h=24), which is the whole point of the comparison.
    ref_manifest = Path("runs/nyc16-h24-obs-s0/manifest.json")
    base_manifest = json.loads(ref_manifest.read_text())

    results = {}
    for weight in args.reloc_weight:
        tag = "" if weight == 1.0 else f"-rw{int(weight)}"
        out_dir = Path(f"{args.out_dir}{tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[bc] === reloc_weight={weight} -> {out_dir}")
        net = train_bc(d_train, obs_size, action_dim, args.hidden_dim,
                       max_pending, args.epochs, args.lr, args.batch_size,
                       weight, args.seed)
        acc_train = accuracy(net, d_train, max_pending)
        acc_val = accuracy(net, d_val, max_pending)
        print(f"[bc] accuracy train (in-sample): {acc_train}")
        print(f"[bc] accuracy val (teacher-forced): {acc_val}")

        # Package as a normal MAPPO run dir so the existing harness loads it.
        cfg = NYCConfig()
        env = RideGymAEC(cfg)
        env.reset(seed=0)
        # D42 layout, as built by experiments/train.py: the critic input is
        # [global state | own observation]. Using len(state()) alone here wrote
        # checkpoints whose critic could not be loaded by `train --resume`
        # (1464 vs 849 input width), so the warm start below was impossible
        # until this matched.
        state_dim = len(env.state()) + obs_size
        learner = MAPPO(
            observation_dim=obs_size, state_dim=state_dim, action_dim=action_dim,
            config=MAPPOConfig(hidden_dim=args.hidden_dim, arch="mlp"),
            num_zones=16, max_pending=max_pending,
            hist_len=int(env.observe(env.possible_agents[0])["demand_hist"].shape[1]),
        )
        learner.actor.load_state_dict(net.state_dict())
        torch.save(learner.state_dict(), out_dir / "checkpoint.pt")

        manifest = copy.deepcopy(base_manifest)
        manifest.update({
            "scenario_id": f"BC-lp-{tag or '-plain'}-nyc",
            "algo": "mappo",
            "iterations": 0,
            "trained_by": "behaviour cloning of ZoneLPRebalancer (no PPO)",
            "bc": {
                "teacher": "ZoneLPRebalancer() defaults",
                "train_episodes": args.episodes,
                "train_seeds": [args.base_seed, args.base_seed + args.episodes - 1],
                "val_episodes": args.val_episodes,
                "val_seeds": [args.val_base_seed,
                              args.val_base_seed + args.val_episodes - 1],
                "epochs": args.epochs, "lr": args.lr,
                "reloc_weight": weight, "hidden_dim": args.hidden_dim,
                "state_dim": state_dim, "actor_only_trained": True,
                "acc_train": acc_train, "acc_val": acc_val,
                "class_mix_train": class_report(d_train["act"], max_pending),
            },
        })
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[bc] wrote {out_dir}/checkpoint.pt + manifest.json")
        results[tag or "plain"] = {"acc_train": acc_train, "acc_val": acc_val}

    out = Path("results/bc_lp_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[bc] wrote {out}")


if __name__ == "__main__":
    main()
