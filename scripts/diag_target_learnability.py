"""Is the information needed for good targeting actually LEARNABLE?

Two probes so far have narrowed the problem:

  * `diag_logit_sensitivity.py` -- the logits move a lot when demand inputs
    change, but the argmax relocation target never does. The policy reads
    demand MAGNITUDE without binding it to a zone.
  * `diag_reward_targeting_sensitivity.py` -- aiming at a zone with unmet
    demand is paid 58.6% of the time vs 37.8% for an empty zone. So the reward
    does reward coarse targeting, and there is gradient the policy is leaving
    on the table.

Before building a destination-conditioned encoder, check the precondition: can
ANY model recover the best target from the features the env already emits? If a
small supervised probe cannot, the features are insufficient and the encoder
change will not help.

This trains that probe on real observations, real legal-target menus and real
demand, and reports top-1 accuracy against the per-move chance rate. It is a
deliberately generous test: full observation vector, no architecture
constraints, no RL, plenty of data.

    uv run python scripts/diag_target_learnability.py --samples 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

A0 = 6
NZ = 16
MP = 16


def collect(cfg, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(obs, label) where label = index of the legal target with most demand."""
    env = RideGymAEC(cfg)
    env.reset(seed=seed)
    mp = env._max_pending
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    ep = 0
    while len(X) < n_samples:
        if not env.agents:
            ep += 1
            env.reset(seed=seed + ep)
            continue
        for _ in range(env._num_zones):
            if not env.agents:
                break
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            m = env.action_mask(ag)
            legal = env._legal_relocate_targets(zi)
            if legal:
                pend = np.asarray(env._pending_per_zone, dtype=float) \
                    * max(env._num_drivers, 1)
                lg = np.asarray(legal, dtype=int)
                best = int(lg[int(np.argmax(pend[lg]))])
                if pend[best] > 1e-9:      # only when a needy target exists
                    v = obs_to_vector(env.observe(ag)).copy()
                    # Append the legal-target mask so the probe knows the menu.
                    X.append(np.concatenate([v, np.array(
                        [1.0 if z in set(legal) else 0.0 for z in range(NZ)])]))
                    lab = np.full(NZ, -1.0)
                    lab[best] = 1.0
                    Y.append(lab)
            d = np.flatnonzero(m[1: 1 + mp])
            a = int(d[0]) + 1 if d.size else 0
            if not m[a]:
                a = int(np.flatnonzero(m)[0])
            env.step(a)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=8000)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    print("collecting observations...")
    X, Y = collect(cfg, args.samples, args.base_seed)
    print(f"  samples: {len(X)}   feature dim: {X.shape[1]}")

    # Split by time, not randomly, to avoid leakage from adjacent frames.
    cut = int(0.8 * len(X))
    Xtr, Xte = X[:cut], X[cut:]
    Ytr, Yte = Y[:cut], Y[cut:]
    if len(Xte) < 200:
        print("not enough held-out samples")
        return

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(
        nn.Linear(X.shape[1], 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, NZ),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Keep the legal-target mask as a trailing NUMPY block and slice it on CPU,
    # then move the mask in with the batch. Indexing a CUDA tensor with a CUDA
    # index tensor built this way corrupted the context; the mask is 16 floats
    # wide and free to keep on the host.
    mask_tr = Xtr[:, -NZ:]
    mask_np = Xte[:, -NZ:]

    bs = 512
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            sel = perm[i:i + bs]
            xb = torch.as_tensor(Xtr[sel], device=dev)
            mb = torch.as_tensor(mask_tr[sel], device=dev)
            yb = torch.as_tensor(Ytr[sel], device=dev)
            logits = net(xb).masked_fill(mb < 0.5, -1e9)
            loss = nn.functional.cross_entropy(logits, yb.argmax(1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, len(Xte), 1024):
            xb = torch.as_tensor(Xte[i:i + 1024], device=dev)
            mb = torch.as_tensor(mask_np[i:i + 1024], device=dev)
            pred.append(net(xb).masked_fill(mb < 0.5, -1e9)
                        .argmax(1).cpu().numpy())
        pred = np.concatenate(pred)
    true = Yte.argmax(1)
    menu_size = mask_np.sum(1)
    acc = float((pred == true).mean()) * 100
    chance = float(np.mean(1.0 / np.maximum(menu_size, 1))) * 100

    print()
    print("=" * 70)
    print("Can a supervised probe recover the best target from the features?")
    print("=" * 70)
    print(f"  held-out samples     : {len(Xte)}")
    print(f"  mean menu size       : {menu_size.mean():.2f} zones")
    print(f"  probe top-1 accuracy : {acc:.2f}%")
    print(f"  chance (1/menu)      : {chance:.2f}%")
    print(f"  lift over chance     : {acc/chance:.2f}x")
    print()
    if acc > chance * 1.5:
        print("  VERDICT: the information IS present and learnable from the")
        print("  current observation. A destination-conditioned encoder has")
        print("  something to learn, so the architecture change is justified.")
    else:
        print("  VERDICT: the features do NOT support good targeting. A")
        print("  destination-conditioned encoder would not help; the")
        print("  observation itself would have to change.")


if __name__ == "__main__":
    main()
