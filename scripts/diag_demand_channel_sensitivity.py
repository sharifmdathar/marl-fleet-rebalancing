"""Throwaway: is the actor even SENSITIVE to the demand channel it was given?

The oracle-forecast probe injected a perfect forecast into `demand_hist` and
nothing changed. That has two possible causes:

  (a) our injection never reached the network, or
  (b) the trained actor does not respond to that slice at all.

This separates them without retraining. It perturbs ONE channel of the
observation at a time by a large amount and measures how far the actor's
relocation logits move. A channel the actor uses produces a big shift; an
unused channel produces ~0.

Run: uv run python scripts/diag_demand_channel_sensitivity.py
"""
from __future__ import annotations

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
from diag_oracle_forecast import slice_table  # noqa: E402


def net_of(policy):
    """IPPO stores the shared net as `.policy`; MAPPO as `.actor`."""
    return getattr(policy, "policy", None) or policy.actor


def reloc_logits(policy, vec, mask, mp, n_zones):
    """Logits over relocation destinations only (action indices > max_pending)."""
    net = net_of(policy)
    dev = next(net.parameters()).device
    with torch.no_grad():
        logits, _ = net(
            torch.as_tensor(vec, dtype=torch.float32, device=dev)[None, :],
            torch.as_tensor(mask, dtype=torch.bool, device=dev)[None, :],
        )
    return logits[0, 1 + mp:].detach().cpu().numpy()


def main() -> None:
    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    for k, p in enumerate(("runs/d43-ippo-s0/checkpoint_best.pt",
                           "runs/d43-mappo-s0/checkpoint_best.pt")):
        policy, man, _ = load_trained_policy(Path(p))
        bounds = slice_table(env._num_zones, env._max_pending, env._hist_len)
        env.reset(seed=1000 + k)

        # Collect real observations from real turns.
        obs_list = []
        for _ in range(40):
            if env.agents:
                ag = env.agent_selection
                m = env.action_mask(ag)
                obs_list.append((obs_to_vector(env.observe(ag)), m.copy(),
                                 int(env._zone_int_id(ag))))
                d = np.flatnonzero(m[1:1 + env._max_pending])
                env.step(int(d[0]) + 1 if len(d) else 0)
            else:
                break

        lo, hi = bounds["demand_hist"]
        n_legal = float(np.mean([m[1 + env._max_pending:].sum() for _, m, _ in obs_list]))
        print(f"\n=== {Path(p).parent.name} ===")
        print(f"  obs {obs_list[0][0].shape[0]} dims, demand_hist [{lo}:{hi}], "
              f"mean legal reloc targets {n_legal:.1f}")

        # Max possible logit swing per channel, over all sampled observations.
        for name in ("demand_hist", "all_zone_state", "zone_state",
                     "neighbor_mask", "pending_orders"):
            s, e = bounds[name]
            deltas = []
            for vec, m, z in obs_list:
                base = reloc_logits(policy, vec, m, env._max_pending, env._num_zones)
                legal = m[1 + env._max_pending:]
                if not legal.any():
                    continue
                for val in (0.0, 1.0):
                    pert = vec.copy()
                    pert[s:e] = val
                    alt = reloc_logits(policy, pert, m, env._max_pending,
                                       env._num_zones)
                    deltas.append(np.abs(alt - base)[legal].max())
            # Spread among the LEGAL logits on the current turn. Illegal ones
            # are -inf after masking and carry no information.
            spreads = []
            for vec, m, z in obs_list:
                lg = reloc_logits(policy, vec, m, env._max_pending, env._num_zones)
                legal = m[1 + env._max_pending:]
                v = lg[legal]
                if v.size:
                    spreads.append(float(v.max() - v.min()))
            scale = float(np.mean(spreads)) if spreads else float("nan")
            mean_d = float(np.mean(deltas)) if deltas else float("nan")
            print(f"  {name:18s} mean|max dlogit| = {mean_d:.5f}   "
                  f"(mean legal-logit spread {scale:.4f}, "
                  f"ratio {mean_d / scale:.3f})")


if __name__ == "__main__":
    main()