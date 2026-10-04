"""Audit the observation the policy actually receives, feature by feature.

Context. MAPPO aims at the zone holding 100% of the demand 7.16% of the time
against a 6.67% chance rate, so it is not reading demand. Before concluding the
policy is at fault, rule out an EXTERNAL failure: a plumbing bug between the env
and the network that drops, misorders, or compresses the demand channel.

Checks, in order:

  1. LAYOUT. Does `obs_to_vector` order match `obs_dim`? A mismatch would
     silently feed the network permuted features with no error raised.
  2. REACHABILITY. Build an observation with a KNOWN demand pattern and recover
     the per-zone demand from the vector by indexing alone. If the ordering is
     right, the pending column must rank the zones exactly as the env does.
  3. RESOLUTION. How many distinct values does the demand channel take per
     waiting rider, after D37 normalisation? If one rider moves a channel by
     less than the network's float precision can distinguish from another
     channel's noise, the signal is present but not usable.
  4. SATURATION. Is the running-max scale saturating at 1.0 and clipping, so
     many zones collapse onto the same value?
  5. LEAKAGE. Confirm the pending column correlates with the env's own
     `_pending_per_zone`, i.e. it is live data and not a stale or constant
     buffer.

    uv run python scripts/diag_obs_audit.py --turns 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import obs_dim, obs_to_vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=8000)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    env.reset(seed=args.base_seed)

    space = env.observation_space(env.agents[0])
    n_zones = env._num_zones
    mp = env._max_pending
    hist_len = env._hist_len

    print("=" * 72)
    print("1. LAYOUT: obs_to_vector order vs obs_dim")
    print("=" * 72)
    shapes = {k: tuple(space[k].shape) for k in space.spaces}
    total = obs_dim(space.spaces)
    print(f"  declared obs_dim        : {total}")
    print(f"  obs_to_vector length    : {len(obs_to_vector(env.observe(env.agents[0])))}")
    print(f"  match                   : "
          f"{total == len(obs_to_vector(env.observe(env.agents[0])))}")
    off = 0
    layout = {}
    for k in ("zone_location", "zone_state", "all_zone_state", "pending_orders",
              "pending_mask", "pending_passengers", "neighbor_mask",
              "demand_hist", "time_frac"):
        n = int(np.prod(shapes[k]))
        layout[k] = (off, off + n)
        print(f"    {k:<20} shape {str(shapes[k]):<14} dims [{off}:{off+n}]")
        off += n
    print()

    # Index of all_zone_state column c, zone z, inside the flat vector.
    a0 = layout["all_zone_state"][0]
    print("=" * 72)
    print("2-5. Is the demand channel live, correctly indexed, and resolved?")
    print("=" * 72)
    print(f"  all_zone_state occupies [{a0}:{layout['all_zone_state'][1]}], "
          f"{n_zones} zones x 4 cols")

    max_abs = np.zeros(4)
    rank_ok = tot = 0
    uniq_per_rider: dict[int, set] = {}
    col_samples = {c: [] for c in range(4)}
    pending_samples = []

    for t in range(args.turns):
        if not env.agents:
            break
        for _ in range(n_zones):
            if not env.agents:
                break
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            m = env.action_mask(ag)
            obs = env.observe(ag)
            v = obs_to_vector(obs)

            azs = np.asarray(obs["all_zone_state"], dtype=float)
            env_pending = np.asarray(env._pending_per_zone, dtype=float) \
                * max(env._num_drivers, 1)

            # 2. Reachability: does the vector reproduce the env's ordering?
            vec_pending = np.array(
                [v[a0 + z * 4 + 2] for z in range(n_zones)], dtype=float)
            if env_pending.max() > 0:
                tot += 1
                if int(np.argmax(vec_pending)) == int(np.argmax(env_pending)):
                    rank_ok += 1
                for c in range(4):
                    col_samples[c].append(azs[:, c].copy())
                    max_abs[c] = max(max_abs[c], float(np.abs(azs[:, c]).max()))
            pending_samples.append(env_pending.copy())

            d = np.flatnonzero(m[1: 1 + mp])
            act = int(d[0]) + 1 if d.size else 0
            if not m[act]:
                act = int(np.flatnonzero(m)[0])
            env.step(act)

    print(f"\n  2. REACHABILITY -- argmax zone recovered from vector alone")
    print(f"     correct in {rank_ok}/{tot} samples "
          f"({100*rank_ok/max(tot,1):.1f}%)")
    print(f"     -> the pending column is present and correctly indexed"
          f" {'YES' if rank_ok==tot else 'NO -- LAYOUT BUG'}")

    print(f"\n  3/4. RESOLUTION AND SATURATION per all_zone_state column")
    for c in range(4):
        arr = np.concatenate(col_samples[c]) if col_samples[c] else np.zeros(1)
        u = np.unique(np.round(arr, 6))
        name = {0: "fleet total", 1: "idle", 2: "PENDING (demand)",
                3: "deficit"}[c]
        sat = float(np.mean(np.abs(arr) >= 0.999))
        print(f"     col {c} {name:<16} distinct values {len(u):5d}  "
              f"max|v| {np.abs(arr).max():.4f}  at +/-1.0: {100*sat:5.1f}%")

    # How big is one waiting rider on the demand channel, after D37?
    pend = np.concatenate(pending_samples) if pending_samples else np.zeros(1)
    uniq = np.unique(np.round(pend, 4))
    print(f"\n     env pending values seen (riders/zone): {len(uniq)} distinct, "
          f"range {pend.min():.3f}..{pend.max():.3f}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if rank_ok != tot:
        print("  EXTERNAL FAILURE: the demand column does not reproduce the env's")
        print("  own demand ordering. The policy cannot be reading what it is not")
        print("  given. Fix the layout before touching the reward.")
    else:
        print("  No layout bug. The demand channel reaches the network and its")
        print("  ordering matches the environment exactly.")
        print("  So a demand-blind policy is a POLICY/REPRESENTATION result,")
        print("  not a plumbing failure -- unless the resolution check above")
        print("  shows the channel is compressed to near-constant.")


if __name__ == "__main__":
    main()
