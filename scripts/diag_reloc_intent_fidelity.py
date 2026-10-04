"""Does the car arrive where the agent CHOSE to send it?

The policy picks an action index; the wrapper decodes that to a target zone and
passes a destination point to RideGym. If the decode or the assignment drifts,
the agent may be aiming correctly and the simulation sending the car somewhere
else -- which would look identical to "targeting at chance" from every metric
measured so far, and would send the whole investigation in the wrong direction.

This compares, per relocation:
  * the zone the agent's action index decodes to (the INTENT)
  * the zone the driver was last observed in (the OUTCOME)

and reports the agreement rate, plus the confusion cases. Reads only; never
mutates driver state.

    uv run python scripts/diag_reloc_intent_fidelity.py --episodes 3 \
        --ippo runs/d51-ippo-s0/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
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
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--tol", type=float, default=0.75,
                    help="km slack when matching an outcome point to a zone")
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    mp_env = RideGymAEC(cfg)
    mp = mp_env._max_pending
    n_zones = mp_env._num_zones

    arms: list[tuple[str, object]] = []
    if args.ippo:
        learner, _m, _ = load_trained_policy(Path(args.ippo))
        arms.append(("IPPO", learner))
    if args.mappo:
        learner, _m, _ = load_trained_policy(Path(args.mappo))
        arms.append(("MAPPO", learner))
    if not arms:
        print("pass --ippo and/or --mappo")
        return

    for label, policy in arms:
        env = RideGymAEC(cfg)
        cents = np.asarray(env.env.relocation_points, dtype=float)

        agree = 0
        mismatch = 0
        never_arrived = 0
        confusion: Counter[tuple[int, int]] = Counter()
        intended: Counter[int] = Counter()

        for ep in range(args.episodes):
            if hasattr(policy, "reset") and not hasattr(policy, "act"):
                policy.reset(seed=args.base_seed + ep)
            env.reset(seed=args.base_seed + ep)
            # did -> zone the agent chose for it this episode
            intent: dict[int, int] = {}
            arrived: dict[int, int] = {}

            while env.agents:
                ag = env.agent_selection
                zi = env._zone_int_id(ag)
                m = env.action_mask(ag)
                obs = obs_to_vector(env.observe(ag))
                if hasattr(policy, "act"):
                    with torch.no_grad():
                        act, _, _ = policy.act(obs[None, :], m[None, :],
                                               deterministic=False)
                    a_val = int(act[0])
                else:
                    a_val = int(policy.select_action(ag, env, env.observe(ag)))
                if not m[a_val]:
                    a_val = int(np.flatnonzero(m)[0])

                decoded = env._decode_action(a_val, zi)
                if isinstance(decoded, dict) and "relocate" in decoded:
                    # Which driver would this move? Same selection order the
                    # wrapper uses: idle drivers of this zone, FIFO.
                    idle = [d for d in range(env._num_drivers)
                            if env._last_obs[d]["self"]["status"] == "idle"
                            and max(0, min(int(env._last_obs[d]["self"]
                                               .get("current_region", 0)),
                                           n_zones - 1)) == zi]
                    tgt = int(decoded["relocate"])
                    if idle:
                        intent[idle[0]] = tgt
                        intended[tgt] += 1
                env.step(a_val)

                # Observe arrivals: a relocating driver that is no longer
                # relocating has reached its destination.
                for did in range(env._num_drivers):
                    d = env._last_obs[did]["self"]
                    st = d.get("status")
                    if did in intent and did not in arrived and st == "idle":
                        pos = np.asarray(d.get("location", [0.0, 0.0]),
                                         dtype=float)
                        dist = np.linalg.norm(cents - pos[None, :], axis=1)
                        arrived[did] = int(np.argmin(dist))

            for did, want in intent.items():
                got = arrived.get(did)
                if got is None:
                    never_arrived += 1
                elif got == want:
                    agree += 1
                else:
                    mismatch += 1
                    confusion[(want, got)] += 1

        total = agree + mismatch + never_arrived
        print("=" * 72)
        print(f"{label}: does the car arrive where the agent aimed?")
        print("=" * 72)
        print(f"relocations issued                  : {total}")
        print(f"  arrived in the intended zone      : {agree} "
              f"({100*agree/max(total,1):.1f}%)")
        print(f"  arrived somewhere ELSE            : {mismatch} "
              f"({100*mismatch/max(total,1):.1f}%)")
        print(f"  never observed arriving (episode end): {never_arrived} "
              f"({100*never_arrived/max(total,1):.1f}%)")
        if confusion:
            print("\n  top aim->outcome confusions (aimed, landed):")
            for (w, g), n in confusion.most_common(8):
                print(f"     zone {w:2d} -> zone {g:2d}   n={n}")
        if total:
            print()
            if 100 * agree / total > 95:
                print("  VERDICT: decode is faithful. The car lands where the agent")
                print("  aimed, so aiming at chance is a POLICY problem, not a bug.")
            else:
                print("  VERDICT: decode/assignment is lossy. The agent is aiming")
                print("  somewhere other than where the car ends up.")


if __name__ == "__main__":
    main()
