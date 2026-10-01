"""Is the relocator completion credit ever reachable?

Traces, per completed order: which zone dispatched it, whether that driver was
in _driver_relocator at completion time, and whether the two differ (the
condition for the credit to fire at all).
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = NYCConfig()
    env = RideGymAEC(cfg)
    policy, _m, _ = load_trained_policy(Path(args.checkpoint))

    stats = Counter()
    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        seen_before = set(env._seen_completed_ids)
        reloc_snapshot = dict(env._driver_relocator)
        for oid in env._newly_completed_ids():
            o = env.env.orders[oid]
            did = getattr(o, "assigned_driver", None)
            disp = env._order_dispatcher.get(oid)
            stats["completions"] += 1
            if did is None:
                stats["did_is_None"] += 1
                continue
            stats["relocator_map_size_at_completion"] += len(reloc_snapshot)
            stats["did_in_relocator_map"] += int(did in reloc_snapshot)
            if did in reloc_snapshot:
                r = reloc_snapshot[did]
                if disp is None:
                    stats["relocator_present_but_no_dispatcher"] += 1
                elif r != disp:
                    stats["CREDIT_FIRES"] += 1
                else:
                    stats["relocator_equals_dispatcher"] += 1
        # re-seed seen set to what the real call expects
        env._seen_completed_ids = seen_before
        return orig(events, dispatched_this_turn)

    env._compute_zone_rewards = traced

    for i in range(args.episodes):
        env.reset(seed=args.base_seed + i)
        while env.agents:
            agent = env.agent_selection
            obs = obs_to_vector(env.observe(agent))
            mask = env.action_mask(agent)
            with torch.no_grad():
                act, _, _ = policy.act(
                    obs[None, :], mask[None, :], deterministic=False,
                )
            a = int(act[0])
            if not mask[a]:
                a = int(np.flatnonzero(mask)[0])
            env.step(a)

    print("\ncompletion attribution, over %d episodes:" % args.episodes)
    for k, v in sorted(stats.items()):
        print("  %-42s %d" % (k, v))
    comp = stats["completions"]
    print()
    print("completions total          : %d" % comp)
    print("credit fired               : %d  (%.1f%% of completions)"
          % (stats["CREDIT_FIRES"], 100 * stats["CREDIT_FIRES"] / max(comp, 1)))
    print("did present in reloc map   : %d  (%.1f%%)"
          % (stats["did_in_relocator_map"], 100 * stats["did_in_relocator_map"] / max(comp, 1)))
    print("relocator == dispatcher    : %d" % stats["relocator_equals_dispatcher"])


if __name__ == "__main__":
    main()