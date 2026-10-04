"""Does the policy FAIL to relocate, or is there nothing to relocate?

D44 measured that relocation targeting sits at chance, and that 95.5% of the
turns where a far zone was NOT a legal target were turns where the acting zone
had no idle car at all. That left two candidate explanations, and they imply
opposite fixes:

  (a) "won't"  the policy has idle cars, has legal targets, and chooses not to
      move. Then the reward/credit path is still the problem.
  (b) "can't"  the policy almost never has an idle car at the moment it would
      need one. Then no reward change can fix it, and the action is simply not
      available.

This measures opportunity, not behaviour. For every zone-turn it records:

  * did the zone have >= 1 idle car?
  * how many relocation targets were legal?
  * did it relocate anyway?

and reports the relocation rate CONDITIONAL on having an idle car and at least
one legal target -- the only turns on which relocating was possible at all.

Also reports the opportunity supply: how often an idle car and a needy zone
coincide. A conditional rate near zero means (a); a tiny denominator means (b),
and both are reported so neither can be mistaken for the other.

    uv run python scripts/diag_reloc_opportunity.py --episodes 3
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


def run_arm(policy, cfg, args, label: str) -> dict:
    env = RideGymAEC(cfg)
    mp = env._max_pending
    n = env._num_zones

    zone_turns = 0
    with_idle = 0
    with_idle_and_target = 0
    relocated = 0
    relocated_when_possible = 0
    target_counts: list[int] = []
    idle_counts: list[int] = []

    for ep in range(args.episodes):
        if hasattr(policy, "reset") and not hasattr(policy, "act"):
            policy.reset(seed=args.base_seed + ep)
        env.reset(seed=args.base_seed + ep)
        while env.agents:
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            m = env.action_mask(ag)

            idle = int(env._zone_idle_counts[zi])
            legal = env._legal_relocate_targets(zi)
            n_legal = len(legal)

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

            did_reloc = a_val > mp
            zone_turns += 1
            idle_counts.append(idle)
            target_counts.append(n_legal)
            if idle > 0:
                with_idle += 1
            if idle > 0 and n_legal > 0:
                with_idle_and_target += 1
            if did_reloc:
                relocated += 1
                if idle > 0 and n_legal > 0:
                    relocated_when_possible += 1
            env.step(a_val)

    return {
        "label": label,
        "zone_turns": zone_turns,
        "with_idle": with_idle,
        "with_idle_and_target": with_idle_and_target,
        "relocated": relocated,
        "relocated_when_possible": relocated_when_possible,
        "mean_idle": float(np.mean(idle_counts)) if idle_counts else 0.0,
        "mean_targets": float(np.mean(target_counts)) if target_counts else 0.0,
        "pct_idle": 100.0 * with_idle / max(zone_turns, 1),
        "pct_opportunity": 100.0 * with_idle_and_target / max(zone_turns, 1),
        "cond_rate": 100.0 * relocated_when_possible / max(with_idle_and_target, 1),
        "overall_rate": 100.0 * relocated / max(zone_turns, 1),
    }


def report(r: dict) -> None:
    print(f"\n--- {r['label']}")
    print(f"  zone-turns                     : {r['zone_turns']}")
    print(f"  turns with >=1 idle car        : {r['with_idle']} "
          f"({r['pct_idle']:.1f}%)")
    print(f"  idle car AND >=1 legal target  : {r['with_idle_and_target']} "
          f"({r['pct_opportunity']:.1f}%)   <- OPPORTUNITY SUPPLY")
    print(f"  relocation actions taken       : {r['relocated']} "
          f"({r['overall_rate']:.2f}% of all zone-turns)")
    print(f"  relocated WHEN POSSIBLE        : {r['relocated_when_possible']}")
    print(f"  CONDITIONAL relocation rate    : {r['cond_rate']:.2f}%   "
          f"<- WILLINGNESS GIVEN THE CHOICE")
    print(f"  mean idle cars per zone-turn   : {r['mean_idle']:.2f}")
    print(f"  mean legal targets per zone-turn: {r['mean_targets']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"

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

    print("=" * 72)
    print("Is relocation not chosen, or not available?")
    print("=" * 72)
    for label, policy in arms:
        r = run_arm(policy, cfg, args, label)
        report(r)
        if r["pct_opportunity"] < 1.0:
            print(f"  VERDICT for {label}: OPPORTUNITY IS SCARCE. Relocating was")
            print("  possible in under 1% of zone-turns, so the low relocation rate")
            print("  is mostly explained by there being no car to send.")
        elif r["cond_rate"] < 5.0:
            print(f"  VERDICT for {label}: the choice WAS available and mostly")
            print("  declined. Opportunity is not the binding constraint.")
        else:
            print(f"  VERDICT for {label}: relocating is both possible and used.")
    print()
    print("Reading: OPPORTUNITY SUPPLY is how often a car and a legal target")
    print("coincided. CONDITIONAL RATE is what the policy did when it did.")
    print("A low conditional rate with plentiful opportunity points at the")
    print("reward/credit path. Plentiful opportunity with a low conditional rate")
    print("means the action is being declined on its merits.")


if __name__ == "__main__":
    main()
