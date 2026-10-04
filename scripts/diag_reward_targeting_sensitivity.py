"""Does the reward distinguish GOOD targeting from BAD targeting?

This is the question that decides whether an architecture change can help.

D48 pays the relocating zone when its car is DISPATCHED to a rider.
`diag_reloc_leg_outcome.py` measured that 7 of 7 relocation legs were cut short
by an opportunistic dispatch -- the car was handed to a rider en route, at a
location that had nothing to do with the zone the policy aimed at.

If that is the rule, then sending a car to the neediest zone and sending it to
a random empty zone earn the SAME credit, provided a rider is picked up either
way. In that case the reward is blind to targeting quality, and no amount of
destination-conditioning in the encoder can help: the policy would be optimising
correctly and still land at chance.

So measure it directly. For every relocator payout, classify the AIM:

    good : the aimed zone held the most unmet demand among legal targets
    bad  : it did not
    worst: the aimed zone had NO unmet demand at all

and report how often each class is paid. If the rates match, the reward does not
care where the car went.

    uv run python scripts/diag_reward_targeting_sensitivity.py --episodes 3
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

    for label, policy in arms:
        env = RideGymAEC(cfg)
        mp = env._max_pending

        # (did) -> (aim_zone, aim_class) recorded at decision time
        aim: dict[int, tuple[int, str]] = {}
        n_aim = {"good": 0, "bad": 0, "worst": 0}
        n_paid = {"good": 0, "bad": 0, "worst": 0}
        paid_amount = {"good": 0.0, "bad": 0.0, "worst": 0.0}

        for ep in range(args.episodes):
            if hasattr(policy, "reset") and not hasattr(policy, "act"):
                policy.reset(seed=args.base_seed + ep)
            env.reset(seed=args.base_seed + ep)

            while env.agents:
                for _ in range(env._num_zones):
                    if not env.agents:
                        break
                    ag = env.agent_selection
                    zi = env._zone_int_id(ag)
                    m = env.action_mask(ag)
                    pend = np.asarray(env._pending_per_zone, dtype=float) \
                        * max(env._num_drivers, 1)
                    legal = env._legal_relocate_targets(zi)

                    obs = obs_to_vector(env.observe(ag))
                    with torch.no_grad():
                        act, _, _ = policy.act(obs[None, :], m[None, :],
                                               deterministic=False)
                    a = int(act[0])
                    if not m[a]:
                        a = int(np.flatnonzero(m)[0])

                    d = env._decode_action(a, zi)
                    if isinstance(d, dict) and "relocate" in d and legal:
                        pick = int(d["relocate"])
                        lg = np.asarray(legal, dtype=int)
                        if pend[pick] <= 1e-9:
                            klass = "worst"
                        elif pick == int(lg[int(np.argmax(pend[lg]))]):
                            klass = "good"
                        else:
                            klass = "bad"
                        n_aim[klass] += 1
                        idle = [x for x in range(env._num_drivers)
                                if env._last_obs[x]["self"].get("status") == "idle"
                                and max(0, min(int(env._last_obs[x]["self"]
                                                   .get("current_region", 0)),
                                               env._num_zones - 1)) == zi]
                        if idle:
                            aim[idle[0]] = (pick, klass)

                    # Detect the payout: a relocator record appears at dispatch.
                    before = set(env._order_relocator_paid)
                    env.step(a)
                    for oid in set(env._order_relocator_paid) - before:
                        did = getattr(env.env.orders[oid], "assigned_driver", None)
                        if did in aim:
                            _pick, klass = aim.pop(did)
                            n_paid[klass] += 1
                            paid_amount[klass] += cfg.relocator_share * cfg.gamma

        print("=" * 72)
        print(f"{label}: does the reward distinguish good targeting from bad?")
        print("=" * 72)
        print(f"  {'aim class':<10} {'aims':>7} {'paid':>7} {'paid rate':>11}")
        for k in ("good", "bad", "worst"):
            rate = 100.0 * n_paid[k] / max(n_aim[k], 1)
            print(f"  {k:<10} {n_aim[k]:7d} {n_paid[k]:7d} {rate:10.1f}%")
        print()
        good = 100.0 * n_paid["good"] / max(n_aim["good"], 1)
        bad = 100.0 * n_paid["bad"] / max(n_aim["bad"], 1)
        worst = 100.0 * n_paid["worst"] / max(n_aim["worst"], 1)
        spread = max(good, bad, worst) - min(good, bad, worst)
        print(f"  spread between best and worst aim: {spread:.1f} pp")
        print()
        if spread < 5.0:
            print("  VERDICT: THE REWARD IS BLIND TO TARGETING. Aiming at the")
            print("  neediest zone and aiming at an empty zone earn the same")
            print("  credit at the same rate. No encoder change can fix this,")
            print("  because the policy is optimising correctly.")
        else:
            print("  VERDICT: the reward does reward better targeting, so an")
            print("  encoder that can express it has something to learn.")


if __name__ == "__main__":
    main()
