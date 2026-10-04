"""RCA: can the reward even express "pick the BEST zone"?

Current state: RL reaches 37.19% precision (5.07x chance) while a supervised
probe on the SAME features reaches 87.5%. So the information is present and
learnable, but RL extracts less than half of it.

Candidate cause: the reward distinguishes "aimed at an empty zone" from "aimed
at a zone with riders", but NOT "aimed at the BEST zone" from "aimed at any
zone with riders". If so, RL can learn the coarse rule it IS rewarded for and
has almost no gradient separating the 13 non-empty zones from each other.

This is directly measurable: classify every relocator payout by aim quality and
compare payout rates. Earlier measurement on a pre-D59 checkpoint showed
good 58.6% / bad 58.1% / worst 37.8% -- i.e. the good-vs-bad gap was 0.5pp while
the empty-vs-nonempty gap was 20pp. If that holds now, the reward is a step
function on emptiness and the ranking is unlearnable from reward alone.

Also reports where the policy's aims actually land, to check whether it has
already mastered the part that IS rewarded.

    uv run python scripts/diag_reward_rank_resolution.py --episodes 3 \
        --ippo runs/d62-ippo-s0/checkpoint_best.pt
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
    ap.add_argument("--base-seed", type=int, default=8100)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split

    arms: list[tuple[str, object]] = []
    if args.ippo:
        l, _m, _ = load_trained_policy(Path(args.ippo))
        arms.append(("IPPO", l))
    if args.mappo:
        l, _m, _ = load_trained_policy(Path(args.mappo))
        arms.append(("MAPPO", l))
    if not arms:
        print("pass --ippo and/or --mappo")
        return

    for label, policy in arms:
        n_aim = Counter()
        n_paid = Counter()
        aim_quality = []      # demand at chosen / demand at best
        best_rank = []        # rank of chosen among legal, 0 = best

        for ep in range(args.episodes):
            env = RideGymAEC(cfg)
            mp = env._max_pending
            if hasattr(policy, "reset") and not hasattr(policy, "act"):
                policy.reset(seed=args.base_seed + ep)
            env.reset(seed=args.base_seed + ep)
            aim: dict[int, str] = {}

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
                    with torch.no_grad():
                        a, _, _ = policy.act(
                            obs_to_vector(env.observe(ag))[None, :], m[None, :],
                            deterministic=True)
                    a = int(a[0])
                    if not m[a]:
                        a = int(np.flatnonzero(m)[0])
                    d = env._decode_action(a, zi)
                    if isinstance(d, dict) and "relocate" in d and legal:
                        pick = int(d["relocate"])
                        lg = np.asarray(legal, dtype=int)
                        best = int(lg[int(np.argmax(pend[lg]))])
                        if pend[pick] <= 1e-9:
                            klass = "empty"
                        elif pick == best:
                            klass = "best"
                        else:
                            klass = "other_nonempty"
                        n_aim[klass] += 1
                        aim_quality.append(
                            float(pend[pick]) / max(float(pend[best]), 1e-9))
                        best_rank.append(int(np.where(
                            lg[np.argsort(-pend[lg])] == pick)[0][0]))
                        idle = [x for x in range(env._num_drivers)
                                if env._last_obs[x]["self"].get("status") == "idle"
                                and max(0, min(int(env._last_obs[x]["self"]
                                                   .get("current_region", 0)),
                                               env._num_zones - 1)) == zi]
                        if idle:
                            aim[idle[0]] = klass
                    before = set(env._order_relocator_paid)
                    env.step(a)
                    for oid in set(env._order_relocator_paid) - before:
                        did = getattr(env.env.orders[oid], "assigned_driver", None)
                        if did in aim:
                            n_paid[aim.pop(did)] += 1

        print("=" * 74)
        print(f"{label}: does the reward RANK destinations, or only flag empty?")
        print("=" * 74)
        print(f"  {'aim class':<16} {'aims':>7} {'paid':>7} {'paid rate':>11}")
        for k in ("best", "other_nonempty", "empty"):
            rate = 100.0 * n_paid[k] / max(n_aim[k], 1)
            print(f"  {k:<16} {n_aim[k]:7d} {n_paid[k]:7d} {rate:10.1f}%")
        g = 100.0 * n_paid["best"] / max(n_aim["best"], 1)
        o = 100.0 * n_paid["other_nonempty"] / max(n_aim["other_nonempty"], 1)
        e = 100.0 * n_paid["empty"] / max(n_aim["empty"], 1)
        print()
        print(f"  gap best vs other-nonempty : {g-o:+6.1f} pp   <- RANKING signal")
        print(f"  gap nonempty vs empty      : {o-e:+6.1f} pp   <- EMPTINESS signal")
        print()
        if aim_quality:
            print(f"  mean demand ratio at chosen target : "
                  f"{np.mean(aim_quality):.3f}")
            print(f"  mean rank of chosen among legal    : "
                  f"{np.mean(best_rank):.2f} (0 = best)")
            tot = sum(n_aim.values())
            print(f"  aims landing on an EMPTY zone      : "
                  f"{100*n_aim['empty']/max(tot,1):.1f}%")
        print()
        if abs(g - o) < 5.0 and abs(o - e) > 8.0:
            print("  ROOT CAUSE: the reward pays for AVOIDING EMPTY ZONES but")
            print("  not for choosing among non-empty ones. The ranking half of")
            print("  the decision has no gradient, so RL cannot learn it from")
            print("  reward -- while the emptiness half is learnable and learned.")
            print("  A supervised probe reaches 87.5% because it is given the")
            print("  answer directly rather than having to infer it from pay.")
        else:
            print("  The reward does separate ranking, so the gap is elsewhere.")
        print()


if __name__ == "__main__":
    main()