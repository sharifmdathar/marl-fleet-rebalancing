"""Is the policy reading demand at all? Dose-response, then a causal intervention.

The decision-time probe found the policy aims at zones holding 0.30 unmet
riders when zones holding 1.69 were on the menu (ratio 0.18). Two explanations
remain:

  (a) the policy is demand-responsive but weakly, and 6.4% vs 8.4% chance is
      just a shallow slope;
  (b) the policy is not reading demand at all, and targeting is unrelated to it.

A permutation test (shuffle the demand channel) is a weak discriminator: the
shuffled observation is off-manifold, so a null result cannot distinguish "blind"
from "reads it but shallowly". Two stronger tests:

  TEST 1 -- DOSE-RESPONSE. Bin every relocation by the unmet demand at the
  chosen target, and ask whether the chosen zone's demand rank among legal
  targets is better than uniform. Under (b) the rank distribution is flat; under
  (a) it slopes toward the neediest.

  TEST 2 -- CAUSAL INTERVENTION. Use `localhub` to put 100% of demand in a
  single zone, chosen per episode, then measure how often the policy aims at
  that zone. Demand is genuinely moved, the observation is in-distribution, and
  the expected effect size under (a) is enormous (chance ~1/15 = 6.7%). Under
  (b) the aim rate will not track which zone received the demand.

Test 2 is the decisive one: it manipulates the thing the policy allegedly reads,
and the answer does not depend on any model assumption.

    uv run python scripts/diag_demand_sensitivity.py --episodes 3 --mappo ...
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


def act_for(policy, env, ag, m):
    obs = obs_to_vector(env.observe(ag))
    if hasattr(policy, "act"):
        with torch.no_grad():
            a, _, _ = policy.act(obs[None, :], m[None, :], deterministic=False)
        return int(a[0])
    return int(policy.select_action(ag, env, env.observe(ag)))


def pend(env) -> np.ndarray:
    return np.asarray(env._pending_per_zone, dtype=float) * max(env._num_drivers, 1)


def test1_dose_response(policy, cfg, args) -> dict:
    """Rank of the chosen target among legal targets, by demand."""
    env = RideGymAEC(cfg)
    mp = env._max_pending
    ranks: list[float] = []
    chosen_demand: list[float] = []
    best_demand: list[float] = []

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
                p = pend(env)
                legal = env._legal_relocate_targets(zi)
                a = act_for(policy, env, ag, m)
                if not m[a]:
                    a = int(np.flatnonzero(m)[0])
                d = env._decode_action(a, zi)
                if isinstance(d, dict) and "relocate" in d and legal:
                    pick = int(d["relocate"])
                    lg = np.asarray(legal, dtype=int)
                    order = lg[np.argsort(-p[lg])]      # neediest first
                    ranks.append(float(np.where(order == pick)[0][0]))
                    chosen_demand.append(float(p[pick]))
                    best_demand.append(float(p[order[0]]))
                env.step(a)

    if not ranks:
        return {"n": 0}
    r = np.asarray(ranks)
    n_legal_mean = len(legal) if legal else 0
    # Under "blind", the chosen target's rank is uniform over the legal set, so
    # the normalised rank is uniform on [0,1) with mean 0.5.
    return {
        "n": len(r),
        "mean_rank_frac": float(np.mean(ranks)) / max(n_legal_mean - 1, 1)
        if n_legal_mean > 1 else 0.5,
        "frac_chose_top": float(np.mean(np.asarray(ranks) == 0)) * 100,
        "mean_chosen_demand": float(np.mean(chosen_demand)),
        "mean_best_demand": float(np.mean(best_demand)),
    }


def test2_causal(policy, cfg, args) -> dict:
    """Put 100% of demand in one zone per episode; does the policy aim there?"""
    probe = RideGymAEC(NYCConfig())
    n_zones = probe._num_zones
    aim_hits = 0
    moves = 0
    per_episode: list[tuple[int, int, int]] = []   # hub, moves, hits

    for k in range(args.episodes):
        hub = k % n_zones
        c = NYCConfig()
        c.nyc_split = "val"
        c.demand_shift_mode = "localhub"
        c.localhub_stage1_zone = hub
        # stage2 must differ (the generator validates it), but the switch is
        # set past the horizon so stage 2 never activates and demand stays in
        # `hub` for the whole episode.
        other = next((j for j in range(n_zones) if j != hub), 1)
        c.localhub_stage2_zone = other
        c.localhub_switch_min = 10_000.0     # never switches: hub all episode
        c.drift_source_zone = hub
        c.drift_dest_zone = hub
        env = RideGymAEC(c)
        mp = env._max_pending

        if hasattr(policy, "reset") and not hasattr(policy, "act"):
            policy.reset(seed=args.base_seed + k)
        env.reset(seed=args.base_seed + k)

        ep_moves = ep_hits = 0
        while env.agents:
            for _ in range(env._num_zones):
                if not env.agents:
                    break
                ag = env.agent_selection
                zi = env._zone_int_id(ag)
                m = env.action_mask(ag)
                legal = env._legal_relocate_targets(zi)
                a = act_for(policy, env, ag, m)
                if not m[a]:
                    a = int(np.flatnonzero(m)[0])
                d = env._decode_action(a, zi)
                if isinstance(d, dict) and "relocate" in d and legal:
                    ep_moves += 1
                    if int(d["relocate"]) == hub:
                        ep_hits += 1
                env.step(a)
        moves += ep_moves
        aim_hits += ep_hits
        per_episode.append((hub, ep_moves, ep_hits))

    chance = 100.0 / max(n_zones - 1, 1)
    return {
        "moves": moves,
        "hits": aim_hits,
        "rate": 100.0 * aim_hits / max(moves, 1),
        "chance": chance,
        "per_episode": per_episode,
    }


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
    print("Is the policy reading demand at all?")
    print("=" * 72)
    for label, policy in arms:
        d1 = test1_dose_response(policy, cfg, args)
        print(f"\n--- {label}")
        print("  TEST 1  dose-response (natural demand)")
        if d1["n"] == 0:
            print("    no moves observed")
        else:
            print(f"    moves                                  : {d1['n']}")
            print(f"    chose the neediest legal target        : "
                  f"{d1['frac_chose_top']:.2f}%  (chance ~6.7%)")
            print(f"    mean demand at chosen target          : "
                  f"{d1['mean_chosen_demand']:.2f}")
            print(f"    mean demand at best available target  : "
                  f"{d1['mean_best_demand']:.2f}")
            ratio = (d1["mean_chosen_demand"] / d1["mean_best_demand"]
                     if d1["mean_best_demand"] > 0 else float("nan"))
            print(f"    ratio                                  : {ratio:.2f}")

        d2 = test2_causal(policy, cfg, args)
        print("  TEST 2  causal: 100% of demand forced into one zone")
        print(f"    relocation moves                       : {d2['moves']}")
        print(f"    aimed at the zone holding the demand  : {d2['hits']} "
              f"({d2['rate']:.2f}%)")
        print(f"    chance                                 : {d2['chance']:.2f}%")
        print("    per episode (hub, moves, aimed-at-hub):")
        for hub, mv, ht in d2["per_episode"]:
            print(f"      hub zone {hub:2d}: {mv:3d} moves, {ht:3d} aimed at hub")
        print()
        if d2["moves"] < 20:
            print("    too few moves to conclude")
        elif d2["rate"] > d2["chance"] * 1.5:
            print("    -> policy IS demand-responsive. It aims at the needy zone")
            print("       far more than chance when demand is concentrated there.")
        else:
            print("    -> policy does NOT track where demand is. Moving 100% of")
            print("       demand to a zone does not make the policy aim at it.")


if __name__ == "__main__":
    main()
