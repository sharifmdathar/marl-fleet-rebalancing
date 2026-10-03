"""What reward does a zone controller actually receive, and does it reach the agent?

Delivery is measured at the only point that matters per the AEC contract:
``env.last()[1]``, i.e. ``_cumulative_rewards[agent]``, sampled BEFORE each
``step()`` consumes and resets it. The earlier "94.1% of reads are zero" figure
came from a probe reading ``env.rewards``, which is not what an AEC consumer
sums; that number is void.

The earlier version also ran a single 3-episode noop/greedy pass, which is
debugging output by AGENTS.md's own rule. This version runs every corrected
policy arm across >=5 paired seeds and reports the paired difference against the
no-relocation control, so a "zones are never paid" claim has something to be
measured against.

Self-check: every chosen action is asserted legal in the same mask and asserted
to decode as the policy declared.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diag_common import ObservedEnv, make_policies, paired, verify_action

from marl_fleet_rebalancing.config import NYCConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--zone-a", type=int, default=8)
    ap.add_argument("--zone-b", type=int, default=13)
    ap.add_argument("--pct", type=float, default=60.0)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split
    cfg.demand_shift_mode = "hotspot2"
    cfg.hotspot_zone_a = args.zone_a
    cfg.hotspot_zone_b = args.zone_b
    cfg.hotspot_pct = args.pct
    cfg.hotspot_split = 0.5

    env = ObservedEnv(cfg)
    policies = make_policies()
    arms = ["dispatch-only", "dispatch-first", "relocate-first-best", "random-legal"]
    seeds = [args.base_seed + i for i in range(args.episodes)]
    nz = env._num_zones

    print("scenario: hotspot2 A=%d B=%d pct=%.0f%%  split=%s  seeds=%s (%d paired "
          "episodes per arm)" % (args.zone_a, args.zone_b, args.pct, args.split,
                                 seeds, args.episodes))
    print("reward read point: env.last()[1] == _cumulative_rewards[agent], sampled "
          "before each step()\n")

    out: dict[str, dict] = {}
    for arm in arms:
        reads = Counter()
        zone_sum = defaultdict(float)
        zone_turns = Counter()
        zone_nonzero = Counter()
        per_ep_nonzero_frac: list[float] = []
        per_ep_total_reward: list[float] = []
        for seed in seeds:
            env.reset(seed=seed)
            env.reset_observer()
            rng = np.random.default_rng(seed * 1_000_003 + 7)
            nz_ep = 0
            tot_ep = 0.0
            n_ep = 0
            while env.agents:
                agent = env.agent_selection
                received = float(env.last(observe=False)[1])
                zi = env._zone_int_id(agent)
                mask = env.action_mask(agent)
                reads["reads"] += 1
                nz_ep += int(received != 0.0)
                n_ep += 1
                tot_ep += received
                reads["legal_dispatch"] += int(mask[1:1 + env._max_pending].sum())
                reads["legal_relocate"] += int(mask[1 + env._max_pending:].sum())
                reads["noop_legal"] += int(bool(mask[0]))
                zone_sum[zi] += received
                zone_turns[zi] += 1
                zone_nonzero[zi] += int(received != 0.0)
                action, intended = policies[arm](env, agent, rng)
                action = verify_action(env, agent, action, intended)
                env.step(action)
                env.observe_turn()
            per_ep_nonzero_frac.append(100.0 * nz_ep / max(n_ep, 1))
            per_ep_total_reward.append(tot_ep)
            kpi = env.kpi_summary()
            reads["served"] += int(kpi["served"])
            reads["empty_km"] += float(kpi["empty_km"])
        out[arm] = {
            "reads": reads,
            "zone_sum": dict(zone_sum),
            "zone_turns": dict(zone_turns),
            "zone_nonzero": dict(zone_nonzero),
            "nonzero_frac": per_ep_nonzero_frac,
            "total_reward": per_ep_total_reward,
        }

    print("=" * 96)
    print("REWARD DELIVERY: share of agent reads with a non-zero reward "
          "(mean +/- sd over %d paired episodes)" % args.episodes)
    print("=" * 96)
    print("%-24s %18s %16s %14s %14s" % ("arm", "nonzero reads %", "total/ep",
                                         "served", "empty km"))
    for arm in arms:
        o = out[arm]
        v = np.array(o["nonzero_frac"])
        t = np.array(o["total_reward"])
        print("%-24s %11.2f +/- %4.2f %16.2f %14d %14.1f"
              % (arm, v.mean(), v.std(ddof=1), t.mean(),
                 o["reads"]["served"], o["reads"]["empty_km"]))

    print("\nlegal-action census, summed over all zone-turns of all %d episodes:"
          % args.episodes)
    print("%-24s %14s %16s %12s" % ("arm", "dispatch legal", "relocate legal",
                                    "noop legal"))
    for arm in arms:
        r = out[arm]["reads"]
        print("%-24s %14d %16d %12d" % (arm, r["legal_dispatch"],
                                        r["legal_relocate"], r["noop_legal"]))

    print("\nPAIRED vs dispatch-only (never relocates), per-episode")
    base = out["dispatch-only"]
    for key, lbl, unit in (("nonzero_frac", "nonzero reads", "pp"),
                           ("total_reward", "total reward/ep", "")):
        print("  -- %s" % lbl)
        for arm in arms:
            if arm == "dispatch-only":
                continue
            p = paired(out[arm][key], base[key], lbl)
            if unit == "pp":
                print("     %-24s %+7.2fpp +/- %5.2f sd (sem %5.2f, n=%d, "
                      "%dW/%dL/%dT)"
                      % (arm, p["mean"], p["std"], p["sem"], p["n"], p["wins"],
                         p["losses"], p["ties"]))
            else:
                print("     %-24s %s" % (arm, fmt(p)))

    print("\nper-zone reward accounting (all %d episodes pooled)" % args.episodes)
    for arm in arms:
        o = out[arm]
        print("  -- %s" % arm)
        print("     zones with 0 reads          : %d"
              % sum(1 for z in range(nz) if o["zone_turns"].get(z, 0) == 0))
        print("     zones NEVER paid (had reads) : %d"
              % sum(1 for z in range(nz) if o["zone_turns"].get(z, 0) > 0
                    and o["zone_nonzero"].get(z, 0) == 0))
        print("     %-6s %12s %14s" % ("zone", "nonzero %", "mean reward"))
        for z in range(nz):
            t = max(o["zone_turns"].get(z, 0), 1)
            print("     %-6d %12.1f %14.4f"
                  % (z, 100.0 * o["zone_nonzero"].get(z, 0) / t,
                     o["zone_sum"].get(z, 0.0) / t))

    print("\nNOTE: a zero read is not a bug. team_share=%.2f blends every zone's "
          "turn reward\ntoward the global mean, and the pending-waiting and "
          "cancellation terms\nfire only on turns where that zone has waiting "
          "orders. A zone with no\npending orders and no empty move that turn has "
          "nothing to be paid for."
          % cfg.team_share)


def fmt(p: dict) -> str:
    return (f"{p['mean']:+9.2f} +/- {p['std']:7.2f} sd (sem {p['sem']:5.2f}, "
            f"n={p['n']}, {p['wins']}W/{p['losses']}L/{p['ties']}T)")


if __name__ == "__main__":
    main()