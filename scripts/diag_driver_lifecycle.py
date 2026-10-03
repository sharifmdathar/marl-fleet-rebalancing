"""Follow drivers turn by turn: idle -> relocating -> idle? Is anyone stuck?

Two claims this file re-measures:
  * "44.5% of driver-time is spent relocating"
  * "one driver stayed permanently in `relocating` status for a whole episode"

Both were produced by the degenerate relocate-only policy, which can never
dispatch. This runs the same census across all corrected arms on identical
seeds, so the number can be attributed to the policy or to the env.

Method: status is sampled once per turn (not once per agent step, which would
weight every turn by the number of zones) from env._last_obs. A "streak" is a
run of consecutive turns one driver spent in `relocating`; the episode has
horizon/dt + 1 sampled turns, so a streak equal to that length is a genuinely
permanently-relocating driver.

Self-check: every chosen action is asserted legal in the same mask and asserted
to decode as the policy declared.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diag_common import ObservedEnv, make_policies, paired

from marl_fleet_rebalancing.config import NYCConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--zone-a", type=int, default=8)
    ap.add_argument("--zone-b", type=int, default=13)
    ap.add_argument("--pct", type=float, default=60.0)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--trace-driver", type=int, default=-1,
                    help="-1 traces the first driver that ever starts relocating")
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split
    cfg.demand_shift_mode = "hotspot2"
    cfg.hotspot_zone_a = args.zone_a
    cfg.hotspot_zone_b = args.zone_b
    cfg.hotspot_pct = args.pct
    cfg.hotspot_split = 0.5

    print("config: allow_relocating_dispatch=%s neighbor_only=%s timeout=%.1f"
          % (cfg.allow_relocating_dispatch, cfg.neighbor_relocation_only,
             cfg.order_timeout))
    print("        require_destination_demand=%s claim_targets=%s team_share=%.2f"
          % (cfg.require_destination_demand, cfg.claim_relocation_targets,
             cfg.team_share))
    print("        (allow_relocating_dispatch=True means a RELOCATING driver is "
          "eligible for\n         dispatch, so a relocating car is not frozen. "
          "pettingzoo_env.py:778-789,\n         :1019-1020, :1041-1043.)")

    env = ObservedEnv(cfg)
    policies = make_policies()
    arms = ["dispatch-only", "dispatch-first", "relocate-first-best",
            "random-legal", "relocate-only(BROKEN)"]
    seeds = [args.base_seed + i for i in range(args.episodes)]
    turns = int(cfg.horizon / cfg.dt) + 1

    print("\nepisode length: %d sampled turns x %d drivers" % (turns, cfg.num_drivers))

    stats: dict[str, dict] = {}
    for arm in arms:
        census: dict[str, int] = {}
        streaks: list[int] = []
        worst_frac: list[float] = []
        stuck_episodes = 0
        per_ep_reloc_time: list[float] = []
        for seed in seeds:
            env.reset(seed=seed)
            env.reset_observer()
            rng = np.random.default_rng(seed * 1_000_003 + 7)
            while env.agents:
                agent = env.agent_selection
                action, intended = policies[arm](env, agent, rng)
                from diag_common import verify_action
                action = verify_action(env, agent, action, intended)
                env.step(action)
                env.observe_turn()
            for k, v in env.status_samples.items():
                census[k] = census.get(k, 0) + v
            streaks.append(env.longest_reloc_streak())
            worst_frac.append(env.max_driver_reloc_frac())
            per_ep_reloc_time.append(env.reloc_driver_time())
            if env.longest_reloc_streak() >= turns:
                stuck_episodes += 1
        stats[arm] = {
            "census": census,
            "streaks": streaks,
            "worst_frac": worst_frac,
            "stuck": stuck_episodes,
            "reloc_time": per_ep_reloc_time,
        }
        total = sum(census.values())
        print("\n=== %s ===" % arm)
        print("  driver status census (turn x driver samples, all %d episodes):"
              % args.episodes)
        for k, v in sorted(census.items(), key=lambda kv: -kv[1]):
            print("    %-14s %8d  %5.1f%%" % (k, v, 100.0 * v / max(total, 1)))
        rt = np.array(per_ep_reloc_time)
        print("  driver-time spent relocating: %.1f%% +/- %.1f (sd), per episode"
              % (100 * rt.mean(), 100 * rt.std(ddof=1)))
        print("  longest single-driver relocating streak: mean %.1f turns, max %d "
              "(episode is %d turns)" % (np.mean(streaks), max(streaks), turns))
        print("  worst single driver relocating fraction: %.1f%%"
              % (100 * np.mean(worst_frac)))
        print("  episodes with a PERMANENTLY relocating driver (streak == episode "
              "length): %d / %d" % (stuck_episodes, args.episodes))

    print("\n" + "=" * 88)
    print("PAIRED driver-time-spent-relocating, each arm vs dispatch-only "
          "(never relocates)")
    print("=" * 88)
    base = stats["dispatch-only"]["reloc_time"]
    for arm in arms:
        if arm == "dispatch-only":
            continue
        p = paired(stats[arm]["reloc_time"], base, "reloc_time")
        print("  %-22s %s" % (arm, fmt(p)))

    print("\n" + "=" * 88)
    print("PAIRED worst-single-driver relocating fraction, each arm vs dispatch-only")
    print("=" * 88)
    base = stats["dispatch-only"]["worst_frac"]
    for arm in arms:
        if arm == "dispatch-only":
            continue
        p = paired(stats[arm]["worst_frac"], base, "worst_frac")
        print("  %-22s %s" % (arm, fmt(p)))

    if args.trace_driver >= 0:
        print("\n" + "=" * 88)
        print("STATE TRACE, driver %d, dispatch-first, seed %d" % (args.trace_driver, seeds[0]))
        print("=" * 88)
        env.reset(seed=seeds[0])
        env.reset_observer()
        rng = np.random.default_rng(seeds[0])
        rows: list[tuple] = []
        last: tuple | None = None
        while env.agents:
            agent = env.agent_selection
            action, intended = policies["dispatch-first"](env, agent, rng)
            from diag_common import verify_action
            action = verify_action(env, agent, action, intended)
            zi = env._zone_int_id(agent)
            mask = env.action_mask(agent)
            d = env._last_obs[args.trace_driver]["self"]
            sig = (d["status"], zi, len(env._driver_relocator))
            if last != sig:
                rows.append((sig, "t=%5.1f zone=%2d drv%-3d status=%-11s "
                                 "region=%2d  legal noop/disp/reloc=%d/%d/%d  "
                                 "-> act %d (%s)"
                             % (env.env.time, zi, args.trace_driver, d["status"],
                                int(d.get("current_region", 0)),
                                int(mask[0]),
                                int(mask[1:1 + env._max_pending].sum()),
                                int(mask[1 + env._max_pending:].sum()),
                                action, intended)))
                last = sig
            env.step(action)
            env.observe_turn()
        for _s, line in rows[:80]:
            print("  " + line)
        print("  (%d state-change rows)" % len(rows))
        print("  arrivals into zones for this driver: %s"
              % (env.reloc_arrivals.get(args.trace_driver),))
        print("  completions served by this driver  : %d"
              % sum(1 for c in env.completions if c["did"] == args.trace_driver))
    else:
        print("\n(no --trace-driver given; pass an explicit driver id to trace it)")


def fmt(p: dict) -> str:
    unit = "" if p["label"] == "worst_frac" else ""
    return (f"{100 * p['mean']:+6.2f}pp +/- {100 * p['std']:.2f}pp sd "
            f"(sem {100 * p['sem']:.2f}, n={p['n']}, "
            f"{p['wins']}W/{p['losses']}L/{p['ties']}T){unit}")


if __name__ == "__main__":
    main()