"""Does order_timeout explain the failure? Sweep it before changing anything.

Motive: a relocation leg is ~2.5 km at 30 km/h, i.e. ~5 min of travel, against
the default order_timeout of 2.0 min. If a repositioned car always arrives after
its rider gave up, patience -- not the relocation mechanism -- is the binding
constraint, and the fix is a config field rather than an env change.

The earlier version of this sweep was invalid on two counts: its "relocate"
branch could never dispatch (so 0 served was a property of the policy), and its
greedy baseline was untuned here. This version sweeps order_timeout against the
same corrected policy arms used in diag_relocation_conversion.py, paired on
identical seeds, and reports relocation->completion conversion at each timeout so
the crossover can be located rather than asserted.

Self-check: every chosen action is asserted legal in the same mask and asserted
to decode as the policy declared (dispatch -> dict with "orders", relocate ->
dict with "relocate"). A failure raises rather than reporting a number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diag_common import ObservedEnv, make_policies, paired, run_episode

from marl_fleet_rebalancing.config import NYCConfig

# dispatch-only never relocates, so it is the causal control: any completions it
# gains over dispatch-first are attributable to relocation actions alone.
ARMS = ["dispatch-only", "dispatch-first", "relocate-first-best",
        "relocate-first-rand", "relocate-first-first", "random-legal",
        "relocate-only(BROKEN)"]
METRICS = [
    ("sr", "SR %"),
    ("empty_km", "empty km"),
    ("reloc_cmds", "reloc cmds"),
    ("n_relocate", "reloc acts"),
    ("n_dispatch", "disp acts"),
    ("completions", "completions"),
    ("conv_env", "conv_env"),
    ("conv_final_hop", "conv_final"),
    ("conv_fresh", "conv_fresh"),
    ("fresh_delay_mean", "fresh delay"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--timeouts", default="2,3,4,5,6,8")
    ap.add_argument("--zone-a", type=int, default=8)
    ap.add_argument("--zone-b", type=int, default=13)
    ap.add_argument("--pct", type=float, default=60.0)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--allow-demand-free", action="store_true",
                    help="set require_destination_demand=False. The acceptance "
                         "test for learnable targeting: with demand-free "
                         "destinations legal the menu widens and null turns "
                         "(all options equally good) drop from ~81%% to ~16%%, "
                         "so 'best' can finally differ from 'random'.")
    ap.add_argument("--mode", default="hotspot2", choices=["hotspot2", "base"],
                    help="base = unmodified training demand (S0); "
                         "hotspot2 = two-zone concentration scenario")
    args = ap.parse_args()

    seeds = [args.base_seed + i for i in range(args.episodes)]
    timeouts = [float(x) for x in args.timeouts.split(",")]
    policies = make_policies()

    print("scenario: hotspot2 A=%d B=%d pct=%.0f%% split=0.5  data=%s  fleet=%d"
          % (args.zone_a, args.zone_b, args.pct, args.split, NYCConfig().num_drivers))
    print("seeds: %s  (%d paired episodes per arm)" % (seeds, args.episodes))
    print("self-check: every action asserted legal + asserted to decode as declared")
    print("relocate-only(BROKEN) is the old probe: it cannot dispatch, so its 0 "
          "service is a property of the policy.\n")

    rows: dict[float, dict[str, list[dict]]] = {}
    for tmo in timeouts:
        cfg = NYCConfig()
        cfg.nyc_split = args.split
        if args.mode == "hotspot2":
            cfg.demand_shift_mode = "hotspot2"
            cfg.hotspot_zone_a = args.zone_a
            cfg.hotspot_zone_b = args.zone_b
            cfg.hotspot_pct = args.pct
            cfg.hotspot_split = 0.5
        else:
            cfg.demand_shift_mode = "none"
        cfg.order_timeout = tmo
        if args.allow_demand_free:
            cfg.require_destination_demand = False
        env = ObservedEnv(cfg)
        rows[tmo] = {}
        for arm in ARMS:
            out = []
            for seed in seeds:
                rng = np.random.default_rng(seed * 1_000_003 + 7)
                out.append(run_episode(env, policies[arm], seed, rng))
            rows[tmo][arm] = out
        env.close()

    print("\n" + "=" * 108)
    print("ABSOLUTE LEVEL per arm per timeout (mean over %d paired episodes)" % args.episodes)
    print("=" * 108)
    hdr = "%-8s %-22s" % ("tmo", "arm")
    hdr += "".join("%13s" % lbl for _k, lbl in METRICS)
    print(hdr)
    for tmo in timeouts:
        for arm in ARMS:
            v = rows[tmo][arm]
            print("%-8.1f %-22s" % (tmo, arm) + "".join(
                "%13.2f" % np.nanmean([r[k] for r in v]) for k, _l in METRICS))

    print("\n" + "=" * 108)
    print("PAIRED vs dispatch-first at the SAME timeout (docs/eval-noise.md)")
    print("=" * 108)
    for k, lbl in METRICS:
        print("\n-- %s" % lbl)
        print("   %-22s %s" % ("arm", "mean +/- sd (sem, n, W/L/T)"))
        for tmo in timeouts:
            for arm in ARMS:
                if arm == "dispatch-first":
                    continue
                p = paired([r[k] for r in rows[tmo][arm]],
                           [r[k] for r in rows[tmo]["dispatch-first"]], lbl)
                print("   t=%-4.1f %-22s %+9.2f +/- %7.2f sd (sem %5.2f, n=%d, "
                      "%dW/%dL/%dT)"
                      % (tmo, arm, p["mean"], p["std"], p["sem"], p["n"],
                         p["wins"], p["losses"], p["ties"]))

    print("\n" + "=" * 108)
    print("CONVERSION vs TIMEOUT: does relocation start converting as patience rises?")
    print("=" * 108)
    print("  %-8s %-22s %11s %11s %11s %12s"
          % ("tmo", "arm", "conv_fresh", "fresh/reloc", "conv_final", "SR %"))
    for tmo in timeouts:
        for arm in ARMS:
            v = rows[tmo][arm]
            cf = np.mean([r["conv_fresh"] for r in v])
            rc = max(np.mean([r["reloc_cmds"] for r in v]), 1e-9)
            cfin = np.mean([r["conv_final_hop"] for r in v])
            sr = np.mean([r["sr"] for r in v])
            print("  %-8.1f %-22s %11.1f %11.3f %11.1f %12.1f"
                  % (tmo, arm, cf, cf / rc, cfin, sr))

    print("\nVERDICT INPUT 1 -- paired difference vs dispatch-only (never relocates)")
    print("this isolates the causal contribution of relocation actions")
    for k, lbl in METRICS:
        if k in ("n_dispatch", "dispatches"):
            continue
        print("\n-- %s" % lbl)
        for tmo in timeouts:
            line = "   t=%-4.1f " % tmo
            for arm in ARMS:
                if arm == "dispatch-only":
                    continue
                p = paired([r[k] for r in rows[tmo][arm]],
                           [r[k] for r in rows[tmo]["dispatch-only"]], lbl)
                line += "%s %+7.2f (%d/%dW)  " % (arm[:14], p["mean"],
                                                   p["wins"], p["n"])
            print(line)

    print("\nVERDICT INPUT 2 -- paired conv_fresh slope across the timeout range")
    for arm in ARMS:
        xs, ys = [], []
        for tmo in timeouts:
            ys.append(np.mean([r["conv_fresh"] for r in rows[tmo][arm]]))
            xs.append(tmo)
        slope, intercept = np.polyfit(xs, ys, 1)
        # paired first-to-last difference, which is what a claim should rest on
        d = paired([r["conv_fresh"] for r in rows[timeouts[-1]][arm]],
                   [r["conv_fresh"] for r in rows[timeouts[0]][arm]], "conv_fresh")
        print("  %-22s slope %+7.2f completions per min   "
              "first->last %s" % (arm, slope, fmt(d)))


def fmt(p: dict) -> str:
    return (f"{p['mean']:+.2f} +/- {p['std']:.2f} sd (sem {p['sem']:.2f}, "
            f"n={p['n']}, {p['wins']}W/{p['losses']}L/{p['ties']}T)")


if __name__ == "__main__":
    main()