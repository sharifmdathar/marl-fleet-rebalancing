"""Assert relocation is profitable in EXPECTED value, not just nominally.

`AGENTS.md` records the invariant from env 2.8: a CORRECT relocation must still
net positive, or the agent learns to churn instead of to serve. The check that
was written for it compares `relocator_share` to `beta * km` directly, and that
form is WRONG: the relocator credit is CONDITIONAL -- it fires only when the
relocated car is actually dispatched.

Measured (D63, D62 checkpoints, NYC16 val): the credit lands 56.8% of the time
even for a PERFECT aim. So with the old constants:

    0.45 * 0.568 = 0.2556  against  cost 0.2616   ->  NET NEGATIVE

The literal check (0.45 > 0.2616) passed the whole time while every relocation
was a losing move in expectation, and the policy was correctly ambivalent about
relocating.

So this asserts the conditional form:

    relocator_share * gamma * P(dispatch | relocated)  >  beta * km_total

`km_total` is the whole job the relocating zone is charged for under D59: the
relocation leg plus the pickup deadhead. `P(dispatch)` is measured, not assumed.

    uv run python scripts/diag_reloc_invariant.py
    uv run python scripts/diag_reloc_invariant.py --p-dispatch 0.568 --km 4.36
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig

# Measured, not assumed:
#   km_relocation_leg : mean distance of a relocation leg (D50/D57)
#   km_pickup_deadhead: empty km per completed ride (D57)
#   p_dispatch        : fraction of relocation credits that actually fire,
#                       measured for a PERFECT aim (D63)
KM_LEG = 2.9
KM_PICKUP = 1.46
P_DISPATCH_PERFECT = 0.568


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--km-leg", type=float, default=KM_LEG)
    ap.add_argument("--km-pickup", type=float, default=KM_PICKUP)
    ap.add_argument("--km", type=float, default=None,
                    help="total km charged per relocation (defaults to leg+pickup)")
    ap.add_argument("--p-dispatch", type=float, default=P_DISPATCH_PERFECT,
                    help="P(credit fires | best aim); re-measure with "
                         "scripts/diag_reward_rank_resolution.py")
    args = ap.parse_args()

    cfg = NYCConfig()
    km = args.km if args.km is not None else args.km_leg + args.km_pickup
    cost = cfg.beta * km
    credit = cfg.relocator_share * cfg.gamma
    ev = credit * args.p_dispatch

    print("relocator_share              : %.3f" % cfg.relocator_share)
    print("gamma                        : %.3f" % cfg.gamma)
    print("beta (per empty km)          : %.3f" % cfg.beta)
    print("km charged per relocation    : %.2f  (leg %.2f + pickup %.2f)"
          % (km, args.km_leg, args.km_pickup))
    print("P(credit fires | best aim)   : %.3f  (measured)" % args.p_dispatch)
    print()
    print("cost of one relocation       : %.4f" % cost)
    print("credit if it fires           : %.4f" % credit)
    print("EXPECTED credit             : %.4f  (= credit x %.3f)"
          % (ev, args.p_dispatch))
    print()
    print("  EV(net)                    : %+.4f   (%.2fx cost)"
          % (ev - cost, ev / cost))
    print()
    literal = credit > cost
    print("literal check  share > cost  : %s   <- NOT sufficient; the credit"
          % ("pass" if literal else "fail"))
    print("                              is conditional, so this passes while")
    print("                              every relocation loses money.")
    print()
    ok = ev > cost
    print("EXPECTED-VALUE INVARIANT")
    print("  share * gamma * P(dispatch) > beta * km")
    print("  %.4f > %.4f  ->  %s"
          % (ev, cost, "HOLDS" if ok else "VIOLATED"))
    if not ok:
        print()
        print("  A relocation that works PERFECTLY is net-negative in")
        print("  expectation, so the policy is right to avoid relocating and no")
        print("  amount of training will make it aim well. Required share for")
        print("  EV = 0 at these constants: %.4f"
              % (cost / (cfg.gamma * args.p_dispatch)))
        print("  For a 1.5x expected margin: %.4f"
              % (1.5 * cost / (cfg.gamma * args.p_dispatch)))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()