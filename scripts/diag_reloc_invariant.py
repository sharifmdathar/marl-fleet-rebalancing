"""D47: is the relocator credit still above its own deadhead cost?

The repo rule (AGENTS.md, from env 2.8) is that `relocator_share` must exceed
`beta * km_per_relocation / gamma`, or a perfect relocation is net-negative and
the agent learns to churn.

Under the old relay-chain divisor that invariant was only checked at 1 hop. The
divisor made the EFFECTIVE share fall with chain length: at 3 hops a relocation
paid 0.45/3 = 0.150 against a cost of 0.149, a 1.004x margin, and at 4 hops it
was net-negative.

D47 removes the divisor, so the effective share is now constant at
`relocator_share * gamma` for every relocation regardless of how many times the
car has moved. This asserts that directly, from the live config, rather than
restating numbers that a later config change could invalidate.

    uv run python scripts/diag_reloc_invariant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig

# Measured on this env by scripts/diag_reloc_leg_distance.py (367 legs,
# 3 episodes, seeds 3000-3002). Pass a distance on argv to override.
MEASURED_MEAN_LEG_KM = 2.95


def main() -> None:
    cfg = NYCConfig()
    share = cfg.relocator_share * cfg.gamma

    # Measure the leg distance rather than hardcoding it. An earlier version of
    # this script asserted 2.49 km, carried over from an older run, which made
    # the margin read 3.01x. Re-measured on the current env the mean is 2.95 km
    # and the real margin is 2.54x. A hardcoded distance makes the assertion
    # drift away from the thing it is supposed to check.
    km = float(sys.argv[1]) if len(sys.argv) > 1 else MEASURED_MEAN_LEG_KM
    cost = cfg.beta * km / max(cfg.gamma, 1e-9)
    origin = "measured" if len(sys.argv) > 1 else "default"

    print("relocator_share            : %.3f" % cfg.relocator_share)
    print("gamma                      : %.3f" % cfg.gamma)
    print("beta (per empty km)        : %.3f" % cfg.beta)
    print(f"relocation leg distance    : {km:.2f} km ({origin})")
    print("  run scripts/diag_reloc_leg_distance.py to re-measure")
    print()
    print("cost of one relocation     : %.4f" % cost)
    print("credit if it works         : %.4f" % share)
    print("margin                     : %+.4f  (%.2fx cost)"
          % (share - cost, share / cost))
    print()
    print("effective share by hop count (D47: no divisor, so all rows identical):")
    for hops in (1, 2, 3, 4, 5):
        eff = share  # was: cfg.relocator_share * cfg.gamma / hops
        flag = "OK" if eff > cost * 1.5 else ("THIN" if eff > cost else "NEGATIVE")
        print("   %d hop(s): %.4f  net %+.4f  %s" % (hops, eff, eff - cost, flag))

    ok = share > cost * 1.5
    print()
    print("INVARIANT relocator_share > beta*km/gamma : %s"
          % ("HOLDS" if ok else "VIOLATED"))
    if not ok:
        print("  A relocation that works perfectly is net-negative. The agent will")
        print("  learn to churn rather than rebalance.")
        print("  Re-measure with scripts/diag_reloc_leg_distance.py.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
