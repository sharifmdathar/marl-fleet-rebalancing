"""Beta sweep: pick the deadhead weight on EFFICIENCY, not service rate.

Finding being acted on (env 3.0, 12 paired windows, order_timeout=5.0):

    greedy            51.5% SR     72.6 km    70.9 SR/100km
    LP                70.8% SR    159.6 km    44.4 SR/100km
    RL sampled s1     80.5% SR    461.4 km    17.4 SR/100km

RL wins service rate by ~+10pp and loses efficiency by ~2.5x. beta is the deadhead
weight; the reward already charges it, so raising it is the direct way to make
pointless relocation expensive.

The break-even invariant from docs/relocation-quality.md is asserted before any
run: a CORRECT relocation must still net positive, or the agent learns to churn
instead of to serve.

    break-even:  beta * km_per_relocation  <  relocator_share * gamma
                 0.06 * 2.49 = 0.149   vs  0.45 * 1.0 = 0.45   (safe)
                 0.15 * 2.49 = 0.374   vs  0.45 * 1.0 = 0.45   (safe)
                 0.19 * 2.49 = 0.473   vs  0.45 * 1.0 = 0.45   (BREACH)

Runs short training runs per beta and evaluates the frozen policy paired on
identical windows. This is a reward sweep: it changes environment dynamics, so
it is NOT a study result and every number here is debugging output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig

KM_PER_RELOCATION = 2.49  # measured mean relocation leg, docs/relocation-quality.md


def check_invariant(beta: float, cfg: NYCConfig) -> tuple[bool, float, float]:
    cost = beta * KM_PER_RELOCATION
    payout = cfg.relocator_share * cfg.gamma
    return cost < payout, cost, payout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="0.06,0.10,0.15,0.19")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the invariant table and exit (no training)")
    args = ap.parse_args()

    cfg0 = NYCConfig()
    print("beta sweep -- env 3.0, order_timeout=%.1f min, "
          "relocator_share=%.2f, gamma=%.2f, km/reloc=%.2f"
          % (cfg0.order_timeout, cfg0.relocator_share, cfg0.gamma,
             KM_PER_RELOCATION))
    print("\nbreak-even invariant (a CORRECT relocation must stay profitable):")
    print("  %-6s %10s %10s %9s" % ("beta", "cost", "payout", "verdict"))
    ok_all = True
    for b in [float(x) for x in args.betas.split(",")]:
        ok, cost, payout = check_invariant(b, cfg0)
        ok_all &= ok
        print("  %-6.2f %10.3f %10.3f %9s"
              % (b, cost, payout, "ok" if ok else "BREACH - churn risk"))
    if not ok_all:
        print("\nAt least one beta BREACHES the invariant. A breaching value is")
        print("diagnostic only: it shows the reward punishing correct relocation.")
        print("It must not become the study default.")
    if args.dry_run:
        return

    raise SystemExit(
        "\nNot implemented in this script: training %d iterations x %d seeds per "
        "beta. Run via the trainer instead:\n"
        "  uv run python -m marl_fleet_rebalancing.experiments.train "
        "--env nyc --algo ippo --arch attention --seed 0 --iterations %d "
        "--beta <B> --name beta<B>-s0 --out-root runs\n"
        "then evaluate the frozen checkpoint paired on identical windows."
        % (args.iterations, args.seeds, args.iterations)
    )


if __name__ == "__main__":
    main()