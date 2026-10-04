"""D48 invariant: the relocator share is never paid twice for one ride.

Moving the relocator payment from ride completion to the dispatch of the
relocated car creates two ways to overpay:

  1. paying at dispatch while section 1 still pays at completion, so one ride
     pays 1.45 gamma instead of 1.0;
  2. paying at dispatch for a ride that is later CANCELLED, so a relocated
     cancelled ride pays 0.45 while a plain cancelled ride pays nothing --
     a bias toward relocating into zones where orders time out.

Neither is visible from config arithmetic; both have to be measured. This
asserts the emitted reward is fully accounted for.

Method: alpha, beta and cancel_penalty are zeroed and team_share is 0, so
served credit is the ONLY nonzero term and the sum of every reward the env
emits is exactly the served credit paid. Then

    emitted = completed * gamma + (payouts whose ride never completed) * share * gamma

Anything above that second term is an overpay. Payouts are counted from the
env's own `_order_relocator_paid` inserts rather than from order status, so
reused order ids across episodes cannot corrupt the count -- ids run 0-129
every episode, so id-keyed accounting silently double-counts.

    uv run python scripts/diag_served_credit_total.py --episodes 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC

LIVE = (OrderStatus.ASSIGNED, OrderStatus.ONBOARD)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=2000)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.team_share = 0.0
    cfg.alpha = 0.0
    cfg.beta = 0.0
    cfg.cancel_penalty = 0.0
    env = RideGymAEC(cfg)
    mp = env._max_pending

    gamma = float(cfg.gamma)
    share = float(cfg.relocator_share)

    emitted = 0.0
    payouts = 0
    unpaid_payouts = 0   # payouts whose ride was still running at cutoff
    completed = 0
    inflight_detail: dict[int, int] = {}
    live: set[int] = set()

    orig = env._compute_zone_rewards

    def traced(events, dispatched_this_turn=None):
        nonlocal emitted, payouts
        before = set(env._order_relocator_paid)
        out = orig(events, dispatched_this_turn)
        fresh = set(env._order_relocator_paid) - before
        payouts += len(fresh)
        live.update(fresh)
        emitted += float(np.sum(out))
        return out

    env._compute_zone_rewards = traced

    for ep in range(args.episodes):
        env.reset(seed=args.base_seed + ep)
        live.clear()
        while env.agents:
            a = env.agent_selection
            m = env.action_mask(a)
            d = np.flatnonzero(m[1: 1 + mp])
            r = np.flatnonzero(m[1 + mp:])
            act = int(d[0]) + 1 if len(d) else (
                int(r[0]) + 1 + mp if len(r) else 0)
            env.step(act)
        completed += sum(1 for o in env.env.orders.values()
                         if o.status == OrderStatus.COMPLETED)
        # A payout still standing at cutoff belongs to a ride in progress: it
        # got the relocator share and has not yet earned the dispatcher's.
        # These are rides paid at dispatch that the episode ended before
        # finishing, so they are legitimate, not overpayments.
        live_at_cutoff = [
            oid for oid in live
            if (o := env.env.orders.get(oid)) is not None and o.status in LIVE
        ]
        unpaid_payouts += len(live_at_cutoff)
        inflight_detail[ep] = len(live_at_cutoff)

    # Expected total, computed from what the env actually did on THIS run.
    #
    # A control run with relocator_share = 0 is NOT a valid baseline here:
    # changing the share changes zone rewards, which changes the policy, so the
    # two runs follow different trajectories and complete different numbers of
    # rides. Measured: 8/7/9 relocator payments on the control's trajectories
    # against 6/5/7 still-running here.
    #
    # Instead count directly from this run. Every completed ride pays exactly
    # gamma between its dispatcher and its relocator, and a relocator payout
    # still standing at cutoff has paid share*gamma and nothing more. Both
    # numbers are observed, not inferred from config.
    baseline = float(completed)
    expected = baseline + unpaid_payouts * share * gamma
    over = emitted - expected

    print(f"episodes                     : {args.episodes}")
    print(f"completed rides              : {completed}")
    print(f"relocator payouts at dispatch: {payouts}")
    print(f"  still running at cutoff    : {unpaid_payouts} "
          f"{dict(inflight_detail)}")
    print()
    print(f"baseline (completed x gamma) : {baseline:.2f}")
    print(f"served credit emitted        : {emitted:.2f}")
    print(f"expected                      = {baseline:.2f} + "
          f"{unpaid_payouts} x {share * gamma:.2f} = {expected:.2f}")
    print(f"overpayment                   = {over:+.2f}")
    print()
    ok = over < 0.5
    print(f"INVARIANT no ride is paid twice: {'HOLDS' if ok else 'VIOLATED'}")
    if not ok:
        print(f"  {over/ (share*gamma):.0f} rides were paid roughly twice. Either the")
        print("  completion path re-pays the relocator, or the cancellation clawback")
        print("  is not firing. Both inflate relocation against its deadhead cost.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
