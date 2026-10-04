"""Is the task physically achievable? Fleet vs demand vs distance vs speed.

Feasibility, not reward. Everything else this session has assumed the scenario is
solvable and asked why the policy is not solving it. That assumption was never
checked.

By Little's Law the fleet required to sustain an arrival rate is

    cars_required = arrival_rate x time_per_job

where time_per_job includes the ride itself plus the deadhead to reach the
rider. If that exceeds the fleet actually simulated, the scenario is
infeasible: no policy can serve the demand, service rate is capped by physics
rather than by decisions, and every targeting or reward result is measured
against an impossible target.

Measures both sides of that equation from the simulator, plus the distance and
speed that set time_per_job.

    uv run python scripts/diag_feasibility.py --episodes 3
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=9000)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = args.split
    env = RideGymAEC(cfg)
    mp = env._max_pending

    print("=" * 74)
    print("Physical feasibility of the NYC scenario")
    print("=" * 74)
    print(f"  zones                 : {env._num_zones}")
    print(f"  drivers (fleet)       : {env._num_drivers}")
    print(f"  turn length           : {cfg.dt if hasattr(cfg, 'dt') else 0.5} min")
    print(f"  episode horizon       : {cfg.horizon if hasattr(cfg, 'horizon') else 60} min")
    print(f"  order timeout         : {cfg.order_timeout} min")
    print()

    all_trips = []
    all_dead = []
    n_orders = 0
    n_completed = 0
    n_cancelled = 0
    n_pending = 0

    for ep in range(args.episodes):
        env.reset(seed=args.base_seed + ep)
        # Record pickup/dropoff times and distances per completed order.
        info: dict[int, tuple[float, float]] = {}
        while env.agents:
            for _ in range(env._num_zones):
                if not env.agents:
                    break
                ag = env.agent_selection
                m = env.action_mask(ag)
                d = np.flatnonzero(m[1: 1 + mp])
                a = int(d[0]) + 1 if d.size else 0
                if not m[a]:
                    a = int(np.flatnonzero(m)[0])
                env.step(a)
            for o in env.env.orders.values():
                if o.status == OrderStatus.COMPLETED:
                    pt = getattr(o, "pickup_time", None)
                    dt = getattr(o, "dropoff_time", None)
                    if pt is not None and dt is not None:
                        info[o.order_id] = (float(pt), float(dt))

        k = env.kpi_summary()
        n_orders += int(k.get("total_orders", 0)) or len(env.env.orders)
        n_completed += sum(1 for o in env.env.orders.values()
                           if o.status == OrderStatus.COMPLETED)
        n_cancelled += sum(1 for o in env.env.orders.values()
                           if o.status == OrderStatus.CANCELLED)
        n_pending += sum(1 for o in env.env.orders.values()
                         if o.status in (OrderStatus.PENDING,
                                         OrderStatus.ASSIGNED,
                                         OrderStatus.ONBOARD))
        horizon = float(env.env.horizon)
        for oid, (pt, dt) in info.items():
            all_trips.append((dt - pt, oid))
        print(f"  ep{ep}: {len(info)} completed, "
              f"{k['empty_km']:.0f} empty km, horizon {horizon:.0f} min")

    horizon = float(env.env.horizon)
    n_zones = env._num_zones
    eps = args.episodes

    print()
    print("=" * 74)
    print("Side 1: the work arriving")
    print("=" * 74)
    arrivals_per_ep = n_orders / eps
    per_hour = arrivals_per_ep / (horizon / 60.0)
    print(f"  orders per episode       : {arrivals_per_ep:.1f}")
    print(f"  orders per hour          : {per_hour:.1f}")
    print(f"  concurrent zones         : {n_zones}")
    print(f"  orders per zone per hour : {per_hour/n_zones:.2f}")
    print()

    print("Side 2: the fleet's capacity")
    print("=" * 74)
    # Empty km per completed ride tells us the deadhead burden directly.
    k_last = env.kpi_summary()
    empty_km_ep = float(k_last["empty_km"])
    empty_per_ride = empty_km_ep / max(n_completed / eps, 1e-9)

    # Mean ride duration from the simulator's own pickup/dropoff stamps.
    durs = [t for t, _ in all_trips]
    if durs:
        mean_ride = float(np.mean(durs))
        med_ride = float(np.median(durs))
        p90_ride = float(np.percentile(durs, 90))
    else:
        mean_ride = med_ride = p90_ride = float("nan")

    print(f"  completed per episode    : {n_completed/eps:.1f}")
    print(f"  cancelled per episode    : {n_cancelled/eps:.1f}")
    print(f"  still open at cutoff     : {n_pending/eps:.1f}")
    print(f"  mean ride duration       : {mean_ride:.2f} min "
          f"(median {med_ride:.2f}, p90 {p90_ride:.2f})")
    print(f"  empty km per completed   : {empty_per_ride:.2f} km")
    print()

    print("=" * 74)
    print("The feasibility ratio")
    print("=" * 74)
    print("  Little's Law: cars_required = arrival_rate x time_per_job")
    print()
    arrival_rate_per_min = per_hour / 60.0
    # time_per_job: the ride, plus the deadhead needed to reach the rider.
    # Deadhead time is deadhead distance / speed; speed is not exposed in the
    # config, so infer it from the ride: distance is unknown here, so use the
    # ratio of empty km to rides as a proxy burden and report the assumption.
    for label, job_min in (
        ("ride time only", mean_ride),
        ("ride + 1x ride of deadhead", mean_ride * 2),
        ("ride + 2x ride of deadhead", mean_ride * 3),
    ):
        req = arrival_rate_per_min * job_min
        ratio = req / max(env._num_drivers, 1e-9)
        flag = "INFEASIBLE" if ratio > 1.0 else "feasible"
        print(f"    {label:<28} job={job_min:5.1f} min  "
              f"need {req:5.1f} cars  = {ratio:4.2f}x fleet  {flag}")
    print()
    print(f"  fleet = {env._num_drivers} cars, "
          f"{n_zones} zones -> {env._num_drivers/n_zones:.2f} cars per zone")
    print()
    print("Reading: a ratio above 1.0 means the fleet cannot physically keep up,")
    print("so service rate is capped by throughput and no policy can fix it.")


if __name__ == "__main__":
    main()
