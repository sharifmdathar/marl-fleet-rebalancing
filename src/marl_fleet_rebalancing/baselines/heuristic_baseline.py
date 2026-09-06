"""Rolling-horizon imbalance/LP rebalancer baseline (MIVR-lite).

This baseline follows the RLSCALE-BENCH lesson: a tuned classical baseline
for fair comparison against RL agents. It is NOT the "floor" — it should be
competitive when well-tuned.

Algorithm:
1. Track observed demand rates per region over a sliding window.
2. Forecast demand for the lookahead horizon using recent rates.
3. Compute supply-demand imbalance per region (idle vehicles - predicted demand).
4. Solve a transportation LP: minimize repositioning cost subject to
   balancing imbalances across regions.
5. Execute repositioning actions for idle vehicles using env's relocation_points.

The `relocate` action sends an idle vehicle to a relocation point (region centroid).
Dispatch (order bidding) is handled by a fixed nearest-neighbor rule,
not by this rebalancer — D5 in docs/decisions.md.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linprog

if TYPE_CHECKING:
    from ride_gym import RidePoolEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LP rebalancer core
# ---------------------------------------------------------------------------


@dataclass
class RebalancePlan:
    """Output of the LP rebalancer: how many vehicles to send from each region to each region."""

    transfers: np.ndarray  # shape (n_regions, n_regions)
    cost: float


def solve_rebalance_lp(
    imbalances: np.ndarray,
    costs: np.ndarray,
    n_idle: int,
    max_per_route: int | None = None,
) -> RebalancePlan:
    """Solve a transportation LP to rebalance idle vehicles across regions.

    Args:
        imbalances:     shape (n_regions,). Negative = surplus, positive = deficit.
        costs:         shape (n_regions, n_regions). Cost per vehicle to move i -> j.
        n_idle:        Total number of idle vehicles available.
        max_per_route: Optional cap on vehicles per (i, j) route.

    Returns:
        RebalancePlan with optimal transfers and cost.
    """
    n_regions = len(imbalances)
    surplus = np.maximum(-imbalances, 0.0)   # regions with extra vehicles
    deficit = np.maximum(imbalances, 0.0)    # regions that need vehicles

    # Flatten decision variables: x[i * n_regions + j] = vehicles moved i -> j
    n_vars = n_regions * n_regions

    # Objective: minimize sum(costs[i,j] * x[i,j])
    c = costs.flatten()

    # Outflow from region i <= surplus[i]
    A_ub_out = np.zeros((n_regions, n_vars))
    for i in range(n_regions):
        for j in range(n_regions):
            A_ub_out[i, i * n_regions + j] = 1.0
    b_ub_out = surplus

    # Inflow to region j <= deficit[j]
    A_ub_in = np.zeros((n_regions, n_vars))
    for i in range(n_regions):
        for j in range(n_regions):
            A_ub_in[j, i * n_regions + j] = 1.0
    b_ub_in = deficit

    # Total outflow <= n_idle
    A_ub_total = np.ones((1, n_vars))
    b_ub_total = np.array([n_idle])

    A_ub = np.vstack([A_ub_out, A_ub_in, A_ub_total])
    b_ub = np.concatenate([b_ub_out, b_ub_in, b_ub_total])

    # Bounds: x >= 0, optionally x <= max_per_route
    bounds = [(0, max_per_route or np.inf) for _ in range(n_vars)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        transfers = result.x.reshape((n_regions, n_regions))
        return RebalancePlan(transfers=transfers, cost=result.fun)
    else:
        logger.warning("LP solver failed, falling back to greedy assignment")
        return _greedy_rebalance(imbalances, costs, n_idle)


def _greedy_rebalance(
    imbalances: np.ndarray,
    costs: np.ndarray,
    n_idle: int,
) -> RebalancePlan:
    """Greedy fallback when LP solver fails."""
    n_regions = len(imbalances)
    transfers = np.zeros((n_regions, n_regions))

    surplus_zones = np.where(imbalances < 0)[0]
    deficit_zones = np.where(imbalances > 0)[0]

    surplus_order = surplus_zones[np.argsort(imbalances[surplus_zones])]
    deficit_order = deficit_zones[np.argsort(-imbalances[deficit_zones])]

    remaining_idle = n_idle

    for s_idx in surplus_order:
        if remaining_idle <= 0:
            break
        for d_idx in deficit_order:
            if remaining_idle <= 0:
                break
            available = min(
                int(-imbalances[s_idx]),
                int(imbalances[d_idx]),
                remaining_idle,
            )
            if available <= 0:
                continue
            transfers[s_idx, d_idx] = available
            remaining_idle -= available

    return RebalancePlan(transfers=transfers, cost=float(np.sum(transfers * costs)))


# ---------------------------------------------------------------------------
# Main baseline class
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """Tunable knobs for the heuristic baseline.

    Default values are reasonable starting points; tune on base scenario B
    with comparable effort to RL training (D3 / RLSCALE-BENCH lesson).
    """

    lookback_minutes: float = 10.0      # sliding window for rate estimation
    lookahead_minutes: float = 5.0     # horizon for demand prediction
    rebalance_frequency_minutes: float = 1.0
    max_rebalance_per_step: int = 5
    min_idle_to_rebalance: int = 3
    demand_model: str = "exponential_smoothing"
    smoothing_alpha: float = 0.3
    max_per_route: int | None = None


class HeuristicBaseline:
    """Rolling-horizon LP rebalancer.

    Uses the RideGym environment's built-in region system (from
    relocation_points) for demand forecasting and rebalancing.
    """

    def __init__(
        self,
        config: BaselineConfig | None = None,
    ) -> None:
        self.config = config or BaselineConfig()
        self._n_regions: int | None = None
        self._region_centroids: list[tuple[float, float]] | None = None

        # Internal state (reset on env.reset)
        self._last_rebalance_time: float = 0.0
        self._current_plan: RebalancePlan | None = None
        self._idle_vehicle_regions: dict[int, int] = {}  # vid -> region
        self._demand_rate: np.ndarray | None = None
        self._network = None

    def reset(self, seed: int | None = None) -> None:
        """Reset internal state for a new episode."""
        self._last_rebalance_time = 0.0
        self._current_plan = None
        self._idle_vehicle_regions.clear()
        if self._n_regions is not None:
            self._demand_rate = np.ones(self._n_regions) * 10.0  # default 10 veh/hr

    def actions(self, env: "RidePoolEnv", obs: dict[str, Any]) -> dict[str, Any]:
        """Produce per-vehicle actions for the current step.

        Args:
            env:  RidePoolEnv instance
            obs:  Current observation dict keyed by vehicle ID (int)

        Returns:
            actions: {vid: {"orders": [...]}} or {vid: {"relocate": (lon, lat)})
        """
        current_time = env.time

        # Initialize region info from first observation
        self._init_from_env(env, obs)

        # 1. Update demand rate estimates
        self._update_demand_rates(env, obs, current_time)

        # 2. Recompute plan if needed
        time_since_last = current_time - self._last_rebalance_time
        if (
            self._current_plan is None
            or time_since_last >= self.config.rebalance_frequency_minutes
        ):
            self._compute_rebalance_plan(env, obs)
            self._last_rebalance_time = current_time

        # 3. Execute dispatch (nearest-neighbor) + rebalancing
        return self._execute_actions(env, obs)

    def _init_from_env(self, env: "RidePoolEnv", obs: dict) -> None:
        """Initialize region count and centroids from env."""
        if self._n_regions is None:
            # Get relocation points from first vehicle's observation
            sample_obs = next(iter(obs.values())) if obs else {}
            relocation_points = sample_obs.get("relocation_points", [])
            self._n_regions = len(relocation_points)
            self._region_centroids = [
                (float(lon), float(lat)) for lon, lat in relocation_points
            ]
            self._demand_rate = np.ones(self._n_regions) * 10.0
            logger.info(
                f"HeuristicBaseline initialized with {self._n_regions} regions"
            )

    def _update_demand_rates(
        self,
        env: "RidePoolEnv",
        obs: dict,
        current_time: float,
    ) -> None:
        """Update per-region demand rate estimates using recent orders."""
        lookback = self.config.lookback_minutes
        n_regions = self._n_regions or 0

        # Count orders per region in lookback window
        region_counts = np.zeros(n_regions)
        for order in env.orders.values():
            if order.request_time >= current_time - lookback:
                # Map order origin (lon, lat) to nearest region
                region_id = self._find_nearest_region(order.origin)
                if 0 <= region_id < n_regions:
                    region_counts[region_id] += 1

        # Convert to rate (vehicles/hour)
        observed_rate = (region_counts / lookback) * 60.0 if lookback > 0 else region_counts

        # Update EMA
        alpha = self.config.smoothing_alpha
        if self._demand_rate is None:
            self._demand_rate = observed_rate
        else:
            self._demand_rate = alpha * observed_rate + (1 - alpha) * self._demand_rate

    def _find_nearest_region(self, coord: tuple[float, float]) -> int:
        """Find nearest region centroid to a coordinate (lon, lat)."""
        if self._region_centroids is None:
            return 0
        lon, lat = coord
        best_idx = 0
        best_dist = float("inf")
        for i, (c_lon, c_lat) in enumerate(self._region_centroids):
            dist = (lon - c_lon) ** 2 + (lat - c_lat) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _compute_rebalance_plan(
        self,
        env: "RidePoolEnv",
        obs: dict[str, Any],
    ) -> None:
        """Compute optimal rebalancing plan using LP."""
        if self._n_regions is None or self._demand_rate is None:
            return

        n_regions = self._n_regions

        # 1. Forecast demand for lookahead horizon
        forecast_horizon = self.config.lookahead_minutes / 60.0
        predicted_demand = self._demand_rate * forecast_horizon

        # 2. Count current supply (idle vehicles) per region
        supply = np.zeros(n_regions)
        idle_vehicles: dict[int, int] = {}  # vid -> region

        for vid, vstate in obs.items():
            region_id = vstate.get("self", {}).get("current_region", 0)
            if 0 <= region_id < n_regions:
                supply[region_id] += 1
                if vstate.get("self", {}).get("status") == "idle":
                    idle_vehicles[vid] = region_id

        self._idle_vehicle_regions = idle_vehicles

        # 3. Compute imbalance: positive = deficit, negative = surplus
        imbalance = predicted_demand - supply

        n_idle = len(idle_vehicles)
        if n_idle < self.config.min_idle_to_rebalance:
            self._current_plan = None
            return

        # 4. Compute inter-region costs (Euclidean)
        costs = self._compute_region_costs()

        # 5. Solve LP
        self._current_plan = solve_rebalance_lp(
            imbalances=imbalance,
            costs=costs,
            n_idle=n_idle,
            max_per_route=self.config.max_per_route,
        )

    def _compute_region_costs(self) -> np.ndarray:
        """Compute cost matrix using Euclidean distance between region centroids."""
        if self._region_centroids is None:
            return np.zeros((1, 1))

        n_regions = len(self._region_centroids)
        costs = np.zeros((n_regions, n_regions))

        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    lon_i, lat_i = self._region_centroids[i]
                    lon_j, lat_j = self._region_centroids[j]
                    # Approximate: 1 deg lat ~ 111km, 1 deg lon ~ 111km * cos(lat)
                    km_per_deg_lat = 111.0
                    km_per_deg_lon = 111.0 * math.cos(math.radians(lat_i))
                    eucl_km = math.sqrt(
                        ((lat_j - lat_i) * km_per_deg_lat) ** 2
                        + ((lon_j - lon_i) * km_per_deg_lon) ** 2
                    )
                    costs[i, j] = (eucl_km / 30.0) * 60.0  # minutes at 30 km/h

        return costs

    def _execute_actions(
        self,
        env: "RidePoolEnv",
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute nearest-neighbor dispatch + rebalancing from plan."""
        actions: dict[str, Any] = {}
        dispatched_orders: set[int] = set()
        remaining_idle: list[tuple[int, int]] = [  # (vid, region)
            (vid, region) for vid, region in self._idle_vehicle_regions.items()
        ]

        # 1. Dispatch: nearest-neighbor assignment to pending orders
        from ride_gym.enums import OrderStatus
        pending_orders = [
            o for o in env.orders.values()
            if o.status == OrderStatus.PENDING and o.order_id not in dispatched_orders
        ]

        for order in pending_orders:
            best_vid = None
            best_dist = float("inf")
            best_region = -1

            for vid, region in remaining_idle:
                dist = self._region_distance(
                    region, self._find_nearest_region(order.origin)
                )
                if dist < best_dist:
                    best_dist = dist
                    best_vid = vid
                    best_region = region

            if best_vid is not None:
                actions[best_vid] = {"orders": [order.order_id]}
                dispatched_orders.add(order.order_id)
                remaining_idle = [(v, r) for v, r in remaining_idle if v != best_vid]

        # 2. Rebalancing: execute from current plan
        if self._current_plan is not None and remaining_idle:
            transfers = self._current_plan.transfers.copy()
            reloc_count = 0

            for vid, region in remaining_idle:
                if reloc_count >= self.config.max_rebalance_per_step:
                    break

                available = int(transfers[region, :].sum())
                if available <= 0:
                    continue

                # Find best destination
                best_dest = int(np.argmax(transfers[region, :]))
                if best_dest != region and transfers[region, best_dest] > 0:
                    target_lon, target_lat = self._region_centroids[best_dest]
                    actions[vid] = {"relocate": (target_lon, target_lat)}
                    transfers[region, best_dest] -= 1
                    reloc_count += 1

        # 3. Remaining vehicles: no orders (idle)
        all_vids = set(obs.keys())
        for vid in all_vids - set(actions.keys()):
            actions[vid] = {"orders": []}

        return actions

    def _region_distance(self, r1: int, r2: int) -> float:
        """Euclidean distance between region centroids."""
        if self._region_centroids is None:
            return 0.0
        lon1, lat1 = self._region_centroids[r1]
        lon2, lat2 = self._region_centroids[r2]
        return math.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)


# ---------------------------------------------------------------------------
# Convenience wrapper matching random_policy.py interface
# ---------------------------------------------------------------------------


def heuristic_actions(env: "RidePoolEnv", obs: dict[str, Any]) -> dict[str, Any]:
    """Produce heuristic (LP) rebalancing actions with default config."""
    baseline = _global_baseline
    if baseline is None:
        baseline = get_baseline()
    return baseline.actions(env, obs)


_global_baseline: HeuristicBaseline | None = None


def get_baseline(
    config: BaselineConfig | None = None,
) -> HeuristicBaseline:
    """Get (or create) the global baseline instance."""
    global _global_baseline
    if _global_baseline is None:
        _global_baseline = HeuristicBaseline(config)
    return _global_baseline


def reset_baseline() -> None:
    """Reset the global baseline (call between episodes)."""
    global _global_baseline
    if _global_baseline is not None:
        _global_baseline.reset()