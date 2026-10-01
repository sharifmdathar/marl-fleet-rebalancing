"""Rolling-horizon LP rebalancer baseline for PettingZoo AEC zone controllers.

Adapts the MIVR-lite LP rebalancer (D3 / RLSCALE-BENCH) to the PettingZoo AEC
zone-controller API (RideGymAEC).

Algorithm:
1. Maintain exponential moving average (EMA) of observed arrival rates per zone.
2. At rebalancing intervals (default 1 min), compute supply-demand imbalance:
     imbalance_z = predicted_demand_z - idle_vehicles_z
3. Solve a transportation LP (scipy.optimize.linprog) to determine optimal
   inter-zone vehicle transfers T[i, j]. The objective trades travel cost
   against *unmet predicted demand*:

       minimise  sum_ij c_ij x_ij  -  lambda * sum_j (sum_i x_ij)
       s.t.      sum_j x_ij <= surplus_i      (outflow capped by local excess)
                 sum_i x_ij <= deficit_j      (inflow capped by local shortfall)
                 sum_ij x_ij <= n_idle
                 x_ii = 0

   where ``lambda = deficit_penalty_scale * mean(off-diagonal c)``. The
   second term is the only reason the LP ever moves a vehicle: with
   ``lambda = 0`` every cost is non-negative and every constraint is an upper
   bound, so the all-zero plan is trivially optimal and this baseline silently
   no-ops (see references/lp-rebalancer-zero-transfers.md). Equivalently, the
   reduced cost ``c_ij - lambda`` is the net value of the transfer, so a route
   is used only when filling the destination's predicted shortfall is worth
   more than the travel time it costs.
4. At each zone agent's AEC turn:
   a. First priority: dispatch nearest pending candidate order within dispatch radius
      (lowest valid action index in [1, max_pending]).
   b. Second priority: if idle vehicles remain and T[i, j] > 0 for a valid
      neighboring zone j (respecting action_mask), issue relocation action
      1 + max_pending + j and decrement T[i, j].
   c. Otherwise: return action 0 (no-op).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linprog

if TYPE_CHECKING:
    from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC

logger = logging.getLogger(__name__)


@dataclass
class LPBaselineConfig:
    """Tunable hyperparameters for the LP rebalancer baseline.

    ``speed_kmh`` must stay at the environment's true value: the cost matrix is
    travel *time*, so assuming a faster fleet than the simulator actually runs
    makes the LP under-price relocations and over-move vehicles onto plans that
    arrive late.
    """

    lookback_minutes: float = 5.0  # sliding window for rate estimation
    # Rolling-horizon length for demand prediction. Overshooting is *not* free:
    # predicted demand grows linearly with the horizon while idle supply does
    # not, so a long horizon marks nearly every zone as short and destroys the
    # spatial selectivity that makes the plan useful. Swept on S0/S3; 5 min beat
    # 3/8/12/15/30 on service rate at equal or lower empty km.
    lookahead_minutes: float = 5.0
    rebalance_frequency_minutes: float = 1.0  # LP replanning interval
    smoothing_alpha: float = 0.3  # EMA smoothing factor for demand rates
    demand_init_rate: float = 30.0  # prior demand rate per zone (orders/hour)
    min_idle_to_rebalance: int = 1  # minimum fleet idle count to trigger LP
    speed_kmh: float = 30.0  # assumed average vehicle travel speed
    # Value of filling one vehicle of predicted shortfall, as a multiple of the
    # mean inter-zone travel time (dimensionless, so it ports across networks).
    # 0.0 reproduces the historical no-op behaviour; higher values relocate more
    # aggressively. Re-swept on the 16-zone build (24 configs, 5 episodes, S0
    # and S3 only per AGENTS.md rule 3): the 0.4 default from the 62-zone
    # sweep gave 54.0% service rate, while 0.5 gives 58.6%. Re-tune before
    # quoting this baseline on any other scale.
    deficit_penalty_scale: float = 0.5


def solve_rebalance_lp(
    imbalances: np.ndarray,
    costs: np.ndarray,
    n_idle: int,
    max_per_route: int | None = None,
    deficit_penalty: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Solve the transportation LP to balance idle vehicles across zones.

    Args:
        imbalances: shape (n_zones,), positive = deficit, negative = surplus.
        costs: shape (n_zones, n_zones), cost per vehicle moved i -> j.
        n_idle: total number of idle vehicles across the fleet.
        max_per_route: optional cap on transfers along a single route.
        deficit_penalty: value of delivering one vehicle of predicted shortfall
            (same units as ``costs``). 0.0 makes the all-zero plan optimal, so
            the solver will never relocate; the caller must pass a positive
            value for proactive rebalancing to happen at all.
        rng: optional generator used to round fractional route flows. Pass a
            seeded generator for reproducible plans.

    Returns:
        Integer transfer matrix of shape (n_zones, n_zones).
    """
    n_zones = len(imbalances)
    if deficit_penalty <= 0.0:
        # Nothing to gain from moving: all costs are non-negative and every
        # constraint is an upper bound, so the zero plan is exactly optimal.
        # Skip the solver entirely (it would return zeros anyway).
        return np.zeros((n_zones, n_zones), dtype=int)

    surplus = np.maximum(-imbalances, 0.0)
    deficit = np.maximum(imbalances, 0.0)

    n_vars = n_zones * n_zones
    # Net cost of a transfer = travel time - value of the shortfall it fills.
    c = costs.flatten() - float(deficit_penalty)

    # Outflow from region i <= surplus[i]
    A_ub_out = np.zeros((n_zones, n_vars))
    for i in range(n_zones):
        for j in range(n_zones):
            A_ub_out[i, i * n_zones + j] = 1.0
    b_ub_out = surplus

    # Inflow to region j <= deficit[j]
    A_ub_in = np.zeros((n_zones, n_vars))
    for i in range(n_zones):
        for j in range(n_zones):
            A_ub_in[j, i * n_zones + j] = 1.0
    b_ub_in = deficit

    # Total transfers <= n_idle
    A_ub_total = np.ones((1, n_vars))
    b_ub_total = np.array([n_idle])

    A_ub = np.vstack([A_ub_out, A_ub_in, A_ub_total])
    b_ub = np.concatenate([b_ub_out, b_ub_in, b_ub_total])

    # Self-transfers are never legitimate: pin their variables to zero.
    route_cap = max_per_route if max_per_route else np.inf
    bounds = [
        (0.0, 0.0) if i == j else (0.0, route_cap)
        for i in range(n_zones)
        for j in range(n_zones)
    ]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success and result.x is not None:
        raw_transfers = result.x.reshape((n_zones, n_zones))
        # Discretize: integer part + randomized rounding of fractional remainders
        floor_t = np.floor(raw_transfers).astype(int)
        frac_t = raw_transfers - floor_t
        generator = rng if rng is not None else np.random.default_rng()
        rnd = (generator.random(raw_transfers.shape) < frac_t).astype(int)
        transfers = floor_t + rnd
        # Disallow self-transfers
        np.fill_diagonal(transfers, 0)
        return transfers
    else:
        return _greedy_rebalance_fallback(imbalances, n_idle, deficit_penalty)


def _greedy_rebalance_fallback(
    imbalances: np.ndarray, n_idle: int, deficit_penalty: float = 0.0
) -> np.ndarray:
    """Greedy surplus-to-deficit pairing fallback when LP solver fails."""
    n_zones = len(imbalances)
    transfers = np.zeros((n_zones, n_zones), dtype=int)

    if deficit_penalty <= 0.0:
        # Same reasoning as the LP path: with nothing to gain, don't move.
        return transfers

    surplus_zones = np.where(imbalances < 0)[0]
    deficit_zones = np.where(imbalances > 0)[0]

    surplus_order = surplus_zones[np.argsort(imbalances[surplus_zones])]
    deficit_order = deficit_zones[np.argsort(-imbalances[deficit_zones])]

    remaining_idle = n_idle
    for s_idx in surplus_order:
        if remaining_idle <= 0:
            break
        for d_idx in deficit_order:
            if remaining_idle <= 0 or s_idx == d_idx:
                continue
            available = min(
                int(-imbalances[s_idx]),
                int(imbalances[d_idx]),
                remaining_idle,
            )
            if available <= 0:
                continue
            transfers[s_idx, d_idx] = available
            remaining_idle -= available
            imbalances[s_idx] += available
            imbalances[d_idx] -= available

    return transfers


class ZoneLPRebalancer:
    """Rolling-horizon LP rebalancer policy for RideGymAEC."""

    def __init__(self, config: LPBaselineConfig | None = None) -> None:
        self.config = config or LPBaselineConfig()
        self._n_zones: int | None = None
        self._costs: np.ndarray | None = None
        self._mean_cost: float = 0.0
        self._demand_rates: np.ndarray | None = None
        self._last_rebalance_time: float = -1.0
        self._planned_transfers: np.ndarray | None = None
        # Seeded in reset() so LP rounding is reproducible across runs.
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state at episode boundary."""
        self._last_rebalance_time = -1.0
        self._planned_transfers = None
        self._rng = np.random.default_rng(seed)
        if self._n_zones is not None:
            self._demand_rates = np.ones(self._n_zones) * self.config.demand_init_rate

    def _initialize_from_env(self, env: "RideGymAEC") -> None:
        """Initialize zone centroids, cost matrix, and demand estimates."""
        if self._n_zones is not None and self._costs is not None:
            return

        self._n_zones = env._num_zones
        self._demand_rates = np.ones(self._n_zones) * self.config.demand_init_rate

        # Compute pairwise centroid distance costs
        if hasattr(env, "relocation_points"):
            cents = np.asarray(env.relocation_points, dtype=float)
        else:
            cents = np.asarray(getattr(env.env, "relocation_points", []), dtype=float)
        costs = np.zeros((self._n_zones, self._n_zones), dtype=np.float32)

        for i in range(self._n_zones):
            for j in range(self._n_zones):
                if i != j:
                    lon_i, lat_i = cents[i]
                    lon_j, lat_j = cents[j]
                    km_per_deg_lat = 111.0
                    km_per_deg_lon = 111.0 * math.cos(math.radians((lat_i + lat_j) / 2.0))
                    dist_km = math.sqrt(
                        ((lat_j - lat_i) * km_per_deg_lat) ** 2
                        + ((lon_j - lon_i) * km_per_deg_lon) ** 2
                    )
                    # Cost in travel minutes
                    costs[i, j] = (dist_km / max(self.config.speed_kmh, 1.0)) * 60.0

        self._costs = costs
        off_diagonal = costs[~np.eye(self._n_zones, dtype=bool)]
        self._mean_cost = float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
        self._planned_transfers = np.zeros((self._n_zones, self._n_zones), dtype=int)

    def _deficit_penalty(self) -> float:
        """Value of filling one vehicle of predicted shortfall, in travel minutes."""
        return float(self.config.deficit_penalty_scale) * self._mean_cost

    def _update_demand_rates(self, env: "RideGymAEC") -> None:
        """Update EMA of zone demand rates based on active pending orders.

        ``env._pending_per_zone`` is pending orders per zone *divided by the
        fleet size* (D22 normalization), so recovering a count requires scaling
        by ``num_drivers``. Scaling by the total pending count mixed units and
        shrank every zone's estimate by ``total_pending / num_drivers``.
        """
        if self._n_zones is None or self._demand_rates is None:
            return

        num_drivers = max(int(getattr(env, "_num_drivers", 1) or 1), 1)
        pending_fraction = np.asarray(
            getattr(env, "_pending_per_zone", np.zeros(self._n_zones)), dtype=float,
        )[: self._n_zones]
        if pending_fraction.size != self._n_zones:
            return
        zone_pending_counts = pending_fraction * num_drivers

        # An order that times out after `order_timeout` minutes turns over about
        # once per window, so `count` pending orders imply count / timeout orders
        # per minute (i.e. count * 60 / timeout per hour).
        timeout_min = float(
            getattr(getattr(env, "_cfg", None), "order_timeout", 2.0) or 2.0
        )
        observed_rate = zone_pending_counts * (60.0 / max(timeout_min, 1e-6))

        alpha = self.config.smoothing_alpha
        self._demand_rates = alpha * observed_rate + (1.0 - alpha) * self._demand_rates

    def _replan_transfers(self, env: "RideGymAEC") -> None:
        """Solve LP and update transfer plan."""
        if self._n_zones is None or self._costs is None or self._demand_rates is None:
            return

        horizon_hours = self.config.lookahead_minutes / 60.0
        predicted_demand = self._demand_rates * horizon_hours
        idle_supply = getattr(env, "_zone_idle_counts", np.zeros(self._n_zones, dtype=int)).astype(float)

        imbalances = predicted_demand - idle_supply
        n_idle_total = int(np.sum(idle_supply))

        if n_idle_total < self.config.min_idle_to_rebalance:
            self._planned_transfers = np.zeros((self._n_zones, self._n_zones), dtype=int)
            return

        self._planned_transfers = solve_rebalance_lp(
            imbalances=imbalances,
            costs=self._costs,
            n_idle=n_idle_total,
            deficit_penalty=self._deficit_penalty(),
            rng=self._rng,
        )

    def select_action(
        self,
        agent: str,
        env: "RideGymAEC",
        obs: dict[str, Any] | None = None,
    ) -> int:
        """Select action for acting zone controller.

        1. Dispatch nearest candidate order if valid.
        2. Relocate idle vehicle according to LP transfer plan if valid.
        3. No-op (0).
        """
        self._initialize_from_env(env)
        current_time = float(env.env.time)

        # Check if rebalance plan needs refresh
        if (
            self._last_rebalance_time < 0
            or (current_time - self._last_rebalance_time) >= self.config.rebalance_frequency_minutes
        ):
            self._update_demand_rates(env)
            self._replan_transfers(env)
            self._last_rebalance_time = current_time

        mask = env.action_mask(agent)
        zone_idx = env._zone_int_id(agent)
        max_pending = env._max_pending

        # Priority 1: Local order dispatch
        dispatch_mask = mask[1 : 1 + max_pending]
        valid_dispatches = np.where(dispatch_mask)[0]
        if len(valid_dispatches) > 0:
            return int(1 + valid_dispatches[0])

        # Priority 2: Rebalancing according to LP plan
        if (
            self._planned_transfers is not None
            and self._n_zones is not None
            and 0 <= zone_idx < self._n_zones
        ):
            targets = np.where(self._planned_transfers[zone_idx] > 0)[0]
            # Filter targets that are valid according to action mask
            valid_targets = [
                int(t)
                for t in targets
                if mask[1 + max_pending + t]
            ]
            if len(valid_targets) > 0:
                # Pick target with highest planned transfer
                best_target = max(
                    valid_targets,
                    key=lambda t: self._planned_transfers[zone_idx, t],
                )
                self._planned_transfers[zone_idx, best_target] -= 1
                return int(1 + max_pending + best_target)

        return 0
