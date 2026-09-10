"""PettingZoo AEC wrapper around RideGym at ZONE-CONTROLLER granularity (D10).

One agent per relocation zone (taxi zone for NYC, grid cell for synthetic
maps). Each zone controller observes zone-level state and issues one action
per turn:

    0                              -> no zone action
    1 .. max_pending               -> dispatch pending order (i-1) to an idle
                                      driver currently in this zone
    max_pending+1 .. +num_zones    -> relocate one idle driver of this zone
                                      to zone (action - max_pending - 1)

At turn flush the wrapper resolves each zone action to a concrete RideGym
driver: per-zone FIFO queues of idle drivers (rebuilt from the last RideGym
observation), consumed in the agents' shuffled first-come-first-served order
(seeded RNG). Drivers never selected issue no action and keep their current
state. This keeps the D9 joint dispatch+rebalancing action space while the
decision granularity is zones, per D10.

All numeric constants come from EnvConfig — no magic numbers here.
Design decisions are documented in docs/pettingzoo-design-decisions.md and
docs/decisions.md (D9, D10, D13).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import AECEnv

from ride_gym import RidePoolEnv
from ride_gym.order_generator import (
    DataFrameOrderGenerator,
    MultiWindowNYCOrderGenerator,
    OSMnxOrderGenerator,
)
from ride_gym.osmnx_network import OSMnxNetwork

from marl_fleet_rebalancing.config import EnvConfig

_M_PER_KM = 1000.0


def _normalize_xy(
    x: float, y: float, area_min: tuple[float, float], area_range: tuple[float, float],
) -> tuple[float, float]:
    return (
        (x - area_min[0]) / max(area_range[0], 1e-9),
        (y - area_min[1]) / max(area_range[1], 1e-9),
    )


class RideGymAEC(AECEnv):
    """PettingZoo AEC environment wrapping a RidePoolEnv with zone-controller agents.

    Parameters
    ----------
    config:
        Experiment configuration (fleet size, horizon, reward weights, order
        source, etc.).
    seed:
        RNG seed for the agent-order shuffle. Also forwarded to the underlying
        RideGym env on reset.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = EnvConfig()

        self._cfg = config
        self._num_drivers = config.num_drivers

        network = OSMnxNetwork(
            graph_path=config.network_path, speed_kmh=config.speed_kmh,
        )
        order_gen = self._make_order_generator(config, network)
        # The region model (one zone per agent, D10) must live in the network's
        # real coordinate space: taxi-zone centroids for NYC, a grid over the
        # network bounds for synthetic maps (RideGym's default grid is built
        # over the abstract area (0,0,100,100), whose targets would snap to
        # arbitrary graph nodes). The env's service area is the network bounds
        # so relocation-target validation matches the same space.
        net_bounds = network.bounds
        self.env = RidePoolEnv(
            area=net_bounds,
            num_drivers=config.num_drivers,
            driver_capacity=config.driver_capacity,
            dt=config.dt,
            horizon=config.horizon,
            order_timeout=config.order_timeout,
            order_generator=order_gen,
            road_network=network,
            relocation_centroids=self._region_centroids(config, network),
        )

        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._geo_min: tuple[float, float] = (net_bounds[0], net_bounds[1])
        self._geo_range: tuple[float, float] = (
            net_bounds[2] - net_bounds[0],
            net_bounds[3] - net_bounds[1],
        )

        # Dynamic from RideGym — adapts to any map's zone/centroid count
        self._num_zones = len(self.env.relocation_points)
        # Zone-controller agents (D10) — one agent per zone
        self.possible_agents = [f"zone_{i}" for i in range(self._num_zones)]

        # From config — should be measured from actual data for each map
        self._max_pending = config.max_pending

        # Action space: idle (1) + dispatch to pending order (max_pending) + relocate to zone (num_zones)
        self._action_dim = 1 + self._max_pending + self._num_zones

        self.observation_spaces: dict[str, Dict] = {
            a: self._make_obs_space() for a in self.possible_agents
        }
        self.action_spaces: dict[str, Discrete] = {
            a: Discrete(self._action_dim) for a in self.possible_agents
        }

        self._last_obs: dict[int, dict] | None = None
        self._current_pending: list[dict] = []
        self._agent_order = np.arange(self._num_zones)
        self._actions_this_turn: dict[str, int] = {}
        self._turn_complete = False
        self._terminated = False
        self._terminate_idx = 0

        # Reward tracking (for computing the custom reward)
        self._prev_served: int = 0
        self._episode_empty_m: float = 0.0

        # Per-turn agent-invariant caches, rebuilt after every env step
        self._pending_features = np.zeros(
            (self._max_pending, 6), dtype=np.float32,
        )
        self._pending_mask = np.zeros(self._max_pending, dtype=np.float32)
        self._pending_per_zone = np.zeros(self._num_zones, dtype=np.float32)
        self._zone_fleet_state = np.zeros(
            (self._num_zones, 2), dtype=np.float32,
        )
        self._zone_centroids_norm = self._normalized_centroids()

        # Per-zone candidate orders (Point B: sorted closest-first per zone)
        self._zone_candidate_orders: list[list[dict]] = [[] for _ in range(self._num_zones)]
        self._zone_pending_features = np.zeros(
            (self._num_zones, self._max_pending, 6), dtype=np.float32,
        )
        self._zone_pending_mask = np.zeros(
            (self._num_zones, self._max_pending), dtype=np.float32,
        )

        # Local credit assignment mappings (Point A):
        # order_id -> zone_idx (which zone controller dispatched this order)
        self._order_dispatcher: dict[int, int] = {}
        # driver_id -> zone_idx (which zone controller commanded this driver)
        self._driver_commander: dict[int, int] = {}

    # ------------------------------------------------------------------ setup

    def _make_order_generator(self, config: EnvConfig, network: OSMnxNetwork):
        """Build the order source selected by ``config.order_source``."""
        if config.order_source == "synthetic":
            return OSMnxOrderGenerator(
                network=network,
                horizon=config.horizon,
                num_orders=config.num_orders,
                arrival="poisson",
                max_party_size=config.max_party_size,
            )
        if config.order_source == "nyc":
            paths = self._nyc_split_paths(config)
            return MultiWindowNYCOrderGenerator(
                network=network,
                order_paths=paths,
                horizon=config.horizon,
                mode=config.nyc_split,
                limit=config.nyc_window_limit,
                random_party_size=config.nyc_random_party_size,
                max_party_size=config.nyc_max_party_size,
                perturb_od_radius_km=config.nyc_perturb_od_radius_km,
            )
        if config.order_source == "dataframe":
            if config.dataframe is None:
                raise ValueError(
                    "order_source='dataframe' requires config.dataframe "
                    "(pandas DataFrame with origin_x/origin_y/dest_x/dest_y/"
                    "request_time columns; endpoints must lie on the network)."
                )
            return DataFrameOrderGenerator(
                dataframe=config.dataframe,
                columns=config.dataframe_columns,
            )
        raise ValueError(f"Unknown order_source: {config.order_source!r}")

    @staticmethod
    def _nyc_split_paths(config: EnvConfig) -> list[str]:
        """Resolve window files for the configured split from the manifest."""
        manifest_path = Path(config.nyc_splits_dir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        entries = manifest["splits"][config.nyc_split]
        base = Path(config.nyc_splits_dir)
        paths = [str(base / e["file"]) for e in entries]
        if not paths:
            raise ValueError(
                f"Split {config.nyc_split!r} has no windows in {manifest_path}."
            )
        return paths

    @staticmethod
    def _region_centroids(config: EnvConfig, network: OSMnxNetwork) -> list[tuple[float, float]]:
        """Region centres (one per zone-controller agent) in network coordinates.

        - ``order_source="nyc"``: taxi-zone centroids from
          ``config.nyc_zone_centroids_path`` that fall inside the network
          bounds, sorted by LocationID so zone indices are deterministic.
        - otherwise: a ``config.relocation_grid`` grid of cell centres over the
          network bounds (RideGym's default grid geometry, but in the real
          coordinate space).
        """
        bounds = network.bounds
        if config.order_source == "nyc":
            df = pd.read_csv(config.nyc_zone_centroids_path)
            inside = df[
                (df["lon"] >= bounds[0]) & (df["lon"] <= bounds[2])
                & (df["lat"] >= bounds[1]) & (df["lat"] <= bounds[3])
            ].sort_values("LocationID")
            if inside.empty:
                raise ValueError(
                    f"No zone centroids from {config.nyc_zone_centroids_path} "
                    f"fall inside the network bounds {bounds}."
                )
            return list(zip(inside["lon"].astype(float), inside["lat"].astype(float)))

        rows, cols = config.relocation_grid
        xmin, ymin, xmax, ymax = bounds
        xs = (np.arange(cols) + 0.5) / cols * (xmax - xmin) + xmin
        ys = (np.arange(rows) + 0.5) / rows * (ymax - ymin) + ymin
        return [(float(x), float(y)) for y in ys for x in xs]

    def _normalized_centroids(self) -> np.ndarray:
        pts = np.zeros((self._num_zones, 2), dtype=np.float32)
        for i, (rx, ry) in enumerate(self.env.relocation_points):
            nx, ny = _normalize_xy(rx, ry, self._geo_min, self._geo_range)
            pts[i] = (nx, ny)
        return pts

    def _make_obs_space(self) -> Dict:
        z = self._num_zones
        p = self._max_pending
        # Zone-controller observation (D10/D13): own-zone state + all-zone
        # fleet state + shared pending orders + clock. Zone identity comes from
        # zone_location so policies can share parameters across agents.
        return Dict({
            "zone_location": Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            "zone_state": Box(0.0, 1.0, shape=(3,), dtype=np.float32),
            "all_zone_state": Box(0.0, 1.0, shape=(z, 2), dtype=np.float32),
            "pending_orders": Box(0.0, 1.0, shape=(p, 6), dtype=np.float32),
            "pending_mask": Box(0.0, 1.0, shape=(p,), dtype=np.float32),
            "pending_passengers": Box(0.0, 1.0, shape=(p,), dtype=np.float32),
            "time_frac": Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "relocation_points": Box(0.0, 1.0, shape=(z, 2), dtype=np.float32),
        })

    def observation_space(self, agent: str) -> Dict:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Discrete:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ state

    def _extract_pending(self, obs: dict[int, dict]) -> list[dict]:
        any_did = next(iter(obs))
        return list(obs[any_did]["pending_orders"])

    def _rebuild_caches(self) -> None:
        """Recompute agent-invariant per-turn state from the last RideGym obs.

        - Pending-order features (sorted most-urgent-first, capped to
          max_pending) and per-zone counts of pending orders by origin.
        - Per-zone fleet state: fraction of the fleet in the zone and the idle
          fraction within the zone (idle drivers by RideGym status).
        """
        cfg = self._cfg
        assert self._last_obs is not None

        # --- Pending orders: most urgent (longest waiting) first ------------
        sorted_pending = sorted(
            self._current_pending,
            key=lambda o: o.get("waiting_time", 0.0),
            reverse=True,
        )
        capacity = max(cfg.driver_capacity, 1)
        timeout = max(cfg.order_timeout, 1e-9)
        self._pending_features[:] = 0.0
        self._pending_mask[:] = 0.0
        self._pending_per_zone[:] = 0.0
        origins = np.asarray(
            [o["origin"] for o in sorted_pending], dtype=float,
        ).reshape(-1, 2)
        for i, order in enumerate(sorted_pending[: self._max_pending]):
            ox, oy = _normalize_xy(
                order["origin"][0], order["origin"][1],
                self._geo_min, self._geo_range,
            )
            dx, dy = _normalize_xy(
                order["destination"][0], order["destination"][1],
                self._geo_min, self._geo_range,
            )
            self._pending_features[i] = (
                ox, oy, dx, dy,
                float(order["num_passengers"]) / capacity,
                min(float(order.get("waiting_time", 0.0)) / timeout, 1.0),
            )
            self._pending_mask[i] = 1.0
        if len(origins) > 0:
            cents = np.asarray(self.env.relocation_points, dtype=float)
            nearest = np.argmin(
                ((origins[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2),
                axis=1,
            )
            for zi in nearest:
                self._pending_per_zone[zi] += 1.0
        self._pending_per_zone /= max(self._max_pending, 1)

        # --- Per-zone candidate orders (Point B: sorted closest-first per zone) ---
        cents = np.asarray(self.env.relocation_points, dtype=float)
        self._zone_pending_features[:] = 0.0
        self._zone_pending_mask[:] = 0.0
        self._zone_candidate_orders = [[] for _ in range(self._num_zones)]

        if self._current_pending:
            pending_origins = np.asarray(
                [o["origin"] for o in self._current_pending], dtype=float
            ).reshape(-1, 2)
            for zi in range(self._num_zones):
                zc = cents[zi]
                # Euclidean distance squared from zone centroid zi to order origins
                dists_sq = np.sum((pending_origins - zc) ** 2, axis=1)
                # Sort orders: primary key distance ascending, secondary key wait time descending
                order_indices = list(range(len(self._current_pending)))
                order_indices.sort(
                    key=lambda idx: (
                        dists_sq[idx],
                        -float(self._current_pending[idx].get("waiting_time", 0.0)),
                    )
                )
                candidates = [
                    self._current_pending[idx]
                    for idx in order_indices[: self._max_pending]
                ]
                self._zone_candidate_orders[zi] = candidates

                for slot, order in enumerate(candidates):
                    ox, oy = _normalize_xy(
                        order["origin"][0], order["origin"][1],
                        self._geo_min, self._geo_range,
                    )
                    dx, dy = _normalize_xy(
                        order["destination"][0], order["destination"][1],
                        self._geo_min, self._geo_range,
                    )
                    self._zone_pending_features[zi, slot] = (
                        ox, oy, dx, dy,
                        float(order["num_passengers"]) / capacity,
                        min(float(order.get("waiting_time", 0.0)) / timeout, 1.0),
                    )
                    self._zone_pending_mask[zi, slot] = 1.0

        # --- Per-zone fleet state -------------------------------------------
        zone_totals = np.zeros(self._num_zones, dtype=np.float32)
        zone_idle = np.zeros(self._num_zones, dtype=np.float32)
        for did in range(self._num_drivers):
            d = self._last_obs[did]["self"]
            zi = int(d.get("current_region", 0))
            zi = max(0, min(zi, self._num_zones - 1))
            zone_totals[zi] += 1.0
            if d["status"] == "idle":
                zone_idle[zi] += 1.0
        self._zone_fleet_state[:, 0] = zone_totals / max(self._num_drivers, 1)
        self._zone_fleet_state[:, 1] = zone_idle / max(self._num_drivers, 1)
        self._zone_idle_counts = zone_idle.astype(np.int64)

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        # Gym seeding semantics: an explicit seed (re)fixes the RNG streams;
        # reset(seed=None) lets them ADVANCE, so successive episodes differ
        # (fresh synthetic orders / fresh NYC train windows per episode) while
        # a seeded run stays fully reproducible. Passing the stored seed on
        # every reset would replay identical episodes forever.
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        obs, _info = self.env.reset(seed=seed)
        self._last_obs = obs
        self._current_pending = self._extract_pending(obs)
        self._terminated = False

        # Reset reward tracking and attribution
        self._prev_served = 0
        self._episode_empty_m = 0.0
        self._order_dispatcher.clear()
        self._driver_commander.clear()

        self._rebuild_caches()

        self.agents = self.possible_agents[:]
        self._agent_order = self._rng.permutation(self._num_zones)
        self.agent_selection = self.agents[int(self._agent_order[0])]
        self._actions_this_turn = {}
        self._turn_complete = False

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    # --------------------------------------------------------------- observe

    def _zone_int_id(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        assert self._last_obs is not None
        zone_idx = self._zone_int_id(agent)

        zone_drivers = float(self._zone_fleet_state[zone_idx, 0])
        zone_idle_frac = (
            self._zone_fleet_state[zone_idx, 1] / zone_drivers
            if zone_drivers > 0 else 0.0
        )
        zone_state = np.array(
            [
                zone_drivers,
                zone_idle_frac,
                float(self._pending_per_zone[zone_idx]),
            ],
            dtype=np.float32,
        )

        return {
            "zone_location": self._zone_centroids_norm[zone_idx],
            "zone_state": zone_state,
            "all_zone_state": self._zone_fleet_state,
            "pending_orders": self._zone_pending_features[zone_idx],
            "pending_mask": self._zone_pending_mask[zone_idx],
            "pending_passengers": self._zone_pending_features[zone_idx, :, 4],
            "time_frac": np.array(
                [min(self.env.time / max(self.env.horizon, 1e-9), 1.0)],
                dtype=np.float32,
            ),
            "relocation_points": self._zone_centroids_norm,
        }

    def action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(self._action_dim, dtype=bool)
        mask[0] = True  # no zone action is always valid
        if self._last_obs is None:
            return mask
        zone_idx = self._zone_int_id(agent)
        has_idle = bool(self._zone_idle_counts[zone_idx] > 0)
        if has_idle:
            n_pending = int(self._zone_pending_mask[zone_idx].sum())
            for i in range(min(n_pending, self._max_pending)):
                mask[1 + i] = True
            # Relocate to any OTHER zone (same-zone relocation is a no-op).
            for j in range(self._num_zones):
                if j != zone_idx:
                    mask[1 + self._max_pending + j] = True
        return mask

    # ---------------------------------------------------------------- actions

    def _decode_action(self, action: int, zone_idx: int = 0) -> dict | None:
        if action is None or action == 0:
            return None
        if 1 <= action <= self._max_pending:
            idx = action - 1
            if 0 <= zone_idx < len(self._zone_candidate_orders):
                candidates = self._zone_candidate_orders[zone_idx]
                if idx < len(candidates):
                    return {"orders": [candidates[idx]["order_id"]]}
            elif idx < len(self._current_pending):
                return {"orders": [self._current_pending[idx]["order_id"]]}
            return None
        target_zone = action - self._max_pending - 1
        if 0 <= target_zone < self._num_zones:
            return {"relocate": int(target_zone)}
        return None

    def _advance_selection(self) -> None:
        if not self.agents:
            return
        if not self._turn_complete:
            zone_idx = self._zone_int_id(self.agent_selection)
            pos = int(np.where(self._agent_order == zone_idx)[0][0])
            if pos + 1 < self._num_zones:
                next_zone_idx = int(self._agent_order[pos + 1])
                self.agent_selection = f"zone_{next_zone_idx}"
                return
        self._turn_complete = True
        self._flush_turn()

    def _compute_zone_rewards(self, events: dict[int, dict]) -> np.ndarray:
        """Compute localized credit assignment rewards per zone controller.

        R_z = (1 - team_share) * R_local,z + team_share * R_global_mean

        Local credit assignment:
        1. Served demand (+gamma): credited directly to the zone controller that
           dispatched the completed order (via self._order_dispatcher[order_id]).
           Fallback: nearest zone centroid to order origin.
        2. Empty deadhead distance penalty (-beta): charged directly to the zone
           controller that commanded the driver (via self._driver_commander[driver_id]).
           Fallback: driver's current_region.
        3. Pending waiting time penalty (-alpha): charged to the zone containing
           the pending order's origin (nearest zone centroid).
        """
        from ride_gym.enums import OrderStatus

        cfg = self._cfg
        local_rewards = np.zeros(self._num_zones, dtype=np.float32)
        cents = np.asarray(self.env.relocation_points, dtype=float)

        self._prev_served = sum(
            1 for o in self.env.orders.values()
            if o.status == OrderStatus.COMPLETED
        )

        # 1. Served demand attribution (+gamma)
        for did, event in events.items():
            for oid in event.get("completed_orders", []):
                dispatcher = self._order_dispatcher.get(oid)
                if dispatcher is not None and 0 <= dispatcher < self._num_zones:
                    local_rewards[dispatcher] += cfg.gamma
                else:
                    order = self.env.orders.get(oid)
                    if order is not None:
                        orig = np.asarray(order.origin, dtype=float)
                        zi = int(np.argmin(np.sum((cents - orig) ** 2, axis=1)))
                        local_rewards[zi] += cfg.gamma
                    else:
                        local_rewards += cfg.gamma / self._num_zones

        # 2. Empty deadhead distance penalty (-beta)
        for did, event in events.items():
            if event.get("is_empty_move", False):
                empty_m = float(event.get("distance_moved", 0.0))
                self._episode_empty_m += empty_m
                empty_km = empty_m / _M_PER_KM / max(cfg.empty_distance_norm_km, 1e-9)
                commander = self._driver_commander.get(did)
                if commander is not None and 0 <= commander < self._num_zones:
                    local_rewards[commander] -= cfg.beta * empty_km
                else:
                    if self._last_obs is not None and did in self._last_obs:
                        d = self._last_obs[did]["self"]
                        zr = max(0, min(int(d.get("current_region", 0)), self._num_zones - 1))
                        local_rewards[zr] -= cfg.beta * empty_km
                    else:
                        local_rewards -= (cfg.beta * empty_km) / self._num_zones

        # 3. Pending waiting time penalty (-alpha)
        if self._current_pending:
            zone_wait_times: dict[int, list[float]] = {
                zi: [] for zi in range(self._num_zones)
            }
            for o in self._current_pending:
                orig = np.asarray(o["origin"], dtype=float)
                zi = int(np.argmin(np.sum((cents - orig) ** 2, axis=1)))
                zone_wait_times[zi].append(float(o.get("waiting_time", 0.0)))
            for zi in range(self._num_zones):
                if zone_wait_times[zi]:
                    w_norm = np.mean(zone_wait_times[zi]) / max(cfg.horizon, 1e-9)
                    local_rewards[zi] -= cfg.alpha * w_norm

        # 4. Team cooperation blending (e.g. 80% local + 20% global team)
        team_share = float(getattr(cfg, "team_share", 0.2))
        if team_share > 0.0:
            mean_r = float(np.mean(local_rewards))
            blended_rewards = (1.0 - team_share) * local_rewards + team_share * mean_r
        else:
            blended_rewards = local_rewards

        return blended_rewards

    def _compute_reward(self, events: dict[int, dict]) -> float:
        """Scalar global mean reward (kept for backward compatibility)."""
        return float(np.mean(self._compute_zone_rewards(events)))

    def _flush_turn(self) -> None:
        if self._terminated or self._last_obs is None:
            return

        # Per-zone FIFO queues of idle drivers (driver-id order), consumed by
        # the zone agents in shuffled first-come-first-served order.
        zone_idle_queues: dict[int, deque[int]] = {
            zi: deque() for zi in range(self._num_zones)
        }
        for did in range(self._num_drivers):
            d = self._last_obs[did]["self"]
            if d["status"] == "idle":
                zi = max(0, min(int(d.get("current_region", 0)), self._num_zones - 1))
                zone_idle_queues[zi].append(did)

        ridegym_actions: dict[int, dict | None] = {}
        claimed_orders: set[int] = set()

        for order_pos in self._agent_order:
            agent = self.agents[int(order_pos)]
            zone_idx = self._zone_int_id(agent)
            action = self._actions_this_turn.get(agent, 0)
            decoded = self._decode_action(action, zone_idx)
            if decoded is None:
                continue
            queue = zone_idle_queues[zone_idx]
            if not queue:
                continue  # masked out upstream; defensive no-op
            driver_id = queue.popleft()
            if "orders" in decoded:
                oid = decoded["orders"][0]
                if oid in claimed_orders:
                    # Order claimed by an earlier zone agent this turn.
                    zone_idle_queues[zone_idx].appendleft(driver_id)
                    continue
                claimed_orders.add(oid)
                self._order_dispatcher[oid] = zone_idx
                self._driver_commander[driver_id] = zone_idx
                ridegym_actions[driver_id] = decoded
            elif "relocate" in decoded:
                self._driver_commander[driver_id] = zone_idx
                ridegym_actions[driver_id] = decoded

        obs, _rewards, dones, info = self.env.step(ridegym_actions)
        self._last_obs = obs
        self._current_pending = self._extract_pending(obs)
        self._rebuild_caches()

        # Compute localized credit assignment rewards
        zone_rewards = self._compute_zone_rewards(info.get("events", {}))

        # Standard AEC delivery: each agent reads its attributed reward
        for a in self.agents:
            zi = self._zone_int_id(a)
            self.rewards[a] += float(zone_rewards[zi])
        self._accumulate_rewards()

        done_flag = bool(dones["__all__"])

        if done_flag:
            self._terminated = True
            kpis = self._kpi_counts()
            self.terminations = {a: True for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {
                a: {
                    "time": info.get("time", self.env.time),
                    "served_total": int(kpis["served"]),
                    "cancelled_total": int(kpis["cancelled"]),
                    "avg_wait": kpis["avg_wait"],
                    "empty_km": kpis["empty_km"],
                }
                for a in self.agents
            }
            self.agent_selection = self.agents[0]
        else:
            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {a: {"time": info.get("time", self.env.time)} for a in self.agents}
            self._agent_order = self._rng.permutation(self._num_zones)
            self.agent_selection = self.agents[int(self._agent_order[0])]
            self._actions_this_turn = {}
            self._turn_complete = False

    def _kpi_counts(self) -> dict[str, float]:
        from ride_gym.enums import OrderStatus

        served = sum(
            1 for o in self.env.orders.values()
            if o.status == OrderStatus.COMPLETED
        )
        cancelled = sum(
            1 for o in self.env.orders.values()
            if o.status == OrderStatus.CANCELLED
        )
        wait_times = [
            float(o.pickup_time - o.request_time)
            for o in self.env.orders.values()
            if o.status == OrderStatus.COMPLETED and o.pickup_time is not None
        ]
        avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
        empty_km = float(self._episode_empty_m / _M_PER_KM)
        return {
            "served": float(served),
            "cancelled": float(cancelled),
            "avg_wait": avg_wait,
            "empty_km": empty_km,
        }

    def kpi_summary(self) -> dict[str, float]:
        """Return episode summary KPIs (served, cancelled, avg_wait, empty_km)."""
        return self._kpi_counts()

    # -------------------------------------------------------------------- AEC

    def step(self, action: int | None) -> None:
        if not self.agents:
            return
        if self._terminated:
            done_agent = self.agent_selection
            remaining = [a for a in self.agents if a != done_agent]
            self.rewards.pop(done_agent, None)
            self._cumulative_rewards[done_agent] = 0.0
            self.agents = remaining
            self.agent_selection = remaining[0] if remaining else None
            self.terminations = {a: True for a in remaining}
            self.truncations = {a: False for a in remaining}
            self.infos = {a: {} for a in remaining}
            return
        current = self.agent_selection
        self.rewards[current] = 0.0
        self._cumulative_rewards[current] = 0.0
        self._actions_this_turn[current] = 0 if action is None else int(action)
        self._advance_selection()

    def state(self) -> np.ndarray:
        """Centralized state (for MAPPO's shared critic): all-zone fleet state
        + pending-order features + clock."""
        if self._last_obs is None:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [
                self._zone_fleet_state.flatten(),
                self._pending_features.flatten(),
                self._pending_mask,
                np.array(
                    [min(self.env.time / max(self.env.horizon, 1e-9), 1.0)],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
