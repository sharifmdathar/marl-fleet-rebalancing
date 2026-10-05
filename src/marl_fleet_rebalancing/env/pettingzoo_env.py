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
from marl_fleet_rebalancing.demand.drifting_hotspot import (
    DriftingHotspotOrderGenerator,
)
from marl_fleet_rebalancing.demand.generators import (
    BurstOrderGenerator,
    ParameterizedZoneOrderGenerator,
    PermutedZoneOrderGenerator,
    UniformSampledNYCOrderGenerator,
)
from marl_fleet_rebalancing.demand.local_hub import LocalHubOrderGenerator
from marl_fleet_rebalancing.demand.sequential_surge import (
    SequentialSurgeOrderGenerator,
)
from marl_fleet_rebalancing.demand.two_zone_hotspot import (
    TwoZoneHotspotOrderGenerator,
)

_M_PER_KM = 1000.0


def _haversine_km(a, b) -> float:
    """Great-circle distance in km between two (lon, lat) pairs.

    D57: order `origin`/`destination` are in DEGREES, the same space as
    `relocation_points`, so a Manhattan trip spans ~0.02-0.05 degrees. Dividing
    that by 1000 as if it were metres yields 0.000 km, which is how the first
    version of the trip-distance log silently reported zero.
    """
    lon1, lat1 = float(a[0]), float(a[1])
    lon2, lat2 = float(b[0]), float(b[1])
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(h)))

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
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        network = OSMnxNetwork(
            graph_path=config.network_path, speed_kmh=config.speed_kmh,
        )
        centroids = self._region_centroids(config, network)
        order_gen = self._make_order_generator(config, network, centroids)
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
            relocation_centroids=centroids,
        )

        self._geo_min: tuple[float, float] = (net_bounds[0], net_bounds[1])
        self._geo_range: tuple[float, float] = (
            net_bounds[2] - net_bounds[0],
            net_bounds[3] - net_bounds[1],
        )

        # Dynamic from RideGym — adapts to any map's zone/centroid count
        self._num_zones = len(self.env.relocation_points)
        # Order generators built above inferred the adjacency graph from the
        # centroids; verify it matches what RideGym derived, so a scenario that
        # relies on adjacency cannot silently use a different graph.
        derived = self._region_neighbours_from_centroids(centroids)
        actual = tuple(
            tuple(int(x) for x in nbrs) for nbrs in self.env.region_neighbours
        )
        if derived != actual:
            raise ValueError(
                "zone adjacency mismatch: generator-side derivation disagrees "
                "with the env's own graph; relocation scenarios would target "
                "the wrong neighbours."
            )
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
        self._zone_total_counts = np.zeros(self._num_zones, dtype=np.int64)
        self._zone_fleet_state = np.zeros(
            (self._num_zones, 4), dtype=np.float32,
        )
        self._hist_len = max(int(getattr(config, "demand_history_len", 4)), 1)
        self._demand_history: deque[np.ndarray] = deque(maxlen=self._hist_len)
        self._demand_hist_arr = np.zeros((self._num_zones, self._hist_len), dtype=np.float32)
        # D37: running max-abs scale per all_zone_state column, kept for audit.
        # Normalisation divides by the scale seen so far in the episode rather
        # than a constant, so no channel is scaled relative to another by a
        # hand-picked number. Exposed via obs_scales() for the scale audit.
        self._obs_scales = np.zeros(4, dtype=np.float64)
        self._demand_hist_scale = 0.0
        # D43: per-zone cars-in-flight and its running scale
        self._zone_inbound_reloc = np.zeros(self._num_zones, dtype=np.float32)
        self._inbound_reloc_scale = 0.0
        self._zone_centroids_norm = self._normalized_centroids()

        # Per-zone candidate orders (Point B: sorted closest-first per zone)
        self._zone_candidate_orders: list[list[dict]] = [[] for _ in range(self._num_zones)]
        self._zone_pending_features = np.zeros(
            (self._num_zones, self._max_pending, 6), dtype=np.float32,
        )
        self._zone_pending_mask = np.zeros(
            (self._num_zones, self._max_pending), dtype=np.float32,
        )

        # Local credit assignment mappings (Point A & D16):
        # order_id -> zone_idx (which zone controller dispatched this order)
        self._order_dispatcher: dict[int, int] = {}
        # Completed order ids already credited this episode, so completion
        # credit is paid exactly once per ride.
        self._seen_completed_ids: set[int] = set()
        # driver_id -> zone_idx (which zone controller commanded this driver)
        self._driver_commander: dict[int, int] = {}
        # driver_id -> zone_idx (which zone initiated the relocation that put
        # this driver where it is). D47: a single int, not a list.
        #
        # This was a list of every zone that had relocated the car since its
        # last completed order, and the relocator credit was divided by its
        # length. Measured, that list is not a relay: over 504 relocations the
        # median gap between two hops of the same car was 20 turns, only 0.4%
        # of hop pairs were back-to-back, and 87% of pairs had the car complete
        # an order in between. It counted how often a car had been reused since
        # it last earned, not how many zones staged a journey, so dividing by
        # it charged zones for coincidence. At 3 hops the payout fell to 0.150
        # against a 0.149 deadhead cost -- a 1.004x margin.
        #
        # Relocations are now independent A->B events. The zone that commanded
        # the most recent move is paid the full relocator_share if that driver
        # later completes an order; no zone is paid for a car it did not move.
        self._driver_relocator: dict[int, int] = {}
        # order_id -> relocator zone already paid at dispatch (D48). Lets
        # section 1 know to reduce the dispatcher's share so the pair pays one
        # gamma between them rather than two.
        self._order_relocator_paid: dict[int, tuple[int, float]] = {}
        # D82: zones that moved a car since it was last dispatched, and how
        # each order's relocator payment was divided among them.
        self._driver_reloc_chain: dict[int, list[int]] = {}
        self._order_relocator_shares: dict[int, list[tuple[int, float]]] = {}
        # Dispatches requested this turn but not yet confirmed (D51). Committed
        # to _order_dispatcher / _driver_commander only after RideGym honours
        # the bid, so a refused dispatch leaves no phantom attribution.
        self._pending_dispatch: list[tuple[int, int, int]] = []
        # driver_id -> target_zone_idx (destination of ongoing relocation)
        self._driver_relocation_target: dict[int, int] = {}

    # ------------------------------------------------------------------ setup

    def _make_order_generator(
        self,
        config: EnvConfig,
        network: OSMnxNetwork,
        centroids: list[tuple[float, float]] | None = None,
    ):
        """Build the order source selected by ``config.order_source``."""
        if config.order_source == "synthetic":
            gen = OSMnxOrderGenerator(
                network=network,
                horizon=config.horizon,
                num_orders=config.num_orders,
                arrival="poisson",
                max_party_size=config.max_party_size,
            )
        elif config.order_source == "nyc":
            paths = self._nyc_split_paths(config)
            gen = UniformSampledNYCOrderGenerator(
                network=network,
                order_paths=paths,
                horizon=config.horizon,
                mode=config.nyc_split,
                limit=config.nyc_window_limit,
                random_party_size=config.nyc_random_party_size,
                max_party_size=config.nyc_max_party_size,
                perturb_od_radius_km=config.nyc_perturb_od_radius_km,
            )
        elif config.order_source == "dataframe":
            if config.dataframe is None:
                raise ValueError(
                    "order_source='dataframe' requires config.dataframe "
                    "(pandas DataFrame with origin_x/origin_y/dest_x/dest_y/"
                    "request_time columns; endpoints must lie on the network)."
                )
            gen = DataFrameOrderGenerator(
                dataframe=config.dataframe,
                columns=config.dataframe_columns,
            )
        else:
            raise ValueError(f"Unknown order_source: {config.order_source!r}")

        # Parameterized zone demand injection (D15)
        if (
            config.target_zone is not None
            and config.target_zone_demand_pct is not None
            and centroids is not None
        ):
            gen = ParameterizedZoneOrderGenerator(
                base_generator=gen,
                network=network,
                relocation_centroids=centroids,
                target_zone=config.target_zone,
                target_zone_pct=config.target_zone_demand_pct,
                seed=self._seed,
            )

        # Demand-shift mechanisms (D21; eval-side, one per scenario)
        mode = getattr(config, "demand_shift_mode", "none") or "none"
        if mode == "burst":
            assert config.burst_zone is not None, "burst mode needs burst_zone"
            assert config.burst_share_pct is not None, "burst mode needs burst_share_pct"
            assert centroids is not None
            gen = BurstOrderGenerator(
                base_generator=gen,
                network=network,
                relocation_centroids=centroids,
                burst_zone=config.burst_zone,
                burst_start=config.burst_start_min,
                burst_end=config.burst_end_min,
                burst_share=config.burst_share_pct,
                seed=self._seed,
            )
        elif mode == "permute":
            assert config.perm_strength is not None, "permute mode needs perm_strength"
            assert centroids is not None
            n_zones = len(centroids)
            offset = config.perm_offset if config.perm_offset is not None else n_zones // 2
            permutation = [(i + offset) % n_zones for i in range(n_zones)]
            gen = PermutedZoneOrderGenerator(
                base_generator=gen,
                network=network,
                relocation_centroids=centroids,
                permutation=permutation,
                strength=config.perm_strength,
                seed=self._seed,
            )
        elif mode == "localhub":
            assert centroids is not None
            gen = LocalHubOrderGenerator(
                base_generator=gen,
                network=network,
                relocation_centroids=centroids,
                stage1_zone=config.localhub_stage1_zone,
                stage2_zone=config.localhub_stage2_zone,
                switch_min=config.localhub_switch_min,
                region_neighbours=self._region_neighbours_from_centroids(centroids),
                seed=self._seed,
            )
        elif mode == "surge2":
            assert centroids is not None
            gen = SequentialSurgeOrderGenerator(
                base_generator=gen,
                network=network,
                relocation_centroids=centroids,
                stage1_zone=config.surge_zone1,
                stage2_zone=config.surge_zone2,
                surge_pct=config.surge_pct,
                stage1_end_min=config.surge_stage1_end_min,
                stage2_start_min=config.surge_stage2_start_min,
                seed=self._seed,
            )
        elif mode == "hotspot2":
            assert centroids is not None
            gen = TwoZoneHotspotOrderGenerator(
                base_generator=gen,
                network=network,
                relocation_centroids=centroids,
                hotspot_zones=(config.hotspot_zone_a, config.hotspot_zone_b),
                hotspot_pct=config.hotspot_pct,
                split=config.hotspot_split,
                seed=self._seed,
            )
        elif mode == "drift":
            assert centroids is not None
            gen = DriftingHotspotOrderGenerator(
                base_generator=gen,
                network=network,
                relocation_centroids=centroids,
                source_zone=config.drift_source_zone,
                dest_zone=config.drift_dest_zone,
                ramp_start_min=config.drift_ramp_start_min,
                ramp_end_min=config.drift_ramp_end_min,
                source_pct=config.drift_source_pct,
                seed=self._seed,
            )
        elif mode != "none":
            raise ValueError(f"Unknown demand_shift_mode: {mode!r}")

        return gen

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
    def _region_neighbours_from_centroids(
        centroids: list[tuple[float, float]], k: int = 8,
    ) -> tuple[tuple[int, ...], ...]:
        """K-nearest-centre adjacency, matching RideGym's centroid model.

        Order generators are built before the RideGym env exists, so the
        adjacency graph has to be derived here. The rule mirrors
        ``RidePoolEnv._build_region_neighbours`` (K nearest other centres by
        Euclidean distance, K = relocation_neighbours_k, default 8); the
        constructor asserts the two agree.
        """
        arr = np.asarray(centroids, dtype=float)
        n = len(arr)
        if n == 0:
            return tuple()
        kk = max(0, min(k, n - 1))
        out = []
        for i in range(n):
            d2 = ((arr - arr[i]) ** 2).sum(axis=1)
            d2[i] = np.inf
            out.append(tuple(int(j) for j in np.argsort(d2)[:kk]))
        return tuple(out)

    @staticmethod
    def _manifest_zone_ids(config: EnvConfig) -> set[int] | None:
        """Zone subset declared by the splits manifest, if it declares one.

        The manifest is the single source of truth for which taxi zones the
        study uses, so a zone subset never has to be restated in code.
        """
        manifest_path = Path(config.nyc_splits_dir) / "manifest.json"
        if not manifest_path.exists():
            return None
        ids = json.loads(manifest_path.read_text()).get("zone_ids")
        return {int(v) for v in ids} if ids else None

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
            selected = RideGymAEC._manifest_zone_ids(config)
            if selected is not None:
                inside = inside[inside["LocationID"].isin(selected)]
                if inside.empty:
                    raise ValueError(
                        f"manifest zone_ids {sorted(selected)} match no in-bounds centroid."
                    )
            return list(zip(inside["lon"].astype(float), inside["lat"].astype(float)))

        rows, cols = config.relocation_grid
        xmin, ymin, xmax, ymax = bounds
        xs = (np.arange(cols) + 0.5) / cols * (xmax - xmin) + xmin
        ys = (np.arange(rows) + 0.5) / rows * (ymax - ymin) + ymin
        return [(float(x), float(y)) for y in ys for x in xs]

    @property
    def relocation_points(self) -> list[tuple[float, float]]:
        """Centroid coordinates of all regions/zones."""
        return list(self.env.relocation_points)

    def _normalized_centroids(self) -> np.ndarray:
        pts = np.zeros((self._num_zones, 2), dtype=np.float32)
        for i, (rx, ry) in enumerate(self.env.relocation_points):
            nx, ny = _normalize_xy(rx, ry, self._geo_min, self._geo_range)
            pts[i] = (nx, ny)
        return pts

    def _make_obs_space(self) -> Dict:
        z = self._num_zones
        p = self._max_pending
        h = max(int(getattr(self._cfg, "demand_history_len", 4)), 1)
        # Zone-controller observation (D23): own-zone state + all-zone
        # fleet state + pending candidate orders + clock. Zone identity comes
        # from zone_location.
        return Dict({
            "zone_location": Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            "zone_state": Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "all_zone_state": Box(-1.0, 1.0, shape=(z, 4), dtype=np.float32),
            "pending_orders": Box(0.0, 1.0, shape=(p, 6), dtype=np.float32),
            "pending_mask": Box(0.0, 1.0, shape=(p,), dtype=np.float32),
            "pending_passengers": Box(0.0, 1.0, shape=(p,), dtype=np.float32),
            "neighbor_mask": Box(0.0, 1.0, shape=(z,), dtype=np.float32),
            "demand_hist": Box(0.0, 1.0, shape=(z, h), dtype=np.float32),
            "time_frac": Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            **({"inbound": Box(0.0, 1.0, shape=(z,), dtype=np.float32)}
               if self._cfg.obs_inbound else {}),
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
        # Vectorized (D17): per-order normalized features are computed ONCE
        # for all pending orders and reused for the global urgency view and
        # every zone's candidate view. Previously each order was normalized
        # 1 + num_zones times via per-order Python calls, and each zone ran
        # a full Python-keyed sort. Ordering semantics are unchanged:
        #   global: waiting_time desc (stable),
        #   per-zone: distance asc, waiting_time desc (stable).
        # np.argsort/lexsort with kind="stable" reproduce Python's stable
        # sort order exactly (identical floats: same IEEE op order).
        capacity = max(cfg.driver_capacity, 1)
        timeout = max(cfg.order_timeout, 1e-9)
        self._pending_features[:] = 0.0
        self._pending_mask[:] = 0.0
        self._pending_per_zone[:] = 0.0
        self._zone_pending_features[:] = 0.0
        self._zone_pending_mask[:] = 0.0
        self._zone_candidate_orders = [[] for _ in range(self._num_zones)]
        cents = np.asarray(self.env.relocation_points, dtype=float)
        max_r_km = getattr(cfg, "max_dispatch_radius_km", None)
        max_r_deg_sq = ((max_r_km / 111.0) ** 2) if (max_r_km is not None and max_r_km > 0) else None

        if self._current_pending:
            pending = self._current_pending
            num_p = len(pending)
            origins = np.asarray(
                [o["origin"] for o in pending], dtype=float,
            ).reshape(-1, 2)
            dests = np.asarray(
                [o["destination"] for o in pending], dtype=float,
            ).reshape(-1, 2)
            waits = np.asarray(
                [float(o.get("waiting_time", 0.0)) for o in pending],
                dtype=float,
            )
            passengers = np.asarray(
                [float(o["num_passengers"]) for o in pending], dtype=float,
            )
            gmin = np.asarray(self._geo_min, dtype=float)
            grange = np.maximum(np.asarray(self._geo_range, dtype=float), 1e-9)
            feat = np.empty((num_p, 6), dtype=np.float32)
            feat[:, 0:2] = (origins - gmin) / grange
            feat[:, 2:4] = (dests - gmin) / grange
            feat[:, 4] = passengers / capacity
            feat[:, 5] = np.minimum(waits / timeout, 1.0)

            # Global view: most urgent first, capped to max_pending.
            urg = np.argsort(-waits, kind="stable")
            keep = min(num_p, self._max_pending)
            self._pending_features[:keep] = feat[urg[:keep]]
            self._pending_mask[:keep] = 1.0

            # Per-order zone attribution (all pending, not capped).
            dists = ((origins[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            nearest = dists.argmin(axis=1)
            self._pending_per_zone[:] = np.bincount(
                nearest, minlength=self._num_zones,
            )
            # D22: was /= max_pending (650), which crushed the demand signal
            # to ~0.001/pending-order. Normalize by fleet size so one pending
            # order reads ~0.02 and a full fleet-worth saturates at 1.0.
            self._pending_per_zone /= max(self._num_drivers, 1)

            # Per-zone candidate orders: strict zone ownership (D23).
            # Orders are partitioned by origin zone centroid (nearest == zi),
            # completely eliminating multi-agent dispatch collision and race conditions.
            for zi in range(self._num_zones):
                idx = np.where(nearest == zi)[0]
                if max_r_deg_sq is not None and idx.size:
                    idx = idx[dists[idx, zi] <= max_r_deg_sq]
                if idx.size:
                    d = dists[idx, zi]
                    order = idx[np.lexsort((d, -waits[idx]))][
                        : self._max_pending
                    ]
                else:
                    order = idx
                n_cand = int(order.shape[0])
                if n_cand:
                    self._zone_candidate_orders[zi] = [pending[i] for i in order]
                    self._zone_pending_features[zi, :n_cand] = feat[order]
                    self._zone_pending_mask[zi, :n_cand] = 1.0

        # --- Per-zone fleet state -------------------------------------------
        zone_totals = np.zeros(self._num_zones, dtype=np.float32)
        zone_idle = np.zeros(self._num_zones, dtype=np.float32)
        zone_dispatchable = np.zeros(self._num_zones, dtype=np.float32)
        allow_reloc_disp = getattr(cfg, "allow_relocating_dispatch", True)

        for did in range(self._num_drivers):
            d = self._last_obs[did]["self"]
            zi = int(d.get("current_region", 0))
            zi = max(0, min(zi, self._num_zones - 1))
            zone_totals[zi] += 1.0
            if d["status"] == "idle":
                zone_idle[zi] += 1.0
                zone_dispatchable[zi] += 1.0
            elif allow_reloc_disp and d["status"] == "relocating":
                zone_dispatchable[zi] += 1.0

        self._zone_fleet_state[:, 0] = zone_totals / max(self._num_drivers, 1)
        self._zone_fleet_state[:, 1] = zone_idle / max(self._num_drivers, 1)
        self._zone_fleet_state[:, 2] = self._pending_per_zone
        # Column 3 (D23): explicit supply-demand deficit (>0 deficit, <0 surplus)
        self._zone_fleet_state[:, 3] = self._pending_per_zone - (zone_idle / max(self._num_drivers, 1))

        # D37: per-channel normalisation of all_zone_state.
        #
        # The four columns previously shared a raw scale and were wildly
        # mismatched. Measured over 400 turns on NYC16 (env 3.0), before this
        # change:
        #
        #   col  fleet total/26   max 0.3846   std 0.1016
        #   col  idle/26          max 0.0385   std 0.0127
        #   col  pending/26       max 0.1538   std 0.0402   <- the demand signal
        #   col  deficit          max 0.1154   std 0.0359
        #
        # Demand (col 2) was ~2.5x weaker than fleet size, and ONE waiting rider
        # in a 26-car fleet reads 1/26 = 0.0385 -- a 0.04 difference on a channel
        # whose neighbour swings by 0.38. The policy could not resolve demand at
        # that resolution, which is consistent with it relocating to demand-free
        # zones 81.6% of the time (LP: 36.2%) and with raising beta changing
        # nothing: a uniform cost cannot fix a missing signal.
        #
        # Each column is now divided by its own running max-abs scale, so a
        # channel occupies the full [-1, 1] range of its Box regardless of fleet
        # size or demand volume. Scales are per-instance state, not constants, so
        # no magic numbers and no fleet-size coupling (col 2 no longer shrinks
        # when the fleet grows).
        for col in range(4):
            # D41: divide by the RUNNING max, and never shrink it. The previous
            # version (D37) divided by the CURRENT turn's max and only recorded
            # the running value without using it, so the same physical
            # configuration encoded differently depending on this turn's peak.
            # Non-stationary input to both value functions.
            m = float(np.max(np.abs(self._zone_fleet_state[:, col])))
            if m > 1e-9:
                self._obs_scales[col] = max(self._obs_scales[col], m)
            s = self._obs_scales[col]
            if s > 1e-9:
                self._zone_fleet_state[:, col] /= s
        self._zone_idle_counts = zone_idle.astype(np.int64)
        # D37: raw per-zone totals, kept so observe() can form ratios without
        # de-normalising the scaled matrix.
        self._zone_total_counts = zone_totals.astype(np.int64)
        self._zone_dispatchable_counts = zone_dispatchable.astype(np.int64)
        self._demand_history.append(self._pending_per_zone.copy())
        self._demand_hist_arr[:] = 0.0
        n_hist = len(self._demand_history)
        if n_hist:
            self._demand_hist_arr[:, -n_hist:] = np.stack(list(self._demand_history), axis=1)
            # D37: same per-channel treatment for the demand history. It is the
            # only other place raw pending demand reaches the policy, and it had
            # the same 0.04-vs-0.38 scale problem as all_zone_state col 2.
            m = float(np.max(np.abs(self._demand_hist_arr)))
            if m > 1e-9:
                # D41: running max, and scale the history ONCE per turn using
                # that running max. D37 divided the whole history array by the
                # max across all entries, so whenever a new peak arrived every
                # PAST entry was retroactively rescaled -- the encoding of a
                # configuration depended on peaks that happened later.
                self._demand_hist_scale = max(self._demand_hist_scale, m)
            if self._demand_hist_scale > 1e-9:
                self._demand_hist_arr /= self._demand_hist_scale

        # Clear active relocation target if driver is no longer in status RELOCATING
        if hasattr(self, "_driver_relocation_target"):
            for did in range(self._num_drivers):
                d = self._last_obs[did]["self"]
                if d["status"] != "relocating":
                    self._driver_relocation_target.pop(did, None)

        # D43 note: the inbound-relocation aggregation deliberately does NOT live
        # here. It must run after this turn's targets are assigned, which is
        # inside _flush_turn. Computing it here read the previous turn's
        # targets and then overwrote the fresh value.

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
        self._seen_completed_ids.clear()
        self._driver_commander.clear()
        self._driver_relocator.clear()
        self._order_relocator_paid.clear()
        self._driver_reloc_chain.clear()
        self._order_relocator_shares.clear()
        self._pending_dispatch.clear()
        self._driver_relocation_target.clear()
        self._demand_history.clear()
        self._reloc_out = {"executed": 0, "dispatched": 0, "in_target": 0}
        self._reloc_to: dict[int, int] = {}
        self._obs_scales[:] = 0.0
        self._demand_hist_scale = 0.0
        self._zone_inbound_reloc[:] = 0.0
        self._inbound_reloc_scale = 0.0
        self._claimed_orders_this_turn: set[int] = set()
        self._reloc_inbound_this_turn: dict[int, int] = {}
        self._seen_cancelled_ids: set[int] = set()

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
        # D37: all_zone_state is per-channel normalised, so ratios must be taken on
        # RAW counts kept alongside it. Deriving them by de-normalising gave
        # idle_frac up to 1.5, outside this Box's (-1, 1) bound -- caught by
        # api_test's out-of-bounds check.
        zone_total_raw = float(self._zone_total_counts[zone_idx])
        zone_idle_frac = (
            float(self._zone_idle_counts[zone_idx]) / zone_total_raw
            if zone_total_raw > 0 else 0.0
        )
        zone_drivers = (
            self._zone_fleet_state[zone_idx, 0]
            if self._obs_scales[0] > 1e-9 else 0.0
        )
        zone_pending = (
            self._zone_fleet_state[zone_idx, 2]
            if self._obs_scales[2] > 1e-9 else 0.0
        )
        zone_state = np.array(
            [
                zone_drivers,
                zone_idle_frac,
                zone_pending,
                float(self._zone_fleet_state[zone_idx, 3]),
            ],
            dtype=np.float32,
        )
        neigh_vec = np.zeros(self._num_zones, dtype=np.float32)
        neigh_vec[self._legal_relocate_targets(zone_idx)] = 1.0

        extra = {}
        if self._cfg.obs_inbound:
            inbound = np.zeros(self._num_zones, dtype=np.float32)
            for tgt in self._driver_relocation_target.values():
                inbound[int(tgt)] += 1.0
            for tgt, n in self._reloc_inbound_this_turn.items():
                inbound[int(tgt)] += n
            extra["inbound"] = np.minimum(inbound / max(self._num_drivers, 1), 1.0)

        return {
            **extra,
            "zone_location": self._zone_centroids_norm[zone_idx],
            "zone_state": zone_state,
            "all_zone_state": self._zone_fleet_state,
            "pending_orders": self._zone_pending_features[zone_idx],
            "pending_mask": self._zone_pending_mask[zone_idx],
            "pending_passengers": self._zone_pending_features[zone_idx, :, 4],
            "neighbor_mask": neigh_vec,
            "demand_hist": self._demand_hist_arr,
            "time_frac": np.array(
                [min(self.env.time / max(self.env.horizon, 1e-9), 1.0)],
                dtype=np.float32,
            ),
        }

    def _hops_to_nearest_demand(self) -> dict[int, int] | None:
        """BFS hop distance from every zone to the nearest zone with demand."""
        demand = [j for j in range(self._num_zones) if self._pending_per_zone[j] > 0]
        if not demand:
            return None
        dist = {j: 0 for j in demand}
        frontier = deque(demand)
        while frontier:
            cur = frontier.popleft()
            nbrs = (self.env.region_neighbours[cur]
                    if cur < len(self.env.region_neighbours) else ())
            for j in nbrs:
                if j not in dist:
                    dist[j] = dist[cur] + 1
                    frontier.append(j)
        return dist

    def _legal_relocate_targets(self, zone_idx: int) -> list[int]:
        counts = getattr(self, "_zone_idle_counts", None)
        if self._last_obs is None or counts is None:
            return []
        if not bool(counts[zone_idx] > 0):
            return []

        neighbor_only = getattr(self._cfg, "neighbor_relocation_only", False)
        neighbours = (
            self.env.region_neighbours[zone_idx]
            if (neighbor_only and zone_idx < len(self.env.region_neighbours))
            else range(self._num_zones)
        )
        inbound = getattr(self, "_reloc_inbound_this_turn", None)
        claim = bool(getattr(self._cfg, "claim_relocation_targets", True))
        require_demand = bool(getattr(self._cfg, "require_destination_demand", True))

        def pending_at(j: int) -> int:
            return int(np.round(self._pending_per_zone[j] * max(self._num_drivers, 1)))

        def surplus_ok(j: int) -> bool:
            return int(counts[j]) <= max(pending_at(j), 1)

        hops = self._hops_to_nearest_demand() if require_demand else None
        demanded_here = any(
            pending_at(k) >= 1 and surplus_ok(k) for k in neighbours
        )

        out: list[int] = []
        for j in neighbours:
            if j == zone_idx or not (0 <= j < self._num_zones):
                continue
            if not surplus_ok(j):
                continue
            # D34 intra-turn arrival claim: every zone acts on the same
            # pre-turn snapshot, so without a claim they all pick the same
            # destination and flood it (104 drivers into one zone per turn).
            if (claim and inbound is not None
                    and int(inbound.get(j, 0)) >= max(pending_at(j), 1)):
                continue
            if not require_demand or pending_at(j) >= 1:
                out.append(int(j))
                continue
            # No demand here. Admit it only if it moves the car closer to the
            # nearest demand AND no neighbour already has demand, so a car
            # several hops from a hotspot can start closing in instead of
            # being frozen. Demanded neighbours are preferred, which keeps the
            # common menu free of demand-free options.
            if (hops is not None and not demanded_here
                    and hops.get(j, 99) < hops.get(zone_idx, 99)):
                out.append(int(j))
        # D77: keep only the k nearest legal targets, so a zone hands cars to
        # its surroundings instead of across the map.
        k = int(self._cfg.relocation_nearest_k)
        if 0 < k < len(out):
            xy = np.asarray(self._zone_centroids_norm)
            out.sort(key=lambda j: float(np.linalg.norm(xy[j] - xy[zone_idx])))
            out = out[:k]
        return out

    def action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(self._action_dim, dtype=bool)
        mask[0] = True  # no zone action is always valid
        if self._last_obs is None:
            return mask
        zone_idx = self._zone_int_id(agent)
        # Dispatch orders: valid if there is at least one dispatchable driver (idle or relocating, D14)
        has_dispatchable = bool(getattr(self, "_zone_dispatchable_counts", self._zone_idle_counts)[zone_idx] > 0)
        if has_dispatchable:
            candidates = self._zone_candidate_orders[zone_idx] if zone_idx < len(self._zone_candidate_orders) else []
            for i, cand in enumerate(candidates[: self._max_pending]):
                # Intra-turn collision masking (D15): mask order if already claimed this turn
                if cand["order_id"] not in getattr(self, "_claimed_orders_this_turn", set()):
                    mask[1 + i] = True

        # Relocate actions mirror _legal_relocate_targets (single source of truth)
        for j in self._legal_relocate_targets(zone_idx):
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
            return None
        target_zone = action - self._max_pending - 1
        if 0 <= target_zone < self._num_zones and target_zone != zone_idx:
            if getattr(self._cfg, "neighbor_relocation_only", False):
                neighbours = self.env.region_neighbours[zone_idx] if zone_idx < len(self.env.region_neighbours) else []
                if target_zone not in neighbours:
                    return None
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

    def _newly_completed_ids(self) -> list[int]:
        """Order ids that reached COMPLETED since the last call.

        Completion, not dispatch, is when a zone earns: it is the only point
        where "sent a car" and "a passenger was actually carried" differ.
        """
        from ride_gym.enums import OrderStatus

        done = {
            oid for oid, o in self.env.orders.items()
            if o.status == OrderStatus.COMPLETED
        }
        new = done - self._seen_completed_ids
        self._seen_completed_ids = done
        return list(new)

    def _compute_zone_rewards(
        self,
        events: dict[int, dict],
        dispatched_this_turn: list[tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """Compute localized credit assignment rewards per zone controller.

        R_z = (1 - team_share) * R_local,z + team_share * R_global_mean

        Local credit assignment (D23):
        1. Served demand (+gamma): credited IMMEDIATELY to the zone controller that
           dispatched the order at this turn (with rebalancing assist sharing).
           Zero temporal credit delay.
        2. Empty deadhead distance penalty (-beta): charged directly to the zone
           controller that commanded the driver (relocations shared 50/50).
        3. Pending waiting time penalty (-alpha): charged to the zone containing
           the pending order's origin (nearest zone centroid).
        4. Cancellation penalty (-cancel_penalty): shared between origin zone and
           neighboring zones holding idle cars.
        """
        from ride_gym.enums import OrderStatus

        cfg = self._cfg
        local_rewards = np.zeros(self._num_zones, dtype=np.float32)
        decision_bonus = getattr(self, "_decision_bonus", None)
        if decision_bonus is not None:
            local_rewards += decision_bonus
            decision_bonus[:] = 0.0
        cents = np.asarray(self.env.relocation_points, dtype=float)

        self._prev_served = sum(
            1 for o in self.env.orders.values()
            if o.status == OrderStatus.COMPLETED
        )

        # 1. Served-demand credit (+gamma) paid on COMPLETION, not on dispatch.
        # Dispatch-time credit paid a zone in full for aiming a car at a waiting
        # rider, whether or not that rider was ever picked up, so two separate
        # gestures (dispatch + arrival) were both paid while the outcome was
        # never paid. The do-nothing policy then becomes competitive and the
        # agent drifts toward freezing. Completion is the only moment that
        # distinguishes a useful relocation from a wasted one.
        #
        # When the serving driver was moved here by another zone's relocation
        # decision, that zone keeps a share: this is the only place a
        # rebalancing decision earns anything.
        #
        # D47: the share is NOT divided. `_driver_relocator` is a single zone
        # -- the one that commanded the most recent relocation of this driver.
        # An earlier version accumulated every zone that had moved the car
        # since its last completed order and split the share across them, on
        # the theory that a relay should pay each hop. Measured, that list was
        # not a relay: median 20 turns between hops of the same car, 0.4%
        # back-to-back, and 87% of hop pairs had the car serve an order in
        # between. Dividing by it charged zones for having their cars reused.
        # Each relocation is now an independent A->B event paid in full to A.
        for oid in self._newly_completed_ids():
            order = self.env.orders[oid]
            did = getattr(order, "assigned_driver", None)
            dispatcher = self._order_dispatcher.get(oid)
            if dispatcher is None or not (0 <= dispatcher < self._num_zones):
                # D59: release the deadhead liability even when there is no
                # dispatcher to pay. Doing this only below the `continue` left a
                # stale commander on the driver, which then billed the next job's
                # deadhead to whichever zone relocated last.
                if did is not None:
                    self._driver_relocator.pop(did, None)
                    self._driver_reloc_chain.pop(did, None)
                    self._driver_commander.pop(did, None)
                continue
            # D49: the dispatcher is paid the FULL served credit, always.
            #
            # D48 paid it (1 - relocator_share) whenever the car had been
            # relocated. That was wrong: the zone did identical work either way
            # -- it dispatched a car to a rider and the ride completed -- so it
            # earned 45% less for using the car another zone had pre-positioned
            # for it. That penalised the outcome rebalancing exists to produce,
            # and under team_share=0.2 the local term carries 80% of the weight,
            # so the penalty was real rather than diluted.
            #
            # The relocator's share is a BONUS for a separate action, not a
            # slice carved out of the dispatcher's credit. Both agents did
            # distinct work, so both are paid for it.
            relocator_paid = self._order_relocator_paid.pop(oid, None)
            self._order_relocator_shares.pop(oid, None)
            served_credit = cfg.gamma
            # D71: in split mode the relocator's payment comes out of this
            # ride's gamma, so the zone rewards sum to the team objective and
            # the shared policy is not paid for relocation volume.
            if cfg.relocator_share_is_split and relocator_paid is not None:
                served_credit -= relocator_paid[1]
            local_rewards[dispatcher] += served_credit
            if did is not None:
                self._driver_relocator.pop(did, None)
                self._driver_reloc_chain.pop(did, None)
                # D59: the relocating zone's liability for this job ends with
                # the job. Section 1c clears it at dispatch, but a driver can
                # also be dispatched and then complete, so clear here too --
                # otherwise a stale commander bills the NEXT job's deadhead to
                # whichever zone relocated last.
                self._driver_commander.pop(did, None)

        # 1c. D48: the relocating zone is paid when its car is DISPATCHED to a
        # rider, not when that ride completes.
        #
        # Why: the relocation decision pays -beta*km immediately, so evidence
        # that it was right should arrive on a comparable timescale. Measured
        # over 131 relocation legs that reached their destination, 98.3% of the
        # cars that were ever dispatched were dispatched ON the arrival turn --
        # a median of 0 turns. Waiting for completion meant paying 21-40 turns
        # later, at which point GAE had attenuated the signal (2.5% survival at
        # lambda=0.95, see D46).
        #
        # Dispatch is verification, not optimism: a car that arrives where
        # nobody is waiting is never dispatched, so it earns nothing. That is
        # the line between this and D28's arrival bonus, which paid for the
        # arrival itself and was switched off precisely because it paid without
        # proof. Here 55.7% of arrivals produce no dispatch and no credit.
        #
        # The relocator's share is a BONUS on top of the dispatcher's full served
        # credit (D49), not a slice of it. A ride served by a rebalanced car is
        # therefore worth 1.45 gamma rather than 1.0. That is intentional: two
        # distinct value-creating actions occurred -- zone A moved a car to
        # where it was needed, zone B put it to work.
        for oid, did, zone_idx in (dispatched_this_turn or ()):
            relocator = self._driver_relocator.get(did)
            if relocator is not None and relocator != zone_idx:
                if 0 <= relocator < self._num_zones:
                    # D66: pay only when the moved car was the marginal one, i.e.
                    # the destination is short of cars and still has riders waiting
                    # after this dispatch. Rebuilt caches make idle/pending post-dispatch.
                    short = (
                        self._zone_idle_counts[zone_idx] == 0
                        and self._pending_per_zone[zone_idx] * max(self._num_drivers, 1) > 0.5
                    )
                    paid = cfg.relocator_share * cfg.gamma if (short or not cfg.relocator_requires_shortage) else 0.0
                    if paid > 0.0:
                        # D82: divide the payment among every zone that moved this
                        # car since its last dispatch. Paying only the last mover
                        # left the earlier one with the cost and no credit, and let
                        # a zone capture the whole payment by re-sending a car.
                        chain = self._driver_reloc_chain.get(did) if cfg.relocator_chain_share else None
                        payees = [z for z in dict.fromkeys(chain or ()) if z != zone_idx] or [relocator]
                        shares = [(z, paid / len(payees)) for z in payees]
                        for z, amount in shares:
                            local_rewards[z] += amount
                        self._order_relocator_paid[oid] = (relocator, paid)
                        self._order_relocator_shares[oid] = shares
            self._driver_relocator.pop(did, None)
            self._driver_reloc_chain.pop(did, None)

        # 1b. Relocation arrival bonus: off by default. It paid the sending zone
        # for the arrival itself, which made moving look profitable without the
        # moved car ever serving anyone. Kept as an opt-in nudge, not a payment.
        prospective_bonus = float(getattr(cfg, "prospective_bonus_scale", 0.0))
        if prospective_bonus > 0.0:
            for did in getattr(self, "_relocator_arrivals", []):
                relocator = self._driver_relocator.get(did)
                if relocator is not None and 0 <= relocator < self._num_zones:
                    d = self._last_obs[did]["self"]
                    dest_zone = int(d.get("current_region", 0))
                    dest_zone = max(0, min(dest_zone, self._num_zones - 1))
                    pending_at_dest = float(self._pending_per_zone[dest_zone])
                    local_rewards[relocator] += prospective_bonus * pending_at_dest

        # 2. Empty deadhead distance penalty (-beta) charged to commanding zone
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

        # 3. Pending waiting time penalty (-alpha) — volume-scaled (D15)
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
                    w_sum_norm = float(np.sum(zone_wait_times[zi])) / max(cfg.order_timeout, 1e-9)
                    # D22: removed the hidden extra x0.1 (now folded into
                    # alpha=0.1 default) so the config value is the true weight.
                    local_rewards[zi] -= cfg.alpha * w_sum_norm

        # 4. Cancellation penalty (D23: Spatially-shared with neighbor surplus)
        cancel_penalty = float(getattr(cfg, "cancel_penalty", 0.5))
        # Cancellation is detected regardless of the penalty weight. The D48
        # clawback below is part of the served-credit accounting, not a
        # penalty, so it must not be gated on cancel_penalty > 0 -- doing that
        # silently overpaid every relocated ride that was later cancelled.
        current_cancelled = {
            oid for oid, o in self.env.orders.items()
            if o.status == OrderStatus.CANCELLED
        }
        newly_cancelled = current_cancelled - getattr(self, "_seen_cancelled_ids", set())
        if not hasattr(self, "_seen_cancelled_ids"):
            self._seen_cancelled_ids = set()
        self._seen_cancelled_ids.update(newly_cancelled)
        for oid in newly_cancelled:
            # D48 clawback: the relocator was paid at dispatch, before it was
            # known whether the ride would survive. Measured, 40.6% of relocator
            # payouts went to orders that were later cancelled, which left a
            # relocated cancelled ride paying 0.45 while a plain cancelled ride
            # paid nothing -- a bias toward relocating into zones where orders
            # time out. Reversing the share keeps one gamma per completed ride.
            paid = self._order_relocator_paid.pop(oid, None)
            if paid is not None and 0 <= paid[0] < self._num_zones:
                for z, amount in self._order_relocator_shares.pop(oid, [paid]):
                    local_rewards[z] -= amount
        if cancel_penalty > 0.0:
            for oid in newly_cancelled:
                order = self.env.orders.get(oid)
                if order is not None:
                    orig = np.asarray(order.origin, dtype=float)
                    zi = int(np.argmin(np.sum((cents - orig) ** 2, axis=1)))
                    # Find neighbors of zi
                    neighbours = list(self.env.region_neighbours[zi]) if (hasattr(self.env, "region_neighbours") and zi < len(self.env.region_neighbours)) else []
                    # Check idle counts in neighbors
                    idle_neighbours = [n for n in neighbours if n < self._num_zones and self._zone_idle_counts[n] > 0]
                    if idle_neighbours:
                        # Zone zi pays 50%, neighbors with idle cars split remaining 50%
                        local_rewards[zi] -= 0.5 * cancel_penalty
                        penalty_per_n = 0.5 * cancel_penalty / len(idle_neighbours)
                        for n in idle_neighbours:
                            local_rewards[n] -= penalty_per_n
                    else:
                        local_rewards[zi] -= cancel_penalty
                else:
                    local_rewards -= cancel_penalty / self._num_zones

        # 5. Team cooperation blending (e.g. 80% local + 20% global team)
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

        # Per-zone FIFO queues of drivers (D14):
        # Purely idle drivers are queued first; empty relocating drivers are queued second.
        allow_reloc_disp = getattr(self._cfg, "allow_relocating_dispatch", True)
        zone_idle_queues: dict[int, deque[int]] = {
            zi: deque() for zi in range(self._num_zones)
        }
        zone_relocating_queues: dict[int, deque[int]] = {
            zi: deque() for zi in range(self._num_zones)
        }
        for did in range(self._num_drivers):
            d = self._last_obs[did]["self"]
            zi = max(0, min(int(d.get("current_region", 0)), self._num_zones - 1))
            if d["status"] == "idle":
                zone_idle_queues[zi].append(did)
            elif allow_reloc_disp and d["status"] == "relocating":
                zone_relocating_queues[zi].append(did)

        ridegym_actions: dict[int, dict | None] = {}
        claimed_orders: set[int] = set()
        dispatched_this_turn: list[tuple[int, int, int]] = []

        for order_pos in self._agent_order:
            agent = self.agents[int(order_pos)]
            zone_idx = self._zone_int_id(agent)
            action = self._actions_this_turn.get(agent, 0)
            decoded = self._decode_action(action, zone_idx)
            if decoded is None:
                continue

            if "orders" in decoded:
                # Dispatch order: prefer purely idle driver; fallback to relocating driver
                driver_id = None
                from_queue = None
                if zone_idle_queues[zone_idx]:
                    driver_id = zone_idle_queues[zone_idx].popleft()
                    from_queue = zone_idle_queues[zone_idx]
                elif zone_relocating_queues[zone_idx]:
                    driver_id = zone_relocating_queues[zone_idx].popleft()
                    from_queue = zone_relocating_queues[zone_idx]

                if driver_id is None:
                    continue  # masked out upstream; defensive no-op

                oid = decoded["orders"][0]
                if oid in claimed_orders:
                    # Order claimed by an earlier zone agent this turn.
                    from_queue.appendleft(driver_id)
                    continue

                claimed_orders.add(oid)
                # D51: record the intent but do NOT commit it yet. RideGym
                # silently drops bids on orders that are no longer PENDING, and
                # its `_cancel_timed_out()` runs before the assignment step, so
                # an order in the last dt of patience is advertised in the
                # menu, masked legal, and already expired when picked.
                #
                # Committing here unconditionally left three phantom records per
                # such void (D50, measured 1.1-2.5% of dispatches): a dispatcher
                # that never dispatched, a commander overwritten to the
                # dispatching zone so the abandoned leg's remaining deadhead was
                # billed to the wrong zone, and `_driver_relocator.pop(did)`
                # destroying the sending zone's credit for a relocation that was
                # never cancelled.
                #
                # The provisional entries are committed after `env.step` below,
                # once RideGym's outcome is known.
                self._pending_dispatch = getattr(
                    self, "_pending_dispatch", [])
                self._pending_dispatch.append((oid, driver_id, zone_idx))
                ridegym_actions[driver_id] = decoded

            elif "relocate" in decoded:
                # Relocation actions only consume purely idle drivers
                if not zone_idle_queues[zone_idx]:
                    continue
                driver_id = zone_idle_queues[zone_idx].popleft()
                target_zone = decoded["relocate"]
                self._driver_commander[driver_id] = zone_idx
                # D47: each relocation is an independent A->B event. Overwrite
                # rather than accumulate -- see the note on the declaration.
                # Only the most recent move is credited, and only if that car
                # later completes an order.
                self._driver_relocator[driver_id] = zone_idx
                self._driver_reloc_chain.setdefault(driver_id, []).append(zone_idx)
                self._driver_relocation_target[driver_id] = target_zone
                self._reloc_out["executed"] += 1
                self._reloc_to[driver_id] = int(target_zone)
                ridegym_actions[driver_id] = decoded
                bonus_scale = float(self._cfg.targeting_bonus)
                if bonus_scale > 0.0:
                    unmet = self._pending_per_zone * max(self._num_drivers, 1)
                    legal_unmet = [float(unmet[j]) for j in self._legal_relocate_targets(zone_idx)]
                    best = max(legal_unmet, default=0.0)
                    if best > 0.0:
                        if not hasattr(self, "_decision_bonus"):
                            self._decision_bonus = np.zeros(self._num_zones, dtype=np.float32)
                        # D73: centred on the mean legal target, so a random
                        # destination earns zero and moving is not paid for itself.
                        baseline = float(np.mean(legal_unmet)) if self._cfg.targeting_bonus_centred else 0.0
                        self._decision_bonus[zone_idx] += bonus_scale * (float(unmet[target_zone]) - baseline) / best

        # D43: refresh the per-zone inbound count AFTER targets are assigned.
        # _rebuild_caches runs before this loop, so aggregating there would
        # only ever expose the PREVIOUS turn's targets and the count read as
        # zero on the turn a car actually left.
        inbound = np.zeros(self._num_zones, dtype=np.float32)
        for did, tgt in self._driver_relocation_target.items():
            t = int(tgt)
            if 0 <= t < self._num_zones:
                inbound[t] += 1.0
        self._inbound_reloc_scale = max(
            self._inbound_reloc_scale, float(np.max(np.abs(inbound))),
        )
        if self._inbound_reloc_scale > 1e-9:
            self._zone_inbound_reloc = inbound / self._inbound_reloc_scale

        obs, _rewards, dones, info = self.env.step(ridegym_actions)

        # D51: commit only the dispatches RideGym actually honoured. An order
        # that is still PENDING or has become CANCELLED was never assigned, so
        # the wrapper must not claim credit for it, bill its deadhead to the
        # dispatching zone, or discard the relocating zone's credit.
        from ride_gym.enums import OrderStatus

        dispatched_this_turn = []
        for oid, did, zone_idx in getattr(self, "_pending_dispatch", ()):
            order = self.env.orders.get(oid)
            status = getattr(order, "status", None)
            if status not in (OrderStatus.ASSIGNED, OrderStatus.ONBOARD,
                              OrderStatus.COMPLETED):
                continue
            if getattr(order, "assigned_driver", None) != did:
                continue
            self._order_dispatcher[oid] = zone_idx
            # D59: do NOT overwrite the commander when the driver was relocated
            # here. The zone that chose the relocation pays for the WHOLE job it
            # caused -- the relocation leg AND the empty drive to the rider.
            #
            # Overwriting meant the sending zone paid only the relocation leg
            # (2.9 km x beta) while the pickup deadhead was billed to the zone
            # that happened to use the car. Relocating was therefore close to
            # free, which is what the measurement shows: 160 relocations x
            # 2.9 km = 464 km, against 476-481 km observed per episode. Deadhead
            # is essentially ALL relocation, and it was being charged to whoever
            # dispatched rather than to whoever moved the car.
            if did not in self._driver_relocator:
                self._driver_commander[did] = zone_idx
            if did in self._reloc_to:
                tgt = self._reloc_to.pop(did)
                self._reloc_out["dispatched"] += 1
                if tgt == zone_idx:
                    self._reloc_out["in_target"] += 1
            self._driver_relocation_target.pop(did, None)
            dispatched_this_turn.append((oid, did, zone_idx))
        self._pending_dispatch = []

        # Detect relocation arrivals BEFORE updating _last_obs (D28):
        # Drivers that were "relocating" in the old obs but are no longer
        # "relocating" in the new obs just completed their relocation hop.
        self._relocator_arrivals = []
        prospective_bonus = float(getattr(self._cfg, "prospective_bonus_scale", 0.0))
        if prospective_bonus > 0.0:
            for did in range(self._num_drivers):
                old_status = self._last_obs[did]["self"]["status"]
                new_status = obs[did]["self"]["status"]
                if old_status == "relocating" and new_status != "relocating":
                    self._relocator_arrivals.append(did)

        self._last_obs = obs
        self._current_pending = self._extract_pending(obs)
        self._rebuild_caches()

        # Compute localized credit assignment rewards (D23: immediate dispatch credit)
        zone_rewards = self._compute_zone_rewards(info.get("events", {}), dispatched_this_turn)

        # Standard AEC delivery: each agent reads its attributed reward
        self.rewards = {
            a: float(zone_rewards[self._zone_int_id(a)]) for a in self.agents
        }
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
                    "mean_trip_km": kpis["mean_trip_km"],
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
            self._claimed_orders_this_turn = set()
            self._reloc_inbound_this_turn = {}

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
        # D57: mean trip distance. Recorded because feasibility depends on it --
        # cars_required = arrival_rate x time_per_job, and time_per_job needs the
        # ride distance to convert deadhead km into deadhead time. Without this,
        # whether the fleet is sufficient is a range (0.55x to 1.09x the fleet on
        # NYC16) rather than a fact. Non-behavioural: a read-only statistic.
        trip_km = []
        for o in self.env.orders.values():
            if o.status != OrderStatus.COMPLETED:
                continue
            org = getattr(o, "origin", None)
            dst = getattr(o, "destination", None)
            if org is None or dst is None:
                continue
            trip_km.append(_haversine_km(org, dst))
        return {
            "served": float(served),
            "cancelled": float(cancelled),
            "total": float(len(self.env.orders)),
            "avg_wait": avg_wait,
            "empty_km": empty_km,
            "mean_trip_km": float(np.mean(trip_km)) if trip_km else 0.0,
            "p90_trip_km": (float(np.percentile(trip_km, 90))
                            if trip_km else 0.0),
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
        self._cumulative_rewards[current] = 0.0
        # Clear the whole rewards dict, not just this agent's entry.
        # _accumulate_rewards() already folded every zone's turn reward into
        # _cumulative_rewards at flush time, so the per-agent dict is spent by
        # this point and must not be left holding stale values from the previous
        # turn: an AEC consumer sums env.rewards after every step, so a surviving
        # entry double-counts (caught by PettingZoo's api_test).
        self.rewards = {a: 0.0 for a in self.agents}
        act_val = 0 if action is None else int(action)
        self._actions_this_turn[current] = act_val
        if 1 <= act_val <= self._max_pending:
            zone_idx = self._zone_int_id(current)
            candidates = self._zone_candidate_orders[zone_idx] if zone_idx < len(self._zone_candidate_orders) else []
            cand_idx = act_val - 1
            if cand_idx < len(candidates):
                if not hasattr(self, "_claimed_orders_this_turn"):
                    self._claimed_orders_this_turn = set()
                self._claimed_orders_this_turn.add(candidates[cand_idx]["order_id"])
        elif act_val > self._max_pending:
            zone_idx = self._zone_int_id(current)
            target = act_val - self._max_pending - 1
            if not hasattr(self, "_reloc_inbound_this_turn"):
                self._reloc_inbound_this_turn = {}
            if self._decode_action(act_val, zone_idx) is not None:
                self._reloc_inbound_this_turn[target] = (
                    self._reloc_inbound_this_turn.get(target, 0) + 1
                )
        self._advance_selection()

    def state(self) -> np.ndarray:
        """Centralized state for MAPPO's shared critic.

        D38: this now carries EVERY channel the actor observation exposes, not
        a subset. It previously returned 177 dims while the actor saw 279 --
        ``demand_hist`` (64) and ``neighbor_mask`` (16) were missing. Those
        arrived at D30 as observation-only additions and this function was
        never revisited, so the one component MAPPO exists to improve was blind
        to recent demand history and to which zones are legal targets. IPPO's
        decentralized critic sees those in its own 279-dim observation, so MAPPO
        was starting with LESS information than IPPO, which inverts the intended
        comparison.

        ``tests/test_fixes_and_demand.py::test_state_covers_every_observation_
        channel`` fails if a future observation channel is added without being
        added here, so this class of drift cannot recur silently.
        """
        if self._last_obs is None:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [
                self._zone_fleet_state.flatten(),
                self._pending_features.flatten(),
                self._pending_mask,
                self._demand_hist_arr.flatten(),
                self._legal_target_mask_flat(),
                self._inbound_reloc_counts(),
                np.array(
                    [min(self.env.time / max(self.env.horizon, 1e-9), 1.0)],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def _inbound_reloc_counts(self) -> np.ndarray:
        """D43: per-zone count of cars currently RELOCATING TOWARD it.

        Why this exists. Wasted relocation is overwhelmingly a car being sent
        where another zone is *already* sending one. Measured on trained MAPPO:
        the critic returned a BIT-IDENTICAL value (0.0390501097) for two
        different relocation destinations for 11 consecutive turns, because
        `state()` never encoded where cars were going -- a relocating car is
        just "in zone, not idle". The redundancy signal existed in the env
        (`_reloc_inbound_this_turn`, `_driver_relocation_target`) but reached
        only `action_mask`, never the critic. So a centralized critic could not
        penalise redundant targeting even in principle.

        D34's intra-turn claim suppresses duplicates *within* one turn. This
        exposes the cross-turn remainder: cars that departed earlier and are
        still in flight. Read-only; does not alter the mask.
        """
        out = np.zeros(self._num_zones, dtype=np.float32)
        if self._last_obs is None:
            return out
        counts = getattr(self, "_zone_inbound_reloc", None)
        if counts is not None:
            out[:] = counts
        return out

    def _legal_target_mask_flat(self) -> np.ndarray:
        """Zone adjacency indicator, the centralized stand-in for neighbor_mask.

        CAVEAT, deliberate: the actor's ``neighbor_mask`` is the ACTING zone's
        legal-target set, which depends on where the agent sits and on live fleet
        state. A single global vector cannot encode one mask per zone without an
        O(num_zones^2) tensor, and the critic is shared across agents, so the
        critic gets the static adjacency structure instead. That is strictly more
        information than before (previously absent entirely) and is the part of
        neighbor_mask that does not vary per acting zone.

        The acting zone's own full mask remains available to the actor, and
        train.py appends zone identity to this state, so the critic can at least
        resolve which zone it is scoring.
        """
        out = np.zeros(self._num_zones, dtype=np.float32)
        if self._last_obs is None:
            return out
        for z in range(self._num_zones):
            nbrs = self.env.region_neighbours[z] if z < len(self.env.region_neighbours) else ()
            for j in nbrs:
                if 0 <= j < self._num_zones:
                    out[j] = 1.0
        return out

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
