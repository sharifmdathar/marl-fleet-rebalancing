"""Central configuration for all experiment parameters.

Every numeric constant lives here — no magic numbers in env or training code.
NYC defaults are documented with their source/rationale.

Usage:
    from marl_fleet_rebalancing.config import NYCConfig
    cfg = NYCConfig()
    env = RideGymAEC(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    """Environment configuration shared across all experiments.
    
    Version: 1.2 (D15: proximity cutoff, collision masking, neighbor relocation,
                  volume-aware rewards, cancellation penalty, parameterized zone demand)
    """

    # --- Fleet ---
    num_drivers: int = 50
    driver_capacity: int = 4

    # --- Dispatch & Relocation policies (D14, D15) ---
    # When True, zone agents can opportunistically dispatch orders to empty
    # drivers in status RELOCATING in addition to IDLE drivers.
    allow_relocating_dispatch: bool = True
    # Maximum distance (km) from zone centroid for an order to be considered a
    # candidate for dispatch. Prevents cross-city deadheading (D15). None disables.
    max_dispatch_radius_km: float | None = 2.0
    # When True, relocation actions are restricted to adjacent zone neighbours (D15).
    neighbor_relocation_only: bool = True

    # --- Parameterized Zone Demand (D15) ---
    # Specific zone ID (0 .. num_zones - 1) to receive a dedicated share of demand.
    target_zone: int | None = None
    # Percentage (0.0 to 100.0) of total episode demand originating in target_zone.
    target_zone_demand_pct: float | None = None

    # --- Simulation ---
    dt: float = 0.1             # minutes per step
    horizon: float = 60.0       # episode length in minutes
    order_timeout: float = 2.0  # minutes before unassigned order is cancelled

    # --- Order generation ---
    num_orders: int = 800       # total orders to generate over the episode
    max_party_size: int = 3

    # --- Network ---
    network_path: str = "data/nyc/manhattan.gpickle"
    speed_kmh: float = 30.0

    # --- Observation budget ---
    # Maximum pending orders included in observation.
    # If actual pending > max_pending, we keep the most urgent orders
    # (closest to timeout) and mask the rest.
    # NYC Bandra West measured max ~70; set 80 as safe headroom.
    # For NYC, this should be measured from the actual order data
    # after running the data pipeline. Update once data is available.
    max_pending: int = 80

    # --- Region model (zone-controller agents, D10) ---
    # Synthetic maps: rows x cols grid of zone centres over the network bounds
    # (Bandra default 5x5 = 25 zones). NYC overrides this with real taxi-zone
    # centroids (nyc_zone_centroids_path).
    relocation_grid: tuple[int, int] = (5, 5)

    # --- Order source ---
    # "synthetic": OSMnxOrderGenerator over the network (num_orders, max_party_size)
    # "nyc":       NYC FHVHV splits via MultiWindowNYCOrderGenerator — one
    #              window per episode (train: random draw; val/test: cyclic);
    #              endpoints are zone centroids snapped to network nodes.
    #              See docs/nyc-pipeline.md.
    # "dataframe": raw DataFrameOrderGenerator — endpoints must already lie on
    #              the network (no snapping is performed).
    order_source: str = "synthetic"
    dataframe: Any = None                          # only for order_source="dataframe"
    dataframe_columns: dict[str, str] | None = None

    # --- NYC split parameters (order_source="nyc") ---
    nyc_split: str = "train"                       # "train" | "val" | "test"
    nyc_splits_dir: str = "data/nyc/splits"
    nyc_zone_centroids_path: str = "data/nyc/zone_centroids.csv"
    # Per-window cap on orders (sample-rate lever; None = full historical rate).
    # Open experiment-design decision (HANDOFF): full-rate peaks (~15k trips/hr)
    # are unservable by a 50-driver fleet. MUST be settled before training.
    nyc_window_limit: int | None = None
    nyc_random_party_size: bool = False  # FHVHV carries no real party size
    nyc_max_party_size: int = 3
    # Spatial jitter (km) of zone-centroid endpoints before snapping; mitigates
    # the centroid-collapse artefact. 0 keeps exact centroids.
    nyc_perturb_od_radius_km: float = 0.0

    # --- Reward normalization ---
    # Empty-distance normalizer in km: the beta term applies per
    # (empty km traveled / this value). Deadhead distance is taken exactly from
    # the RideGym step event log (D13).
    empty_distance_norm_km: float = 1.0

    # --- Reward weights ---
    # R = -alpha * waiting_time - beta * empty_distance + gamma * served_demand
    # Scaled so completing trips (+5.0) strongly outweighs empty distance (-0.1/km).
    alpha: float = 1.0   # weight for passenger waiting time
    beta: float = 0.1    # weight for empty repositioning distance (cost per empty km)
    gamma: float = 5.0   # weight for served demand (payoff per completed order)
    cancel_penalty: float = 5.0  # penalty per cancelled order in zone (D15)
    team_share: float = 0.2  # 80% local credit assignment + 20% team cooperation

    # --- Seeding ---
    seed: int | None = None


# ---------------------------------------------------------------------------
# NYC-specific preset
# ---------------------------------------------------------------------------

@dataclass
class NYCConfig(EnvConfig):
    """Configuration for NYC Manhattan study area.

    Zone count: ~62 taxi zones whose centroids fall inside Region A bbox
    (-74.02, 40.70, -73.93, 40.80); loaded from zone_centroids.csv and passed
    to RideGym as relocation_centroids (region model = real taxi zones).

    Fleet size: 50 drivers for a 60-minute episode. Scale as needed.

    max_pending: 650 — measured worst case of concurrent 2-min arrivals
    (train/window_0071, Fri 19:00 peak).

    Demand: real FHVHV windows from data/nyc/splits via
    MultiWindowNYCOrderGenerator (train: random window per episode;
    val/test: deterministic cyclic traversal). nyc_window_limit is the
    sample-rate lever — still an open experiment-design decision.

    Network: manhattan.gpickle downloaded via
    `python -m ride_gym.data_tools.nyc.build_nyc_network`
    """

    num_drivers: int = 50
    driver_capacity: int = 4
    dt: float = 0.1
    horizon: float = 60.0
    order_timeout: float = 2.0
    num_orders: int = 800  # unused in nyc mode; kept for the synthetic fallback
    max_party_size: int = 3
    order_source: str = "nyc"
    nyc_split: str = "train"
    network_path: str = "data/nyc/manhattan.gpickle"
    speed_kmh: float = 30.0
    max_pending: int = 650
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 5.0
    team_share: float = 0.2


# ---------------------------------------------------------------------------
# Bandra West preset (for quick local testing)
# ---------------------------------------------------------------------------

@dataclass
class BandraConfig(EnvConfig):
    """Configuration for Bandra West, Mumbai (quick iteration/testing).

    Zone count: 25 relocation zones from RideGym defaults.
    Fleet size: 30 drivers, 30-minute episode.
    num_orders: 200 (achievable demand profile for a 30-vehicle fleet).
    max_pending: 80 (measured max ~70 in Bandra West testing).
    """

    num_drivers: int = 30
    driver_capacity: int = 4
    dt: float = 0.1
    horizon: float = 30.0
    order_timeout: float = 2.0
    num_orders: int = 200
    max_party_size: int = 3
    network_path: str = "data/bandra_west.gpickle"
    speed_kmh: float = 30.0
    max_pending: int = 80
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 5.0
    team_share: float = 0.2
