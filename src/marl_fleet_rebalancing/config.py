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


@dataclass
class EnvConfig:
    """Environment configuration shared across all experiments."""

    # --- Fleet ---
    num_drivers: int = 50
    driver_capacity: int = 4

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

    # --- Reward weights ---
    # R = -alpha * waiting_time - beta * empty_distance + gamma * served_demand
    # Tuning these is part of the experiment; start with equal weights.
    alpha: float = 1.0   # weight for passenger waiting time
    beta: float = 1.0    # weight for empty repositioning distance
    gamma: float = 1.0   # weight for served demand

    # --- Seeding ---
    seed: int | None = None


# ---------------------------------------------------------------------------
# NYC-specific preset
# ---------------------------------------------------------------------------

@dataclass
class NYCConfig(EnvConfig):
    """Configuration for NYC Manhattan study area.

    Zone count: ~60 taxi zones whose centroids fall inside Region A bbox
    (-74.02, 40.70, -73.93, 40.80). Exact count depends on data pipeline.

    Fleet size: 50 drivers for a 60-minute episode. Scale as needed.

    max_pending: 80 is carried over from Bandra West measurements.
    MUST be updated after running the NYC data pipeline and measuring
    actual peak pending order counts.

    Network: manhattan.gpickle downloaded via
    `python -m ride_gym.data_tools.nyc.build_nyc_network`
    """

    num_drivers: int = 50
    driver_capacity: int = 4
    dt: float = 0.1
    horizon: float = 60.0
    order_timeout: float = 2.0
    num_orders: int = 800
    max_party_size: int = 3
    network_path: str = "data/nyc/manhattan.gpickle"
    speed_kmh: float = 30.0
    # Measured 2026-09-07 from data/nyc/splits (108 windows): worst case is
    # 604 orders arriving within one 2-min order_timeout span
    # (train/window_0071, Fri 19:00 peak). 650 = measured max + headroom.
    # NOTE: peak windows carry ~15k trips/hr against a 50-driver fleet —
    # full-rate episodes are unservable; expect sample-rate downsampling
    # (or a larger fleet) as an experiment-design decision before training.
    max_pending: int = 650
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


# ---------------------------------------------------------------------------
# Bandra West preset (for quick local testing)
# ---------------------------------------------------------------------------

@dataclass
class BandraConfig(EnvConfig):
    """Configuration for Bandra West, Mumbai (quick iteration/testing).

    Zone count: 25 relocation zones from RideGym defaults.
    Fleet size: 30 drivers, 30-minute episode.
    max_pending: 80 (measured max ~70 in Bandra West testing).
    """

    num_drivers: int = 30
    driver_capacity: int = 4
    dt: float = 0.1
    horizon: float = 30.0
    order_timeout: float = 2.0
    num_orders: int = 400
    max_party_size: int = 3
    network_path: str = "data/bandra_west.gpickle"
    speed_kmh: float = 30.0
    max_pending: int = 80
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
