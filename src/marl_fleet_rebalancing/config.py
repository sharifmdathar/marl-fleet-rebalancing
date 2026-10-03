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

    Version: 3.0 (relay-chain credit: every contributing zone is paid)
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
    # When True, relocation actions are restricted to adjacent zone neighbours (D15, D26).
    # Creates smooth local spatial diffusion without long-distance highway locks.
    neighbor_relocation_only: bool = True

    # --- Parameterized Zone Demand (D15) ---
    # Specific zone ID (0 .. num_zones - 1) to receive a dedicated share of demand.
    target_zone: int | None = None
    # Percentage (0.0 to 100.0) of total episode demand originating in target_zone.
    target_zone_demand_pct: float | None = None

    # --- Demand-shift mechanisms (D21; eval-side scenario knobs) ---
    # Exactly one of "none" | "burst" | "permute" applies per scenario;
    # "concentrate" is expressed with target_zone/target_zone_demand_pct
    # above (S4 cold zones). All preserve order count and timestamps.
    demand_shift_mode: str = "none"
    # S5 event burst: share (0..100) of orders with request_time in
    # [burst_start_min, burst_end_min) reassigned to burst_zone.
    burst_zone: int | None = None
    burst_start_min: float = 20.0
    burst_end_min: float = 30.0
    burst_share_pct: float | None = None
    # S1 spatial permutation: per-order remap probability (0..1) through a
    # fixed rotation of offset perm_offset zones (None = num_zones // 2).
    perm_strength: float | None = None
    # --- Two-zone hotspot (fixed concentration, no drift) ---
    # hotspot_pct of orders are redirected into hotspot_zone_a and
    # hotspot_zone_b and stay there for the whole episode. hotspot_split
    # divides that share between them (0.5 = even). Order count and timestamps
    # are preserved.
    hotspot_zone_a: int = 8
    hotspot_zone_b: int = 13
    hotspot_pct: float = 60.0
    hotspot_split: float = 0.5
    # --- Sequential two-stage surge ---
    # surge_pct of orders concentrate into surge_zone1 for the first stage and
    # into surge_zone2 for the second, handing over linearly between
    # surge_stage1_end_min and surge_stage2_start_min. Unlike the drift
    # scenario this CREATES a surge at the first hub rather than only moving
    # demand that was already there.
    surge_zone1: int = 10
    surge_zone2: int = 12
    surge_pct: float = 60.0
    surge_stage1_end_min: float = 18.0
    surge_stage2_start_min: float = 32.0
    # --- Local hub (all demand at exactly one of two zones) ---
    # Every order originates at localhub_stage1_zone before localhub_switch_min
    # and at localhub_stage2_zone after it; no order originates anywhere else.
    # Destinations are restricted to zones adjacent to the active hub, so each
    # trip is a local hop. Reuses the surge zone pair by default.
    localhub_stage1_zone: int = 10
    localhub_stage2_zone: int = 12
    localhub_switch_min: float = 25.0
    # --- Drifting hotspot (demand migrates between two distant hubs) ---
    # Before drift_ramp_start_min, drift_source_pct of orders originate at
    # drift_source_zone. That share falls to 0 by drift_ramp_end_min, so the
    # late episode has all of its demand at drift_dest_zone. Timestamps and
    # order counts are preserved.
    # Source A=12 has 8 neighbours and sits 9.05 km from D=10, which is 4.26 km
    # from A's nearest neighbour: the new hub is across the map, not one hop.
    drift_source_zone: int = 12
    drift_dest_zone: int = 10
    drift_ramp_start_min: float = 20.0
    drift_ramp_end_min: float = 35.0
    drift_source_pct: float = 0.6
    perm_offset: int | None = None

    # --- Simulation ---
    dt: float = 0.5             # minutes per step (30 sim-seconds, D23)
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
    # Bandra West measured max ~70; NYC measured peak ~28 concurrent (D28).
    # NYCConfig overrides to 30; BandraConfig keeps 80.
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
    # R = -alpha * waiting_time - beta * empty_distance + gamma * served_demand - cancel_penalty
    # Scaled cleanly so serving (+1.0) balances cancellation (-1.0) and empty distance (-0.15/km, D25).
    alpha: float = 0.2   # weight for passenger waiting time (D25)
    # Cost per empty repositioning km. At 0.15 with ~2.5 km per relocation a
    # single move cost 0.374 against a completion worth gamma=1.0, so distance
    # was heavy enough to cancel the entire served-demand credit: the trained
    # policy netted -0.9 reward per episode against LP's +49.6, while buying
    # its service rate with 2.6x the deadhead. 0.06 keeps deadhead a real
    # penalty that discriminates good moves from bad, without dominating the
    # service term it is meant to modulate.
    beta: float = 0.06
    gamma: float = 1.0   # weight for served demand (payoff per COMPLETED order)
    cancel_penalty: float = 1.0  # penalty per cancelled order in zone (D25)
    team_share: float = 0.2  # 80% local credit + 20% team
    # Share of gamma paid to the zone that SUPPLIED a relocated driver when
    # that driver's ride completes.
    #
    # Must exceed beta * km_per_relocation, or a relocation is net-negative even
    # when it works perfectly and the zone learns never to send. Measured at
    # 2.49 km per relocation and beta=0.15 that floor is 0.374; 0.35 sat just
    # under it, which is why the trained policy relocated constantly at ~25%
    # hit rate and still scored worse than doing almost nothing.
    relocator_share: float = 0.45

    # --- Relocation arrival bonus ---
    # Paid to the sending zone when its relocated driver arrives, scaled by the
    # destination's pending demand. OFF by default: it paid for the arrival
    # itself rather than for the car being useful, which rewarded moving cars
    # for no reason. Leave at 0.0; raise only as a small bridge if the
    # completion credit proves too slow to learn through.
    prospective_bonus_scale: float = 0.0
    demand_history_len: int = 4
    # D34: cap relocation arrivals per zone per turn at that zone's pending
    # demand. Zones act on the same pre-turn snapshot, so without a claim every
    # zone picks the same destination. True = demand-based claim, False = legacy.
    claim_relocation_targets: bool = True
    # D35: a relocation destination must have at least one waiting rider.
    # Measured on the 16-zone build, the previous rule offered 5.6 targets per
    # turn of which 69% had nobody waiting, and in 16% of turns even the
    # highest-demand option was demand-free — a ceiling on relocation precision
    # that no reward shaping could lift. Requiring demand cuts the menu to
    # 1.7 targets per turn, all of them useful, and leaves 16% of turns with no
    # legal move (the zone correctly does nothing). False restores the old rule.
    require_destination_demand: bool = True

    # --- Seeding ---
    seed: int | None = None

# ---------------------------------------------------------------------------
# NYC-specific preset
# ---------------------------------------------------------------------------

@dataclass
class NYCConfig(EnvConfig):
    """NYC Manhattan study area: 16 busiest taxi zones, real FHVHV demand.

    Zone count is fixed by the dataset manifest (data/nyc16/splits/manifest.json
    lists the LocationIDs), not by this file — rebuild it with
    `scripts/build_zone_subset.py --num-zones 16`. 16 zones is ~4x cheaper than
    the full 62 and sits inside the brief's scale guidance.

    Fleet and order caps preserve the pressure ratios of the 62-zone build so
    numbers stay comparable: 100/62 = 1.61 drivers per zone -> 26 drivers;
    500/100 = 5 orders per driver -> 130 orders per window. LP reached ~72%
    service rate under the 62-zone ratios, which is the range to expect here.

    Network: manhattan.gpickle (unchanged; only the zone subset and demand
    filtering change).
    """

    num_drivers: int = 26
    driver_capacity: int = 4
    dt: float = 0.5
    horizon: float = 60.0
    order_timeout: float = 2.0
    num_orders: int = 800  # unused in nyc mode; kept for the synthetic fallback
    max_party_size: int = 3
    order_source: str = "nyc"
    nyc_split: str = "train"
    nyc_splits_dir: str = "data/nyc16/splits"
    network_path: str = "data/nyc/manhattan.gpickle"
    speed_kmh: float = 30.0
    nyc_window_limit: int | None = 130
    # Observed peak concurrent pending over 5 episodes is 15; 16 leaves one
    # slot of headroom. Sizing this to the data keeps the action space tight
    # (1 + 16 + 16 = 33) instead of wasting masked slots on empty entries.
    max_pending: int = 16
    alpha: float = 0.2
    beta: float = 0.06
    gamma: float = 1.0
    cancel_penalty: float = 1.0
    team_share: float = 0.2
    relocator_share: float = 0.45
    prospective_bonus_scale: float = 0.0

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
    dt: float = 0.5
    horizon: float = 30.0
    order_timeout: float = 2.0
    num_orders: int = 200
    max_party_size: int = 3
    network_path: str = "data/bandra_west.gpickle"
    speed_kmh: float = 30.0
    max_pending: int = 80
    alpha: float = 0.1
    beta: float = 0.1
    gamma: float = 5.0
    cancel_penalty: float = 0.5
    team_share: float = 0.2
