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

    Version: 3.3 (D82: NYC relocator payment shared along the relocation chain)
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
    # D45: was True. Relocation was restricted to KNN neighbours, which was
    # NOT a fairness-neutral modelling choice: the LP baseline was bound by the
    # same mask (lp_baseline.py "valid neighboring zone j (respecting
    # action_mask)"), so neither method could send a car across the map in one
    # hop. Measured on NYC16, the two-phase competence probe (demand 100% at A,
    # moving to B 2 hops away): under neighbour-only, B was a legal target on
    # 0.5% of turns and only 28.3% of legal targets moved a car CLOSER to B.
    # The action space was always 1 + max_pending + num_zones, so this only
    # filtered the mask -- no obs/action-space change.
    #
    # The -beta*km term already prices distance (0.06/km, so a 2.5 km hop costs
    # 0.149 vs 0.03 for 0.5 km), which is the gradient that should make a
    # policy prefer near moves once long ones are permitted.
    #
    # NOT a fix for payoff timing: 61.9% of legs still never pay back and the
    # median payback is 21-40 turns. This makes the destination reachable; it
    # does not make relocating early rewarding. See D44.
    neighbor_relocation_only: bool = False

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
    # Minutes an unassigned order survives before it cancels (env 3.0, D35).
    # Was 2.0, chosen arbitrarily rather than from any operational figure, and
    # that value sat BELOW every anchor found: a relocation leg is ~2.5 km at
    # 30 km/h (~5 min travel), so 2.0 min of patience meant a repositioned car
    # routinely arrived after its rider had given up. Measured on NYC16 base
    # demand, 5 paired seeds: lifting the timeout to 5.0 raises service rate
    # under every policy arm tested (dispatch-only 46.7%, dispatch-first 65.4%
    # at 2.0 min). NYC anchors: platform average total wait 5.49 min (HVFHV,
    # 2026); TLC WAV benchmarks <10 min for 80% of trips; accept-to-board
    # pickup time ~303 s mean. 5.0 is the primary arm, with 2.0 retained as a
    # documented sensitivity arm. NOTE (inferred, not derived): those sources
    # measure TOTAL wait (accept + en route + board), whereas this field is
    # unassigned-order patience. They bound the plausible range; they do not
    # measure this quantity directly.
    order_timeout: float = 5.0

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
    # NYCConfig overrides this to 130: full-rate peaks (~15k trips/hr) are
    # unservable by a 50-driver fleet, which made unservable windows look like
    # policy failure.
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
    # Scaled cleanly so serving (+1.0) balances cancellation (-1.0) and empty distance (-0.06/km, D25).
    #
    # D62: restored to 0.2. D58 lowered this to 0.0132 on the measured
    # MAGNITUDES of the four reward terms -- waiting was 90% of all reward
    # movement -- and targeting collapsed from 55.06% to 20.95%. The mistake was
    # measuring how LARGE each term is without asking what it TEACHES.
    #
    # The waiting penalty is charged to the zone CONTAINING the waiting rider,
    # so it is the only term that distinguishes WHICH zone needs a car. Served
    # credit pays for any completion regardless of where the car came from, so
    # it cannot teach targeting at all. Cutting the waiting weight therefore
    # removed most of the pressure to aim at needy zones.
    #
    # Measured on NYC16 val, 3 paired seeds, argmax actions:
    #
    #     alpha    precision   chance    lift    SR%    empty km
    #     0.0132     18.73%    7.28%    2.57x   76.49      444.8
    #     0.10       28.53%    8.38%    3.41x   79.45      454.2
    #     0.20       31.36%    7.78%    4.03x   80.85      439.2
    #
    # Higher alpha improves targeting, service rate AND empty km together; only
    # average wait is slightly worse (2.89 vs 2.63 min). Restoring 0.2 is a
    # reversal of D58, recorded here rather than silently applied.
    alpha: float = 0.2   # weight for passenger waiting time (D25; D58 lowered it, D62 restored it)
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
    # Share of gamma paid to the zone that RELOCATED the driver which completed
    # this ride. Paid in full to that one zone, undivided (D47).
    #
    # Must exceed beta * km_per_relocation, or a relocation is net-negative even
    # when it works perfectly and the zone learns never to send. Measured at
    # 2.49 km per relocation and beta=0.15 that floor is 0.374; 0.35 sat just
    # under it, which is why the trained policy relocated constantly at ~25%
    # hit rate and still scored worse than doing almost nothing.
    #
    # At beta=0.06 and a RE-MEASURED mean leg of 2.95 km (367 legs, 3 episodes;
    # scripts/diag_reloc_leg_distance.py) the floor is 0.177, so 0.45 clears it
    # 2.54x. The 2.49 km figure quoted above is from an older run and is stale;
    # it made this margin read 3.0x. scripts/diag_reloc_invariant.py re-measures
    # rather than hardcoding, and accepts a distance on argv.
    #
    # Until D47 this share was DIVIDED by the length of a "relay chain", which
    # made the effective share fall to 0.150 at 3 hops -- a 1.004x margin over
    # cost, and net-negative from 4 hops on.
    relocator_share: float = 0.65
    # D66: pay the relocator only when the moved car was the marginal one (the
    # destination had no idle car left and riders still waiting). False = flat (D64).
    # D81: add to each zone's observation the cars already heading to every
    # zone (in flight, plus claimed earlier this turn), as a share of the fleet.
    obs_inbound: bool = False
    # D77: relocation is limited to the k nearest legal target zones. 0 = any zone.
    relocation_nearest_k: int = 0
    relocator_requires_shortage: bool = False
    # D82: True = the relocator payment is divided equally among every zone that
    # moved the car since its last dispatch. False = last mover takes all (D47).
    relocator_chain_share: bool = False
    # D71: True = the relocator's share is taken out of the ride's gamma (zone
    # rewards sum to the team objective). False = paid on top of it (D49).
    relocator_share_is_split: bool = False
    # D67: decision-time bonus for a relocation, scaled by the destination's unmet
    # demand relative to the best legal destination. 0 = off.
    targeting_bonus: float = 0.0
    targeting_bonus_centred: bool = True  # D73: subtract the mean legal target's demand

    # D64: 0.45 -> 0.65. D59 made the relocating zone pay for the WHOLE job it
    # caused, but did not raise what it earns, which broke the expected-value
    # invariant this comment asserts. Measured cost per relocation is
    # beta * (2.9 km leg + 1.46 km pickup deadhead) = 0.2616, and the credit is
    # CONDITIONAL -- it fires only when the relocated car is dispatched, which
    # happens 56.8% of the time even for a PERFECT aim. So:
    #
    #     EV = relocator_share * gamma * P(dispatch | relocated) > beta * km
    #
    # The old form compared share to cost directly (0.45 > 0.262, which held)
    # and so missed that only ~57% of relocations ever pay. Expected net was
    # NEGATIVE for every class of aim, including the best, so the policy was
    # correctly ambivalent about relocating.
    #
    #     share    credit    expected    net
    #     0.45     0.450      0.2556   -0.0060
    #     0.55     0.550      0.3124   +0.0508
    #     0.65     0.650      0.3692   +0.1076   <- 1.41x cost
    #     0.80     0.800      0.4544   +0.1928
    #
    # scripts/diag_reloc_invariant.py asserts the conditional form, not the
    # literal one.

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
    # D35 (SUPERSEDED by D36, kept so the reversal stays readable): a relocation
    # destination had to have at least one waiting rider. It was introduced to
    # stop RL relocating into empty zones, and it did raise precision -- but it
    # also deleted the decision before the agent saw it (see D36). Set True only
    # to reproduce pre-D36 numbers.
    require_destination_demand: bool = False

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
    order_timeout: float = 5.0  # env 3.0 (D35); see EnvConfig for the rationale
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
    alpha: float = 0.2   # D62; see EnvConfig for the restore rationale
    beta: float = 0.06
    gamma: float = 1.0
    cancel_penalty: float = 1.0
    team_share: float = 0.2
    relocator_share: float = 0.65
    relocator_requires_shortage: bool = False  # D66: reverted to flat D64 credit after negative 5-seed result
    relocator_share_is_split: bool = True  # D71: zone rewards sum to the team objective
    relocator_chain_share: bool = True  # D82: no zone is left with cost and no credit

    # D64: 0.45 -> 0.65. D59 made the relocating zone pay for the WHOLE job it
    # caused, but did not raise what it earns, which broke the expected-value
    # invariant this comment asserts. Measured cost per relocation is
    # beta * (2.9 km leg + 1.46 km pickup deadhead) = 0.2616, and the credit is
    # CONDITIONAL -- it fires only when the relocated car is dispatched, which
    # happens 56.8% of the time even for a PERFECT aim. So:
    #
    #     EV = relocator_share * gamma * P(dispatch | relocated) > beta * km
    #
    # The old form compared share to cost directly (0.45 > 0.262, which held)
    # and so missed that only ~57% of relocations ever pay. Expected net was
    # NEGATIVE for every class of aim, including the best, so the policy was
    # correctly ambivalent about relocating.
    #
    #     share    credit    expected    net
    #     0.45     0.450      0.2556   -0.0060
    #     0.55     0.550      0.3124   +0.0508
    #     0.65     0.650      0.3692   +0.1076   <- 1.41x cost
    #     0.80     0.800      0.4544   +0.1928
    #
    # scripts/diag_reloc_invariant.py asserts the conditional form, not the
    # literal one.
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
    order_timeout: float = 5.0  # env 3.0 (D35); see EnvConfig for the rationale
    num_orders: int = 200
    max_party_size: int = 3
    network_path: str = "data/synthetic_grid_5x5.gpickle"
    speed_kmh: float = 30.0
    max_pending: int = 80
    alpha: float = 0.1
    beta: float = 0.1
    gamma: float = 5.0
    cancel_penalty: float = 0.5
    team_share: float = 0.2
