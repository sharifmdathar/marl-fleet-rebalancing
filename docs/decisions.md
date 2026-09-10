# Decision log (ADR-lite)

Format: date, status, decision, why, what would change our mind.

## D1 — Simulator: RideGym
- Date: 2026-09-03, Status: accepted
- Decision: Use RideGym instead of building a custom zone-based env.
- Why: swappable order generators (injection point for demand shift), real
  OSMnx road networks, existing benchmark harness with model-based baselines,
  seeded eval. Saves weeks vs building from scratch.
- Known risks: dispatch/pooling-centric (conflicting bids raise ConflictError);
  repo is young (created 2026-03), paper "in preparation"; fully-centralized
  action semantics need adaptation for per-vehicle MARL.
- Revisit if: we need zone-level rebalancing semantics RideGym can't express,
  or the ConflictError coordination overhead dominates.

## D2 — Study area: Bandra West, Mumbai (OSMnx graph)
- Date: 2026-09-03, Status: accepted (provisional)
- Why: real network, manageable size for fast iteration.
- Revisit if: demand realism matters more than iteration speed → switch to
  NYC TLC windows (RideGym's MultiWindowNYCOrderGenerator) for the paper.

## D3 — Non-RL baseline: rolling-horizon imbalance/LP rebalancer (MIVR-lite)
- Date: 2026-09-03, Status: accepted
- What: forecast per-zone demand, solve small transportation LP each interval,
  send idle vehicles to largest predicted deficits. Tunable knobs: lookahead
  horizon, forecast source, rebalancing frequency.
- Why: classical, strong, directly comparable; follows the RLSCALE-BENCH
  lesson (tuned baseline or the comparison is meaningless).
- Alternatives considered: nearest-idle (too weak — that's the floor),
  full MIVR with robust optimization (too much implementation for a baseline).

## D4 — Algorithms: IPPO vs MAPPO, PPO family only
- Date: 2026-09-03, Status: accepted
- Why: cleanest centralized-vs-independent contrast with shared PPO backbone;
  both have standard implementations (on-policy repo / MARLlib / EPyMARL).
- Explicitly out of scope: MADDPG, QMIX, novel algorithms.

## D5 — Dispatch vs rebalancing split
- Date: 2026-09-03, Status: accepted
- What: order dispatch handled by a fixed rule (e.g., nearest/hungarian);
  agents learn ONLY the `relocate` action.
- Why: isolates rebalancing (our research question) from dispatch; avoids
  ConflictError coordination problem in multi-agent bidding.
- Revisit if: advisor wants joint dispatch+rebalancing.

## D6 — Study area switched to NYC Manhattan
- Date: 2026-09-05, Status: accepted
- Decision: Primary study area is now NYC Manhattan (not Bandra West).
- Why: NYC is the standard benchmark in ride-hailing research. Real TLC trip data
  provides realistic demand patterns. Bandra code remains for quick local testing.
- Known: ~60 taxi zones in Region A bbox. max_pending=80 carried from Bandra;
  MUST be updated after measuring actual NYC peak pending counts.
- Revisit if: NYC data pipeline too slow for iteration → fall back to Bandra for
  development, NYC for final experiments.

## D7 — Config-driven parameters (no magic numbers)
- Date: 2026-09-05, Status: accepted
- Decision: All experiment parameters live in EnvConfig dataclass (config.py).
  No hardcoded constants in env or training code.
- Why: makes experiments reproducible, configs swappable (NYCConfig, BandraConfig),
  and prevents magic numbers from creeping in.
- Revisit if: config overhead too heavy for quick scripts.

## D8 — Reward function: R = -alpha*wait - beta*empty + gamma*served
- Date: 2026-09-05, Status: accepted
- Decision: Reward is the weighted sum from the project doc. Weights (alpha, beta,
  gamma) are config parameters, tuned as part of the experiment.
- Why: matches the advisor-provided formula. Equal weights as starting point.

## D9 — Joint dispatch+rebalancing action space (supersedes D5)
- Date: 2026-09-06, Status: accepted (implemented in PettingZoo wrapper)
- Decision: agents learn BOTH dispatch (bid on pending orders) and relocate,
  via the wrapper's `Discrete(1 + max_pending + num_zones)` action space.
  This supersedes D5's relocate-only split. Conflicting bids are resolved by
  the wrapper's shuffled first-come-first-served rule (seeded RNG), which is
  what made joint learning tractable — see `docs/pettingzoo-design-decisions.md`.
- Why: joint dispatch+rebalancing matches the deployment reality the study
  claims to measure; the wrapper's conflict resolution removes D5's original
  ConflictError blocker.
- CONFIRMED 2026-09-07 (per PPT + prof chats): action space is
  joint dispatch+rebalancing — fleet rebalancing covers both relocation of
  empty cars to better zones and general fleet rebalancing. The brief's
  "stay/reposition" phrasing describes the rebalancing half, not a
  relocate-only restriction.
- Revisit if: prof overrides in person.
- Revisit if: advisor prefers the D5 isolate-rebalancing design → restrict the
  wrapper action space to relocate-only and re-run.

## D10 — Agent granularity: zone-controllers (accepted)
- Date: 2026-09-07, Status: accepted (per initial PPT: central
  control, not per-car agents — project scope is zone controllers)
- Decision: agents are zone controllers (one per zone), not per-vehicle
  agents. Combined with D9: each zone-controller dispatches + rebalances
  within its scope under central control.
- Consequence: the current wrapper (`pettingzoo_env.py`) is built as
  vehicle-agents and MUST be reworked to zone-controller granularity before
  IPPO/MAPPO training. See HANDOFF Phase 2.
- Revisit if: prof overrides in person.

## D11 — Platform: RideGym confirmed, SUMO avoided (accepted)
- Date: 2026-09-07, Status: accepted
- Decision: no SUMO/MATSim. Prof feedback: avoid them initially (complexity
  not needed). RideGym tested (~60 LOC), works, supports OSM directly.
- A custom zone-based PettingZoo/Gymnasium env (9–20 zones) remains an option
  if iteration speed demands it — see D12 scale note.
- Revisit if: experiments run too slowly on realistic networks, or batches
  require shared SUMO-based evaluation.

## D12 — Advisor feedback round: RQs, datasets, shifts, scale (accepted)
- Date: 2026-09-07, Status: accepted (prof feedback via team summary)
- RQs reframed as RQ1 (how much degradation) + RQ2 (do IPPO/MAPPO differ) —
  reflected in `INTENT.md`. No-steer rule (H1–H4 all publishable) already
  satisfied it; the prof's MAPPO caution confirms rather than changes it.
- Datasets: NYC TLC FHVHV is the main data (columns: dispatching_base_num,
  pickup/dropoff_datetime, PULocationID, DOLocationID, SR_Flag). Full NYC
  dataset confirmed (2026-09-07) — no downscoping; the ~62 data zones
  stand. India Ola/Uber data DROPPED (too hard/brittle for the paper).
- Shift families named by prof (spatial / temporal / higher-demand / event)
  map onto S1–S5 — see `docs/demand-shift.md` mapping note.
- Scale steer: prof prefers simple zone maps (9–20 zones) and fast
  experiments, but full-NYC scope confirmed (2026-09-07), so the
  lever is sample-rate/fleet sizing, not zone count. Peak windows (~15k/hr
  vs 50 drivers) remain unservable at full rate — that sizing decision is
  still open and must precede training (spaces freeze).
- Shifts: keep all 7 scenarios ("if AI suggests then be it"). Reward: equal
  weights stand ("let's see" — revisit if tuning stalls). Baseline: LP +
  greedy plan stands; "gotta understand the baseline" is ongoing work, not
  a design change.
- Gap method: "couldn't find it" ≠ gap; establish (A) IPPO/MAPPO
  generalization + (B) demand-robustness in rebalancing independently —
  see `docs/literature.md`.
- Revisit if: advisor revises any of the above.

## D13 — Zone-controller wrapper implementation (D10 rework)
- Date: 2026-09-08, Status: accepted (Phase 2 step 0 complete)
- Decision: `pettingzoo_env.py` reworked from vehicle-agents to zone-controller
  agents per D10. Implementation contract:
  - **Agents**: one per region — `zone_i` (62 NYC taxi zones / 25 Bandra grid
    cells). AEC turn = every zone agent acts once (shuffled FCFS order, seeded
    RNG, unchanged), then one RideGym `env.step` of `dt`.
  - **Action space** (D9, unchanged): `Discrete(1 + max_pending + num_zones)` —
    idle; dispatch pending order i to an idle driver IN THIS ZONE; relocate one
    idle driver of THIS zone to another zone.
  - **Zone→driver resolution**: per-zone FIFO queues of idle drivers, rebuilt
    from the RideGym obs each turn; agents consume their zone's queue in
    shuffled order. One driver serves one action per turn; drivers never
    selected take no action (keep state). Dispatch claims the order globally
    (FCFS); same-zone relocation is masked out (no-op).
  - **Observation** (zone-level): `zone_location` (2), `zone_state` (3: fleet
    fraction, idle fraction, pending-originating fraction), `all_zone_state`
    (Z×2 fleet+idle fractions), `pending_orders` (max_pending×6: ox, oy, dx,
    dy, pax/capacity, wait/timeout — sorted most-urgent-first), `pending_mask`,
    `pending_passengers`, `time_frac`, `relocation_points`. Zone identity via
    `zone_location` so policies share parameters across agents.
  - **Region model fix**: all region centres now live in the network's real
    coordinate space. Synthetic maps get a `relocation_grid` (config, 5×5)
    grid over the NETWORK BOUNDS instead of RideGym's abstract-area default,
    whose targets previously snapped to arbitrary graph nodes. NYC gets the
    ~62 taxi-zone centroids (zone_centroids.csv, filtered to network bounds,
    sorted by LocationID). Env `area` = network bounds.
  - **Empty-distance term of the D8 reward is now exact**: deadhead metres
    (moved while empty) are summed from the RideGym step event log
    (`is_empty_move`/`distance_moved`), replacing the status proxy that was
    silently always 0 (corrupted key). Normalized to km via
    `empty_distance_norm_km` (config, default 1.0). Formula and weights (D8)
    unchanged.
  - **NYC order source**: `order_source="nyc"` builds
    `MultiWindowNYCOrderGenerator` from `data/nyc/splits/manifest.json`
    (train: random window/episode; val/test: deterministic cyclic), with
    `nyc_window_limit` (per-window order cap — the sample-rate lever, still an
    open experiment-design decision), `nyc_random_party_size`,
    `nyc_perturb_od_radius_km` as config knobs. `order_source="dataframe"`
    remains a raw passthrough (endpoints must be on-network).
- Why: D10 changed agent granularity under the vehicle-agent API; zone-level
  obs/actions keep the learning problem well-posed (fleet aggregates instead
  of per-car states) while reusing the proven AEC turn/conflict machinery.
- Smoke-tested: Bandra (25 zone agents, 300 turns) and NYC train split
  (62 zone agents, 600 turns, real FHVHV window) — see
  `marl_fleet_rebalancing.env.test_wrapper`.
- Revisit if: advisor wants zone controllers to issue multi-driver actions per
  turn (current: one action per zone per turn) or per-driver sub-queues.

## D14 — Opportunistic dispatch of relocating drivers
- Date: 2026-09-10, Status: accepted
- Decision:
  - Empty drivers in transit (`DriverStatus.RELOCATING`) are now eligible to be
    dispatched to pending passenger orders (`"orders"` action) by the zone
    controller where the driver is currently located.
  - In `_flush_turn`, purely `IDLE` drivers are prioritized first in the zone's
    dispatch queue; `RELOCATING` drivers serve as opportunistic fallbacks.
  - When a relocating driver is dispatched, RideGym natively cancels its
    relocation target (`driver.relocation_target = None`) and redirects it to
    passenger pickup.
  - Relocation actions (`"relocate"`) still strictly consume purely `IDLE`
    vehicles, preventing erratic redirection loops of vehicles in flight.
  - Added `allow_relocating_dispatch: bool = True` to `EnvConfig` (config v1.1).
- Why:
  - Without this, an empty rebalancing vehicle passing right next to a newly
    spawned passenger order was forbidden from serving it because its status
    was `RELOCATING` rather than `IDLE`. This locked the local zone out of
    dispatching, allowing distant idle vehicles from across the city to claim
    the order instead, producing severe deadhead detours and delayed pickups.
- Revisit if:
  - Advisor prefers rebalancing legs to be strictly uninterruptible commitments.

## D15 — Fleet rebalancing pipeline fixes & parameterized zone demand injection
- Date: 2026-09-10, Status: accepted
- Decision:
  - Bumped config version to `v1.2` (`EnvConfig`).
  - Added local dispatch radius cutoff (`max_dispatch_radius_km = 2.0` km): candidates beyond 2 km from zone centroid are excluded from pending order queues in `_rebuild_caches`, preventing cross-city deadheading assignments.
  - Added intra-turn collision masking: orders claimed by prior agents within the current simulation turn (`_claimed_orders_this_turn`) are dynamically masked out (`False`) in `action_mask()` for subsequent agents.
  - Added neighbor-only relocation pruning (`neighbor_relocation_only = True`): idle vehicle relocation actions are restricted to adjacent graph regions (`region_neighbours`), pruning 96% irrelevant action exploration.
  - Replaced volume-blind waiting penalty with volume-aware wait penalty (`alpha * total_waiting_seconds / (max_pending * 60)`), scaling linearly with passenger backlog volume.
  - Added explicit cancellation penalty (`cancel_penalty = 5.0` per cancelled order) charged directly to the origin zone controller upon order expiration.
  - Agent-conditioned MAPPO Centralized Critic: concatenated the acting agent's normalized zone centroid (`[state, zone_location]`) into the centralized critic input, giving the shared critic spatial identity to distinguish 25 distinct local return trajectories.
  - Added parameterized zone demand injection (`ParameterizedZoneOrderGenerator`): allows specifying `--target-zone <int>` and `--target-zone-demand-pct <float>` across training and evaluation CLI to inject arbitrary source demand fractions into designated zones at random intervals.
- Why:
  - Fixes the 6 critical operational and reinforcement learning flaws identified in the rebalancing pipeline, and satisfies the user requirement for targeted zone demand distribution control.
- Revisit if:
  - Regional geometries require adaptive radius thresholds or dynamic zone neighborhoods.

## D16 — Macro-demand observation & cooperative rebalancing credit assignment
- Date: 2026-09-11, Status: accepted
- Decision:
  - **Macro-Demand Observation**: Expanded `all_zone_state` in the zone-controller observation space from `(Z, 2)` to `(Z, 3)`. The 3rd column contains each zone's current fraction of citywide pending demand (`self._pending_per_zone`). Every zone agent now sees both the fleet distribution and the unmet demand heatmap across all zones, enabling perception of distribution shifts and distant surges.
  - **Rebalancing Assist Credit**: When a vehicle relocated by Zone A completes an order in Zone B, the completion reward (+gamma) is shared: 70% to the local dispatcher (Zone B) and 30% as a rebalancing assist to the zone that supplied the vehicle (Zone A).
  - **Shared Relocation Deadhead Penalty**: Empty travel during inter-zone relocation is now split 50/50 between the sending zone and the receiving zone, eliminating the altruism penalty where sending zones bore 100% of fuel costs alone.
  - **Increased Team Share**: Set default `team_share: float = 0.4` (from 0.2), creating strong cooperative alignment across all agents while preserving local credit assignment.
- Why:
  - Resolves the sensory blind spot where agents could not detect demand outside their 1.2 km dispatch radius.
  - Fixes the credit assignment free-rider defect where rebalancing agents were strictly penalized for sending vehicles to assist neighboring deficit zones.
- Revisit if:
  - Assist split percentages (70/30) need tuning under different fleet-to-demand ratios.

## Template for new entries
## D<n> — <title>
- Date:   Status: proposed | accepted | superseded by D<m>
- Decision / Why / Revisit if:
