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

## D21 — Demand-shift mechanisms fix: S1/S4/S5 now match the taxonomy
- Date: 2026-09-15, Status: accepted
- Decision: close the implementation gap where S1, S4, and S5 were all
  expressed as the same static `(target_zone, pct)` concentration, despite the
  taxonomy promising three distinct mechanisms (spatial permutation, unseen-hotspot
  emergence, time-localized event burst).
- Why now: the degradation plots made S1/S4/S5 look like a single dose-response
  curve over concentration percent — the honest reading is that the results never
  tested what the taxonomy claimed.
- What changed:
  - `src/marl_fleet_rebalancing/demand/generators.py`:
    `BurstOrderGenerator` (S5) and `PermutedZoneOrderGenerator` (S1) added;
    both wrap the base generator, preserve order count and request timestamps,
    and implement the RideGym (`generate`/``reseed`) interface.
  - `src/marl_fleet_rebalancing/env/pettingzoo_env.py`:
    `_make_order_generator` routes via `demand_shift_mode` (`"none"`,
    `"burst"`, `"permute"`); `"concentrate"` stays target_zone-driven.
  - `src/marl_fleet_rebalancing/config.py`: `EnvConfig` now carries
    `demand_shift_mode`, `burst_zone`, `burst_start_min`, `burst_end_min`,
    `burst_share_pct`, `perm_strength`, `perm_offset`; config version bumped
    to 1.3.
  - `src/marl_fleet_rebalancing/experiments/shift_eval.py`:
    `SHIFTS` now has both mechanism-labeled entries (`S1`, `S4`, `S5`) and
    deprecated backward-compat entries (`S1a`, `S1b`, `S4a`, `S5a`) that keep
    the old CSV names on disk.
  - `docs/demand-shift.md`: mechanism section rewritten; deprecated labels
    documented with the mapping to old CSVs.
- Verification (direct exec, NYC train split, winlim 500):
  - `BurstOrderGenerator`: in-window zone-0 fraction 0.000 → 0.493 at
    `burst_share=50`, outside-window fraction unchanged (0.0117 → 0.0117),
    timestamps and total order count identical.
  - `PermutedZoneOrderGenerator` at strength 1.0: rotated histogram matches the
    source histogram rotated by offset exactly (0 max abs diff on 500 orders);
    strength 0.0 is identity.
- Cold-zone decision for S4: measured train-split demand by zone (winlim 500,
  84 windows, 42000 sampled orders) → zone 1 = 35 orders (0.08%), zone 3 = 56,
  zone 40 = 59; hottest zones ~1400 each (~3.3%). S4 uses zone 1.
- Reporting: new mechanism-labeled results go to `results/shift_eval_fixed/`;
  old `results/shift_eval/` CSVs (S1a/S1b/S4a/S5) are NOT overwritten and stay
  interpretable as "static concentration" runs. Any figure/figure caption that
  uses S1/S4/S5 must say which directory it draws from.
- Revisit if:
  - Advisor wants the temporal-vs-spatial contrast formally tested as distinct
    from the concentration dose-response.
  - We add more hypothetical unseen zones and want a zone-selection rule beyond
    "coldest in train split".
  - We want a magnitude axis for S5 (currently fixed at 20% share) or a
    strength axis for S1 (currently fixed at 1.0 and 0.5).

## D22 — Servable-S0 env defaults (code audit, no training logs)
- Date: 2026-09-16, Status: accepted (researcher directive; advisor review welcome)
- Decision: config v1.3 → v1.4. Six changes so IPPO/MAPPO can learn
  dispatch+relocate on S0 before any shift comparison (RQ1 prerequisite):
  1. `NYCConfig.nyc_window_limit` None → 500: full 5–8k/hr windows are
     unservable by 50 drivers; most cancels were capacity-driven noise.
     500/ep matches fixed-fleet training + D21 verification.
  2. Per-zone dispatch candidates closest-first → most-urgent-first
     (`lexsort((d, -waits))`): action k pointed at a different order each
     turn as distances shifted; now matches the global pending view.
  3. `pending_per_zone` divisor max_pending → num_drivers: /650 crushed the
     demand signal to ~0.001/order; /50 reads ~0.02 with saturation at 1.0.
  4. `max_dispatch_radius_km` 1.2 → 2.0; `neighbor_relocation_only`
     True → False (+ `--relocation-scope anywhere|neighbor` CLI):
     neighbour-only forced multi-hop repositioning while orders timed out
     in 20 turns.
  5. Reward legibility: hidden extra ×0.1 on the wait term removed and
     folded into `alpha` 1.0 → 0.1 (magnitude-neutral); `cancel_penalty`
     2.0 → 0.5 (capacity cancels aren't policy faults); `team_share`
     0.4 → 0.2 (restores D8 local credit). Cancel/team values are judgment
     calls — retune on train/val only.
  6. `NYCConfig.max_pending` 650 → 80: winlim-500 no-op peaks at 28
     concurrent, so 80 is headroom; obs 5516 → ~950, state 4737 → ~750,
     action 713 → 143. `rollout_episodes` stays 6.
  7. Bugfix: `_make_env` passed `nyc_window_limit=None` unconditionally,
     clobbering the config default; now only overrides when the flag is set.
- Why: audited from code + live probes only (no training logs per
  researcher instruction); each item blocks learning independently.
- Revisit if: S0 heuristic SR misses ~60-70% (resize winlim/fleet first);
  urgent-first underperforms closest-first in an A/B; cancel/team need
  retuning once agents learn something.
- Do NOT mix pre/post-D22 seeds in one aggregate (env version change).

## D23 — Purge all pre-D22 artifacts (2026-09-16)
- Date: 2026-09-16, Status: accepted (researcher directive)
- Decision: deleted all contents of `runs/` (checkpoints, metrics) and
  `results/` (evals, shift CSVs, figures, videos). Storage statements in
  D20/D21 ("goes to `results/shift_eval_fixed/`", "old CSVs stay on disk")
  are superseded — those directories are empty.
- `docs/shift-results.md` is a stub (protocol + checklist, no numbers);
  `docs/demand-shift.md` keeps taxonomy/mechanisms/definitions only;
  `docs/HANDOFF.md` Phase 4 reset to TODO pending v1.4 retraining.
- Why: pre-D22 numbers came from the unservable-S0 loop and were
  poisoning analysis context. Raw `data/` untouched.
- Revisit if: never — purged artifacts are unrecoverable by design
  (regenerate under v1.4 instead).

## Template for new entries
## D<n> — <title>
- Date:   Status: proposed | accepted | superseded by D<m>
- Decision / Why / Revisit if:

## D17 — Vectorized rollout hot loop (semantics-preserving)
- Date: 2026-09-14, Status: accepted
- Decision:
  - Rewrote `_rebuild_caches` pending/candidate section (`env/pettingzoo_env.py`) with identical outputs, verified **bit-identical** vs old code on 12 NYC + 25 Bandra turns (all cache arrays + per-zone candidate order-id lists):
    - Per-order normalized features computed ONCE into a `(P, 6)` table, reused for the global urgency view and every zone's candidate view (was: 1 + `num_zones` Python normalizations per order per flush).
    - One `(P, Z)` distance matrix per flush replaces per-zone distance loops.
    - Per-zone Python-keyed `list.sort(key=lambda…)` replaced by `np.lexsort` (stable → identical tie order, identical IEEE floats).
    - Zone attribution loop replaced by `np.bincount`; `max_dispatch_radius` hoisting; duplicate `origins`/`relocation_points` array builds removed.
  - `RolloutBuffer.flatten` zero-copy fast path (`agents/ippo.py`): fully-filled buffers return reshape views instead of boolean-mask copies (saves a ~3.3 GB copy at NYC scale). Masked path kept as fallback.
  - No env dynamics, reward, observation, or action-space change. Config version NOT bumped (infra-only, outputs proven identical).
- Why:
  - NYC rollout was ~9.7 s/episode with per-zone lambda sorts (~220 M calls/episode) dominating the single-threaded rollout loop while GPU sat idle.
  - Measured (NYC, 4 episodes, 1 iter, seed 1, serial): rollout 38.7 s → 31.5 s, update 9.6 s → 5.8 s (flatten), total 61.6 s → 41.7 s (**1.48x**).
- Revisit if:
  - `max_pending` or zone count changes materially (re-profile; matrix is `P × Z`).
  - Moving to shared-memory worker results (see `docs/rollout-performance.md`).
- Note (2026-09-14, resolved): batched `act()` + boundary-drain fix changed the RNG
  stream and per-episode transitions vs pre-batching code, and the first
  batched version additionally scrambled action attribution (fixed under
  D18). Once attribution was fixed, same-seed levels reproduce:
  `mappo-nyc-fixed-seed0` sr≈0.56→0.49 tracks `mappo-nyc-proper-seed0`
  sr≈0.55→0.51 throughout, with EV now climbing (0.00→+0.26) where the
  scrambled run sat at EV≈0. Do not mix pre/post-fix seeds in one reported
  aggregate (trajectories still differ bit-wise from batching).

## D18 — MAPPO centralized-critic rollout gap (fixed 2026-09-14)
- Date: 2026-09-14, Status: accepted (researcher directive; advisor review still welcome)
- Correction to the original entry: the gap was NOT pre-existing. The
  pre-batching loop stepped `agent_selection` directly and already stored
  `state=concat([env.state(), zone_location])` with `learner.value()` critic
  estimates. The batched rewrite dropped both. This fix restores that design
  in batched form — no new science, just the originally intended semantics.
- Fix (`experiments/train.py::collect_rollout`):
  - MAPPO: per-agent state (`env.state()` + own `zone_location`, matching
    `train.py`'s `state_dim`) stored per transition; values from
    `learner.value()` centralized critic; actor-head values discarded (as
    `MAPPO.act`'s contract always specified).
  - Rewards/done-flags read from per-agent tables (`env.rewards`,
    `env.terminations`) instead of `env.last()`, which reports only the
    selected agent. Proven exactly equivalent to ordered `last()` reads
    (both equal the post-flush credit: step() zeroes both tables).
  - Selection-ordered two-phase walk: phase 1 closes + observes (no steps),
    single batched `act()`, phase 2 steps each agent exactly once in
    `env._agent_order`. This preserves the pre-batching
    attribute-every-action-to-the-selected-agent semantics — verified by
    probe (24/24 actions land on the correct zones; the step() walk mirrors
    `_advance_selection` exactly).
- Audit fallout (same session): the batched rewrite had been stepping zones
  in `zone_0..N` order while `step()` records for the selected agent —
  every dispatch/relocate executed on the WRONG zone. This, plus the RNG
  stream change, explains the `mappo-nyc-new-s0` (sr≈0.41) vs
  `mappo-nyc-proper-seed0` (sr≈0.55) gap. Do not mix pre/post-fix seeds in
  one aggregate (see D17 note).
- Verification: Bandra MAPPO probe — states stored/non-zero/finite, values
  differ from actor head, update runs; Bandra IPPO 1-iter smoke passes with
  metrics written.
- Sweep remainder (no action): `IPPO/MAPPO.update` standard PPO-clip (no
  value-clip — legitimate variant); `compute_gae` boundary handling sound;
  env flush/reward attribution coherent; `evaluate.py`/`shift_eval.py` have
  no optimizer/backward/update/save calls (frozen protocol holds); masked
  `-inf` logits produce finite entropy in all runs observed.

## D19 — Entropy-coefficient probe (keep 0.01, no retraining)
- Date: 2026-09-14, Status: accepted
- Probe: `mappo-nyc-ent002-seed0` (entropy_coef 0.02) vs
  `mappo-nyc-fixed-seed0` (0.01), otherwise identical (NYC, 70×3, winlim 500).
- Result: entropy held 2.4x higher (H 0.022 vs 0.009 at it70) but actions
  barely moved (idle 98.3% vs 99.0%, dispatch 1.0% vs 0.9%). End-point SR gap
  (0.573 vs 0.490) on near-identical action mixes reads as window noise on
  n=1, not a policy difference. EV slightly better (+0.33 vs +0.21).
- Decision: keep entropy_coef 0.01 for all reported runs; no retraining.
  The binding constraint is reward shaping (dispatch rarely pays), not
  exploration — a higher entropy bonus burns compute without changing
  behavior. Revisit only alongside a reward-shaping change, with both algos
  retuned equally per the baseline-fairness rule.

## D20 — Batched shift-eval loop (verified within 0.1% of published CSVs)
- Date: 2026-09-14, Status: accepted
- Change (`experiments/shift_eval.py::evaluate_policy` only; env, scenarios,
  seeds, metrics untouched): per-turn selection-ordered walk with a single
  deterministic `act()` over all zones instead of 62 batch-1 calls
  (~90 s/episode → ~15 s/episode; the 10-checkpoint suite was otherwise
  ~14 h). Terminated turns drain; rewards accumulated once per turn from
  per-agent tables (proven equal to the old per-selection reads).
- Verification: `mappo-proper-seed0` S0 reproduces
  `results/shift_eval/mappo-proper-seed0_S0.csv` to served 234.2 vs 234.4,
  SR 0.5109 vs 0.5114, reward 376 vs 374 — residual is a single argmax flip
  from batched-GEMM rounding over 5 episodes, far below seed std (~0.03).
  Old-loop rerun reproduces the CSV exactly, isolating the delta to batching.
- Fixed-code evals go to `results/shift_eval_fixed/` (old results untouched).

## D23 — Structural MARL Env, Demand Generation, and Credit Overhaul
- Date: 2026-09-16, Status: accepted
- Decisions:
  1. **Demand Generation:** In `UniformSampledNYCOrderGenerator`, preserve temporal clustering / burstiness via contiguous block/chunk sampling rather than uniform i.i.d. random subsampling that flattens natural arrival clumps.
  2. **Turn Granularity & Skeleton:** Set `dt = 0.5` min (30 sim-seconds), reducing turns per 60-min episode from 600 to 120 (cutting 80%+ idle filler, matching 2-min timeout timescale).
  3. **Strict Spatial Zone Ownership:** Restrict candidate dispatch orders in Zone $z$ strictly to orders originating in Zone $z$. This eliminates cross-agent race conditions, double-booking, and random-shuffle gradient noise.
  4. **Immediate Dispatch Reward:** Credit $+ \gamma$ immediately when an order is matched/dispatched at turn flush, eliminating the 100+ step temporal credit assignment delay.
  5. **Shared Neighbor Cancellation Penalty:** When an unassigned order times out in Zone $z$, share the penalty with adjacent neighboring zones that held idle vehicles.
  6. **Observation Space Clean-up:** Remove static `relocation_points` (124 dead floats) and add explicit supply-demand surplus/deficit features ($(\text{pending} - \text{idle}) / N$).

## D24 — Relocation Loop Suppression & Spatial Dispersion Tuning
- Date: 2026-09-16, Status: accepted
- Problem: Low empty-travel penalty ($\beta = 0.1$) compared to dispatch payoff ($\gamma = 5.0$) made cross-city relocation virtually cost-free ($-0.25$ penalty for 5 km). Multiple zone agents simultaneously herded idle vehicles into the same popular destination zones, creating cyclic ping-pong loops and leaving peripheral zones unserviced.
- Decisions:
  1. **Tuning $\beta$ Penalty:** Increase empty-distance penalty weight from $\beta = 0.1 \to 0.35$ in `NYCConfig`. Empty deadheading now incurs meaningful economic trade-off, strongly suppressing frivolous round-trip looping.
  2. **Surplus-Target Relocation Masking:** In `action_mask()`, dynamically mask out relocation to target zones that already have an idle vehicle surplus ($\text{idle}_j > \text{pending}_j + 1$). Agents are forced to route surplus vehicles to zones with actual unmet demand or deficit.

## D25 — Reward Magnitude Normalization & Cancellation Parity
- Date: 2026-09-16, Status: accepted
- Problem: Inflated reward magnitude ($\gamma = 5.0$, yields $+1500$ to $+1700$/ep) dwarfed penalties ($-0.5$ cancellation penalty, $-0.1$ waiting), leaving the agent indifferent to high cancellations or long empty travels as long as some rides were completed.
- Decisions:
  1. **Normalized Scale:** Set $\gamma = 1.0$ (served demand payoff).
  2. **Cancellation Parity:** Set $\text{cancel\_penalty} = 1.0$ (matching the $+1.0$ payoff: serving is $+1.0$, dropping is $-1.0$).
  3. **Empty Deadhead Cost:** Set $\beta = 0.15$ per empty km (so a $5\text{ km}$ empty relocation costs $-0.75$, requiring $75\%$ of a ride's payoff).
  4. **Waiting Cost:** Set $\alpha = 0.2$ per normalized waiting timeout.
  5. **Consequence:** Net episode reward centers around $[-200, +150]$, providing sharp gradient incentives against poor rebalancing.

## D26 — Localized Neighbor Relocation & 200-Iteration Training Scale-Up
- Date: 2026-09-16, Status: accepted
- Problem: Global unconstrained relocations allowed agents to order 15-minute cross-Manhattan highway trips, accumulating high empty mileage and leaving vehicles in transit during sudden local demand spikes.
- Decisions:
  1. **Neighbor Relocation Only:** Enable `neighbor_relocation_only = True` in `NYCConfig`. Relocations are restricted to contiguous adjacent zone neighbors, creating smooth spatial diffusion and short 2–4 minute repositioning hops instead of 20-minute cross-city trips.
  2. **Training Horizon Scale-Up:** Scale training to 200 iterations (1.2M transitions) to observe deep convergence properties and long-term rebalancing policy dynamics.

## D27 — LP rebalancer objective repair (silent no-op → tuned baseline)
- Date: 2026-09-16, Status: accepted
- Problem: `ZoneLPRebalancer` was indistinguishable from `GreedyZoneDispatcher`
  on all 10 shift scenarios (`pct_relocate = 0.0`, byte-identical KPIs). Root
  cause is structural, not a tuning accident: the LP minimised travel cost
  subject only to upper bounds (`outflow <= surplus`, `inflow <= deficit`,
  `total <= n_idle`), so with all `c_ij >= 0` the all-zero plan was feasible
  *and* optimal. The solver was correct; it was always right to move nothing.
  No `speed_kmh`, `min_idle_to_rebalance` or `lookahead_minutes` value can fix
  that (verified: even cost scale 0.01 returns zeros).
- Decisions (`baselines/lp_baseline.py`, baseline-local; no env dynamics,
  reward, or observation/action-space change):
  1. **Shortfall term in the objective.** Minimise
     `sum c_ij x_ij - lambda * sum_j inflow_j`, with
     `lambda = deficit_penalty_scale * mean(off-diagonal c)`. The reduced cost
     `c_ij - lambda` is the net value of a relocation, so a route is used only
     when filling the destination's predicted shortfall beats its travel time.
     `deficit_penalty_scale = 0` reproduces the old behaviour exactly.
  2. **Self-transfer variables pinned to zero** (`bounds = (0, 0)`) instead of
     relying on a zero diagonal cost.
  3. **Demand-rate units fixed.** `env._pending_per_zone` is pending orders per
     zone divided by fleet size (D22), but the estimator multiplied it by the
     total pending count; now scaled by `num_drivers`.
  4. **Deterministic rounding.** Fractional LP flows were rounded against the
     global `np.random` stream; the policy now owns a generator seeded from
     `reset(seed)`.
  5. **`speed_kmh` left at the environment's true value (30 km/h).** The
     previous reference note told readers to *lower* it, which is backwards
     (cost is travel time) and would be fudged physics in the other direction.
  6. **Defaults tuned on S0 (train) + S3 (val) only:** `lookahead_minutes 3 → 5`,
     `deficit_penalty_scale 1.0 → 0.4`. S1/S4/S5/S6 stay held out (AGENTS.md
     rule 3).
- Result (5 seeds, base_seed 1000, greedy floor in brackets): S0 service rate
  70.5% [52.9%], S3 70.6% [46.3%], `pct_relocate` 7.2% [0.0%]. Empty distance
  rises 115 km → 812 km, i.e. the gain is bought with 5–7× the repositioning
  distance; both KPIs must be reported together.
- Notably the *shorter* horizon won: a horizon longer than the mean inter-zone
  travel time (8.6 min) flags ~50 of 62 zones as short and destroys the spatial
  selectivity of the plan, so 15–30 min scored worse than 5 min on both service
  rate and empty km.
- The extra distance is not wasted: the tuned LP also beats greedy on the
  environment's own reward (S0 +17.9 vs -112.7), so the gain is not an artifact
  of optimising a single KPI.
- Held-out check (scored once, after defaults were frozen): LP >= greedy on
  service rate in all of S1/S2a-e/S4/S5/S6, mean retention 90.5% vs 85.7%,
  with the largest margin on S1 (79.4% vs 71.1%). Full KPIs in
  `results/shift_eval_lp_fixed/`.
- Consequence: LP is now a genuinely tuned baseline (D3) rather than a floor
  clone. Pre-D27 LP rows in `results/shift_eval/` are invalid and must not be
  cited; re-run baselines before any head-to-head.

## D28 — Relocation arrival bonus & NYC action-space compression
- Date: 2026-09-17, Status: accepted
- Problem: Agents never learn relocation because the credit arrives too late
  (8-18 turns after the command) and is too small relative to the immediate
  empty-km cost. With gamma=0.99, the discounted value of +0.3 assist credit
  perceived 12 turns early is ~0.266, while the immediate cost of a 2 km
  relocation is -0.30 — net negative. Additionally, max_pending=80 wastes
  50+ dispatch slots that are always masked, diluting relocation actions in
  the policy output (62 out of 143 vs 62 out of 93).
- Decisions:
  1. **Relocation arrival bonus** (`prospective_bonus_scale`): When a relocated
     driver arrives at its destination zone (status transitions from
     "relocating" to non-"relocating"), the relocator zone receives an
     immediate bonus: `prospective_bonus_scale * pending_demand_at_dest / num_drivers`.
     This eliminates the temporal credit delay — the relocator is rewarded at
     arrival, not at eventual dispatch. Default 0.0 (disabled); NYCConfig set
     to 2.0.
  2. **NYC max_pending 80 → 30**: Measured peak concurrent pending is ~28.
     Reduces action space from 143 to 93, observation from ~746 to ~506
     features, and makes relocation actions proportionally more likely to be
     sampled (62/93 vs 62/143).
  3. **Config version bump**: 1.8 → 1.9.
- Implementation:
  - `pettingzoo_env.py`: Detect arrivals in `_flush_turn()` by comparing
    old/new driver statuses before/after `env.step()`. Store in
    `_relocator_arrivals`. Apply bonus in `_compute_zone_rewards()` section 1b.
  - `config.py`: Added `prospective_bonus_scale: float = 0.0` to `EnvConfig`;
    NYCConfig sets it to 2.0.
- Revisit if:
  - Agents over-relocate (high empty km, low service rate) → reduce
    prospective_bonus_scale to 1.0.
  - Agents still don't relocate → increase to 3.0-5.0 or add a direct
    relocation success bonus when the relocated driver serves an order.
  - The entropy_coef change (D19 recommends 0.01 alongside reward shaping
    changes) needs retuning — use 0.05 for initial exploration, anneal to
    0.01 after 50 iterations.
- Note: D19 concluded "the binding constraint is reward shaping, not
  exploration." That holds for the old broken reward. With the arrival bonus
  fixed, exploration of relocation actions becomes the next bottleneck.
  Recommend `--entropy-coef 0.05` for the first 50 iterations, then anneal
  to 0.01. This is a judgment call to be validated on train/val only.

## D29 — Decode enforces neighbor + zone ownership
- Date: 2026-09-29, Status: accepted
- `_decode_action` now rejects non-neighbor relocate targets and drops the
  global-pending dispatch fallback. Mask already enforced both; decode was
  the hole. On-mask behavior unchanged. Config 1.9 → 2.0.




