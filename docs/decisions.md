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

## Template for new entries
## D<n> — <title>
- Date:   Status: proposed | accepted | superseded by D<m>
- Decision / Why / Revisit if:
