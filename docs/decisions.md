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

## Template for new entries
## D<n> — <title>
- Date:   Status: proposed | accepted | superseded by D<m>
- Decision / Why / Revisit if: