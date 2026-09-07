# HANDOFF.md — Current state and next steps

Read this file first. It tells you what exists, what's done, what's broken,
and what to do next. Everything else in the repo supports this.
Rules live in `AGENTS.md`; research scope in `INTENT.md` — this file is status, not rules.

---

## What this project is

We measure how MARL fleet-rebalancing policies (IPPO vs MAPPO) degrade under
passenger-demand distribution shift. This is a **measurement study**, not an
algorithm paper. We train on one demand distribution, freeze the policy, then
stress-test on shifted distributions and plot degradation curves.

**Advisor:** Alaa Daoud (INSA Hauts-de-France, LAMIH CNRS)
---

## What exists right now (verified working)

### Simulation (RideGym)
- Bandra West, Mumbai — OSMnx graph at `data/bandra_west.gpickle`
- `main.py` — `--policy random|heuristic`, config-driven (`BandraConfig`),
  saves MP4 or opens interactive viewer
- `src/marl_fleet_rebalancing/visualization.py` — custom MP4 renderer + interactive viewer
- `src/marl_fleet_rebalancing/baselines/heuristic_baseline.py` — LP-based
  rolling-horizon rebalancer (MIVR-lite, D3); raw RideGym interface, needs a
  wrapper port for fair comparison (Phase 3)

### PettingZoo wrapper
- `src/marl_fleet_rebalancing/env/pettingzoo_env.py` — AEC wrapper around RideGym
- `src/marl_fleet_rebalancing/env/test_wrapper.py` — smoke test (passes on Bandra)
- Tested: 30 drivers, 25 zones, 300 turns, 9000 step calls

### Config
- `src/marl_fleet_rebalancing/config.py` — `EnvConfig`, `NYCConfig`, `BandraConfig`
- All parameters documented with source/rationale. No magic numbers.

### Reward function
- Implemented in `pettingzoo_env.py` (`_compute_reward` method)
- Formula: `R = -alpha * waiting_time - beta * empty_distance + gamma * served_demand`
- Weights are config params (alpha, beta, gamma)
- **Caveat:** empty_distance is approximate (uses status proxy, not exact distance tracking)

### Docs
- `docs/decisions.md` — D1-D8 decisions logged
- `docs/demand-shift.md` — S0-S6 shift taxonomy
- `docs/literature.md` — gap sentence + adjacent work table
- `docs/pettingzoo-design-decisions.md` — why AEC, shuffled order, action mask as method
- `docs/pettingzoo-wrapper-plan.md` — implementation plan
- `docs/nyc-pipeline.md` — NYC data setup steps
- `docs/architecture.md` — module map + KPI definitions

---

## What does NOT exist yet (blocking)

### NYC data (required for paper)
- DONE 2026-09-07: `data/nyc/` built via the 4-stage pipeline —
  `zone_centroids.csv` (263 citywide, 62 in Region A bbox),
  `manhattan.gpickle` (4868 nodes / 10478 edges),
  `splits/` 84 train / 12 val / 12 test + `manifest.json`.
- Raw downloads in `dataset/` (git-ignored): taxi-zone shapefile + FHVHV April 2026.
- `max_pending` in NYCConfig is 650 (measured worst case: 604 concurrent
  2-min arrivals in `train/window_0071`; see `config.py`).
- Open scale question: peak windows carry ~15k trips/hr vs a 50-driver fleet —
  full-rate episodes are unservable; needs a sample-rate/fleet/episode-scope
  decision before training (advisor call).
- Wrapper has NO NYC mode yet: `pettingzoo_env.py` only builds the synthetic
  `OSMnxOrderGenerator`; needs `DataFrameOrderGenerator` support for the splits.

### Agents / training code
- Only `src/marl_fleet_rebalancing/agents/random_policy.py` exists
- No IPPO or MAPPO implementation
- No training loop
- No experiment runner

### Baselines
- `src/marl_fleet_rebalancing/baselines/heuristic_baseline.py` — LP-based
  rolling-horizon rebalancer (MIVR-lite, D3) EXISTS but targets the raw
  RideGym dict interface; needs a port to the PettingZoo wrapper env for a
  fair comparison, plus tuning with effort comparable to RL (D3).
- No greedy nearest-idle floor yet (random policy in `main.py` is the
  current floor).
- No tuning infrastructure

### Metrics / evaluation
- No metric logging beyond the reward
- No degradation curve plotting
- No experiment comparison tools

---

## Key design decisions (DO NOT change without logging in decisions.md)

| Decision | Status | File |
|---|---|---|
| Simulator: RideGym (no SUMO) | Accepted | `docs/decisions.md` D1, D11 |
| Study area: NYC Manhattan, full dataset, ~62 zones | Accepted | `docs/decisions.md` D6, D12 |
| Config-driven params | Accepted | `docs/decisions.md` D7, `config.py` |
| Reward: -alpha*wait - beta*empty + gamma*served (equal weights) | Accepted | `docs/decisions.md` D8 |
| IPPO vs MAPPO only | Accepted | `docs/decisions.md` D4 |
| Joint dispatch+rebalancing (not relocate-only) | Accepted | `docs/decisions.md` D9 (supersedes D5) |
| Zone-controller agents (not per-vehicle) | Accepted | `docs/decisions.md` D10 |
| Baseline: LP rebalancer + greedy floor, tuned (D3) | Accepted | `docs/decisions.md` D3 |
| 7-scenario shift taxonomy (S0–S6) | Accepted | `docs/decisions.md` D12, `docs/demand-shift.md` |
| India dataset: dropped | Accepted | `docs/decisions.md` D12 |
| RQ1/RQ2 framing | Accepted | `INTENT.md`, `docs/decisions.md` D12 |

### PettingZoo wrapper design
- **AEC (not Parallel)** — same training signal, matches RideGym API
- **Shuffled agent order** each turn — prevents iteration-order bias
- **Action mask via `action_mask(agent)` method** — not baked into obs
- **Conflict resolution:** shuffled first-come-first-seeded (seeded RNG)
- See `docs/pettingzoo-design-decisions.md` for rationale

---

## Architecture of the wrapper

```
RideGymAEC (PettingZoo AEC)
  ├── RidePoolEnv (RideGym) — the actual simulation
  ├── EnvConfig — all parameters
  ├── Observation space — Dict with per-driver features + pending orders
  ├── Action space — Discrete(1 + max_pending + num_zones)
  │     0 = idle
  │     1..max_pending = dispatch to pending order
  │     max_pending+1.. = relocate to zone
  ├── Reward — computed per turn, same for all agents
  └── Conflict resolution — shuffled first-come-first-served
```

### Observation space keys
- `self_location` (2,) — normalized xy
- `self_status` (4,) — one-hot: idle/to_pickup/to_dropoff/relocating
- `self_passengers_ratio` (1,) — onboard / capacity
- `self_capacity` (1,) — capacity / 10
- `self_committed_ratio` (1,) — committed / capacity
- `self_region` (num_zones,) — one-hot zone
- `all_locations` (n*2,) — all driver locations
- `all_statuses` (n*4,) — all driver statuses
- `all_passengers` (n,) — all driver passenger counts
- `pending_orders` (max_pending, 5) — origin_xy, dest_xy, passengers/capacity
- `pending_mask` (max_pending,) — 1 if order exists
- `pending_passengers` (max_pending,) — passengers/capacity
- `time_frac` (1,) — current_time / horizon
- `relocation_points` (num_zones, 2) — zone centroids (normalized)

### Action space
- `0` = idle (do nothing)
- `1` to `max_pending` = dispatch to pending order index
- `max_pending+1` to `max_pending+num_zones` = relocate to zone index

---

## What to do next (in order)

### Phase 1: NYC data pipeline — DONE 2026-09-07 (all artifacts on disk)
Remaining: add `DataFrameOrderGenerator` support to the wrapper so it can run
the splits, then smoke-test the wrapper on an NYC window.

### Phase 2: Training infrastructure
0. Rework wrapper to zone-controller granularity (D10) + add
   `DataFrameOrderGenerator` support for the NYC splits — vehicle-agent
   API cannot be trained on (decisions changed under it)
1. Implement IPPO (using existing PPO library or from scratch)
2. Implement MAPPO (shared critic variant)
3. Create training loop that uses PettingZoo API
4. Add metric logging (served rate, wait time, empty km, reward)
5. Train on NYC base demand, verify agents learn something

### Phase 3: Baselines
1. Implement greedy nearest-idle dispatch (the floor)
2. Port LP rebalancer (`baselines/heuristic_baseline.py`) to the wrapper env
3. Tune all baselines with comparable effort to RL (D3 / RLSCALE-BENCH lesson)

### Phase 4: Demand shift experiments
1. Implement shift injection (spatial, magnitude, event)
2. Freeze trained policies, evaluate on shifted scenarios
3. Plot degradation curves
4. IPPO vs MAPPO head-to-head

---

## Files to read (in order)

1. `AGENTS.md` — rules and constraints
2. `INTENT.md` — research questions, hypotheses, non-goals
3. `docs/decisions.md` — all design decisions with rationale
4. `docs/demand-shift.md` — shift taxonomy (S0-S6)
5. `docs/literature.md` — what's known, what's not
6. `config.py` — all experiment parameters
7. `env/pettingzoo_env.py` — the wrapper (427 lines)
8. `docs/nyc-pipeline.md` — NYC data setup
9. `08-ps2-project-plan.md` (in parent dir) — full project plan

---

## Common mistakes to avoid

1. **Do NOT hardcode zone counts or max_pending** — always read from config or env
2. **Do NOT skip the demand-shift headline** — the paper is about degradation under shift, not in-distribution performance
3. **Do NOT claim novelty on the algorithm** — this is a measurement study
4. **Do NOT change reward/obs/action space without logging in decisions.md**
5. **Do NOT tune on test scenarios** — train/val split must be respected
6. **Do NOT use single-seed results** — minimum 5 seeds for any reported number

---

## Program gates (Batch 2 brief)

- **Gate 1 (end of week 2)** — simulation producing machine-readable results:
  MET (wrapper runs; splits are parquet windows).
- **Gate 2 (end of week 4)** — baseline + proposed approach running end-to-end:
  HALF (LP baseline runs on raw interface; no IPPO/MAPPO yet).

## Resolved scope (per PPT + prof chats, 2026-09-07 — see D9/D10/D12)

1. **Actions**: joint dispatch+rebalancing (D9 CONFIRMED).
2. **Granularity**: zone-controllers, not per-car agents (D10 CONFIRMED) —
   wrapper rework required before training (Phase 2 step 0).
3. **Scale**: full NYC dataset, ~62 data zones (CONFIRMED, no downscoping).
4. **Shifts**: keep all 7 scenarios. **India dataset**: dropped.
5. **Platform**: no SUMO, RideGym confirmed (D11).

## Still open

- **Sample-rate/fleet sizing**: full-NYC peaks (~15k/hr vs 50 drivers) are
  unservable at full rate. Must settle before training (spaces freeze).
- **Baseline understanding**: LP + greedy plan stands; "gotta understand the
  baseline" is ongoing work.
- **Reward weights**: equal start, revisit if tuning stalls.

## Contact

If something is unclear, check `docs/decisions.md` first. If still unclear,
ask Prashil before making changes that affect the scientific claim.
