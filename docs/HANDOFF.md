# HANDOFF.md — Current state and next steps

Read this file first. It tells you what exists, what's done, what's broken,
and what to do next. Everything else in the repo supports this.

---

## What this project is

We measure how MARL fleet-rebalancing policies (IPPO vs MAPPO) degrade under
passenger-demand distribution shift. This is a **measurement study**, not an
algorithm paper. We train on one demand distribution, freeze the policy, then
stress-test on shifted distributions and plot degradation curves.

**Advisor:** Alaa Daoud (INSA Hauts-de-France, LAMIH CNRS)
**Student:** Prashil

---

## What exists right now (verified working)

### Simulation (RideGym)
- Bandra West, Mumbai — OSMnx graph at `data/bandra_west.gpickle`
- `main.py` — rolls out a random policy, saves MP4 or opens interactive viewer
- `src/marl_fleet_rebalancing/visualization.py` — custom MP4 renderer + interactive viewer

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
- `docs/architecture.md` — has inaccuracies (zone partition claimed as current)

---

## What does NOT exist yet (blocking)

### NYC data (required for paper)
- No `data/nyc/` directory — must run the 4-stage pipeline
- No `dataset/taxi_zones/` or `dataset/fhvhv_tripdata_*.parquet` — must download manually
- `max_pending` in NYCConfig is still 80 (from Bandra) — must measure from NYC data
- See `docs/nyc-pipeline.md` for exact steps

### Agents / training code
- Only `src/marl_fleet_rebalancing/agents/random_policy.py` exists
- No IPPO or MAPPO implementation
- No training loop
- No experiment runner

### Baselines
- No classical rebalancing baselines (greedy, LP, MPC)
- No tuning infrastructure

### Metrics / evaluation
- No metric logging beyond the reward
- No degradation curve plotting
- No experiment comparison tools

---

## Key design decisions (DO NOT change without logging in decisions.md)

| Decision | Status | File |
|---|---|---|
| Simulator: RideGym | Accepted | `docs/decisions.md` D1 |
| Study area: NYC Manhattan | Accepted | `docs/decisions.md` D6 |
| Config-driven params | Accepted | `docs/decisions.md` D7, `config.py` |
| Reward: -alpha*wait - beta*empty + gamma*served | Accepted | `docs/decisions.md` D8 |
| IPPO vs MAPPO only | Accepted | `docs/decisions.md` D4 |
| Dispatch + rebalancing (not dispatch only) | Accepted | wrapper has both actions |

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

### Phase 1: NYC data pipeline
1. Download NYC TLC taxi-zone shapefile → `dataset/taxi_zones/`
2. Download NYC TLC FHVHV parquet → `dataset/`
3. Run `python -m ride_gym.data_tools.nyc.zone_centroids`
4. Run `python -m ride_gym.data_tools.nyc.build_nyc_network`
5. Run `python -m ride_gym.data_tools.nyc.build_splits`
6. Measure actual `max_pending` from NYC data → update `NYCConfig`
7. Test wrapper with NYC config

### Phase 2: Training infrastructure
1. Implement IPPO (using existing PPO library or from scratch)
2. Implement MAPPO (shared critic variant)
3. Create training loop that uses PettingZoo API
4. Add metric logging (served rate, wait time, empty km, reward)
5. Train on NYC base demand, verify agents learn something

### Phase 3: Baselines
1. Implement greedy nearest-idle dispatch
2. Implement demand-following heuristic
3. Implement optimization-based rebalancer (LP/MIVR-lite)
4. Tune all baselines with comparable effort to RL

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

## Contact

If something is unclear, check `docs/decisions.md` first. If still unclear,
ask Prashil before making changes that affect the scientific claim.
