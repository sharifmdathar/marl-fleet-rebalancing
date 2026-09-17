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

## 2026-09-17: D28 paired retraining done — H1–H4 answered

- Both PPO variants retrained under the **identical D28 protocol** (env v1.8,
  two-stage entropy 0.05→0.01, neighbor relocation, validation-selected
  checkpoints on the true held-out val split): `ippo-nyc-d28-stage2` and
  `mappo-nyc-d28-stage2`. The relocation collapse is gone at the source
  (`pct_relocate` stays ~14% / ~10% through training).
- 5-seed frozen shift eval (S0–S6, seeds 1000–1004) complete for both +
  baselines. Head-to-head: MAPPO 56.9% S0 / 87.6% retention vs IPPO 56.5% /
  84.8% — **within noise; H3 is the answer, with a lean toward H4** (MAPPO's
  edge concentrates on demand-level shifts S1/S3; both collapse below greedy
  on S4). Tuned LP still leads every shift at half IPPO's empty km.
- Full writeup with tables + caveats: `docs/shift-results.md` (head-to-head
  rebuilt by `scripts/summarize_shift_eval.py`). Figures updated in
  `results/figures/`.
- **Remaining for submission:** repeat each D28 training with 2–3 training
  seeds (deltas are currently single-training-seed); reconcile `INTENT.md`'s
  relocate-only non-goals line with D9's joint action space.

## 2026-09-16: D22 env overhaul + purge of all pre-D22 artifacts

- Code audit (no training logs) found six S0 blockers; fixed as config
  v1.4, logged in D22 (winlim 500 default, urgent-first candidates,
  legible demand signal, anywhere relocation, reward rescale,
  max_pending 80).
- `runs/` and `results/` purged (checkpoints, evals, figures, videos).
  `docs/shift-results.md` is a stub; `docs/demand-shift.md` keeps the
  taxonomy/mechanisms only. Do NOT mix pre/post-D22 seeds in one aggregate.
- Next: retrain IPPO + MAPPO on S0 (v1.4) to competence, tune baseline
  (D3), then frozen-policy eval S0–S6.

## 2026-09-16 (later): D27 LP baseline repaired and tuned

- `ZoneLPRebalancer` was a silent no-op: its objective priced travel cost but
  put no value on covering a predicted shortfall, so with all costs non-negative
  the all-zero plan was exactly optimal. It matched greedy on all 10 shifts with
  `pct_relocate = 0.0`. No speed/horizon/threshold tuning could change that.
- Fixed in D27: shortfall term in the objective (`deficit_penalty_scale`),
  demand-rate unit bug, deterministic rounding. `speed_kmh` deliberately left at
  the env's true 30 km/h.
- Tuned on S0+S3 only (held-out shifts untouched): defaults now
  `lookahead_minutes=5.0`, `deficit_penalty_scale=0.4`. S0 service rate
  70.5% vs greedy 52.9%; S3 70.6% vs 46.3%; empty km 115 → 812.
- New sweep harness: `experiments/tune_lp_baseline.py` (writes to
  `results/lp_tuning*/`).
- Head-to-head re-run at 5 seeds (seed 1000–1004) over all 6 policies →
  `results/shift_eval/` + `results/figures/`. Numbers and caveats in
  `docs/shift-results.md`. The pre-D27 3-seed run is parked in
  `results/shift_eval_preD27/` (invalid for citation, safe to delete).
- Two things the re-run surfaced: the tuned LP leads every shift (S0 70.5% vs
  best MAPPO 57.2%), and MAPPO at 400 iterations has *stopped relocating*
  entirely (`pct_relocate` 0.01%, service rate down to greedy's). No IPPO
  checkpoints exist, so H1–H4 remain unanswerable.

## 2026-09-17: validation-selection protocol bug found and fixed

- The first validation-selected pair (`ippo-v18-val` it12, `mappo-v18-val`
  it8 → `results/shift_eval_v18best/`) is NOT a clean comparison. Two bugs:
  (1) selection ran on the TRAINING split (same windows the policy learns
  on); (2) selection ran in the training env itself, so its seeded resets
  (9000–9002) rewrote the training RNG stream — enabling validation changed
  subsequent training episodes. A separate probe also confirmed the
  three-episode val reset replays only the first val window (cursor reset).
- Fixed in `train.py`: selection now runs `evaluate_deterministic` in a
  DEDICATED env deep-copied from the training config with `nyc_split="val"`
  (`_make_validation_env`); the training env is never touched by selection
  (toggle-verified: 2-iter training trajectories byte-identical with and
  without `--eval-interval`). Provenance: `TrainConfig` logs
  `eval_split/eval_seed/eval_episodes/eval_window_limit`; `best_checkpoint.json`
  records `val_split`, `eval_seed`, `eval_episodes` (auditable selection).
  Regression-locked in `tests/test_training_validation.py`.
- Consequence: **`results/shift_eval_v18best/` and the v18best rows must not
  be merged into the headline artifacts or cited as validation-selected.**
  Treat as exploratory only. Next: retrain IPPO + MAPPO on S0 with
  `--eval-interval` (now on the true val split), then rerun the 5-seed
  shift eval for the fair H1–H4 comparison.
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

### PettingZoo wrapper (zone-controller, D10 — reworked 2026-09-08)
- `src/marl_fleet_rebalancing/env/pettingzoo_env.py` — AEC wrapper, ONE AGENT
  PER ZONE (`zone_i`); joint dispatch+rebalance action space per D9; per-zone
  idle-driver FIFO resolution (D13). Smoke test passes on Bandra.
- `src/marl_fleet_rebalancing/env/test_wrapper.py` — smoke test; `--nyc` flag
  runs the NYC split mode (62 taxi-zone agents, real FHVHV window). Passes.
- NYC mode: `order_source="nyc"` (NYCConfig) replays split windows via
  `MultiWindowNYCOrderGenerator`; zone model = taxi-zone centroids; verified
  600 turns on a train window.
- Tested: Bandra 25 zones × 300 turns; NYC 62 zones × 600 turns.

### Config
- `src/marl_fleet_rebalancing/config.py` — `EnvConfig`, `NYCConfig`, `BandraConfig`
- All parameters documented with source/rationale. No magic numbers.

### Reward function & Credit Assignment
- Implemented in `pettingzoo_env.py` (`_compute_zone_rewards` method).
- Formula per zone: $R_z = (1 - \text{team\_share}) \cdot R_{\text{local}, z} + \text{team\_share} \cdot \bar{R}_{\text{global}}$.
- Local credit assignment: $+5.0$ to the zone that dispatched the completed order, $-0.1 \times \text{km}$ to the zone commanding empty movement (exact physical distance from RideGym events), and $-1.0 \times \text{wait}$ localized by order origin.
- Default weights: $\alpha = 1.0, \beta = 0.1, \gamma = 5.0, \text{team\_share} = 0.2$.

### Docs
- `docs/agent-mechanics.md` — Diagnostics of root issues, what agents observe, action mapping, and credit assignment.
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
  decision before training (advisor call). The wrapper exposes the lever as
  `NYCConfig.nyc_window_limit` (per-window order cap); even a 2k-order window
  floods a 50-driver fleet (1.97k/2k cancelled at the random-policy floor).

### Agents / training code
- `src/marl_fleet_rebalancing/agents/ippo.py` — IPPO DONE 2026-09-09
  (Phase 2 step 1): shared actor-critic over zone agents, decentralized critic,
  masked PPO-clip + GAE; rollout buffer handles AEC delayed-close rewards.
- `src/marl_fleet_rebalancing/networks/encoder.py` — zone-obs flattening +
  `ZoneActorCritic` MLP (masked logits).
- `src/marl_fleet_rebalancing/experiments/train.py` — training entry
  (`python -m marl_fleet_rebalancing.experiments.train --env bandra --seed 0`);
  logs config/seed/git-commit/scenario-ID + per-iteration KPIs to `runs/`
  (git-ignored). Torch (CUDA) added as dependency.
- Smoke-verified on Bandra (25 zones, obs 746, 106 actions, CUDA): rollout →
  GAE → PPO update → checkpoint all work; 10-iter trend run shows machinery
  healthy. NOT yet tuned; "agents learn something" gate = Phase 2 step 5.
- MAPPO DONE 2026-09-09 (Phase 2 step 2): `agents/mappo.py` — same PPO
  backbone, shared `ZoneActorCritic` actor + `CentralizedCritic` over the
  wrapper's global `state()`; buffer carries per-transition states;
  `train.py --algo ippo|mappo` (default ippo, IPPO path unchanged).
  Smoke-verified on Bandra (CUDA, 2 iters, same-seed byte-identical
  metrics + checkpoint). No experiment runner beyond train.

### Baselines — DONE 2026-09-16 (D27)
- `baselines/greedy_baseline.py` — nearest-idle dispatch floor (zero relocation).
- `baselines/lp_baseline.py` — rolling-horizon LP rebalancer on the wrapper env,
  repaired and tuned in D27 (it was a silent no-op before; see
  `references/lp-rebalancer-zero-transfers.md`).
- `experiments/tune_lp_baseline.py` — sweep harness; tuning evidence in
  `results/lp_tuning*/`.
- `baselines/heuristic_baseline.py` — earlier LP rebalancer on the raw RideGym
  dict interface; superseded by `lp_baseline.py`, kept for reference.

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
| Zone-controller wrapper implementation | Accepted | `docs/decisions.md` D13 |

### PettingZoo wrapper design
- **AEC (not Parallel)** — same training signal, matches RideGym API
- **Shuffled agent order** each turn — prevents iteration-order bias
- **Action mask via `action_mask(agent)` method** — not baked into obs
- **Conflict resolution:** shuffled first-come-first-seeded (seeded RNG)
- See `docs/pettingzoo-design-decisions.md` for rationale

---

## Architecture of the wrapper (zone-controller, D10/D13)

```
RideGymAEC (PettingZoo AEC) — one agent per zone: zone_0 .. zone_{Z-1}
  ├── RidePoolEnv (RideGym) — the actual simulation
  ├── EnvConfig — all parameters (order_source, nyc_split, reward, ...)
  ├── Region model — NYC: ~62 taxi-zone centroids; synthetic: relocation_grid
  │     over the network bounds (config, default 5x5)
  ├── Observation space — Dict with own-zone state, all-zone fleet state,
  │     pending orders, relocation points, clock
  ├── Action space — Discrete(1 + max_pending + num_zones)
  │     0 = no zone action
  │     1..max_pending = dispatch pending order i to an idle driver IN ZONE
  │     max_pending+1.. = relocate one idle driver of the zone to zone j
  ├── Zone→driver resolution — per-zone idle-driver FIFO queues, consumed in
  │     shuffled first-come-first-served order (seeded RNG)
  └── Reward — R = -alpha*wait - beta*empty_km + gamma*served (D8), per turn,
        shared by all zone agents; empty km exact from the RideGym event log
```

### Observation space keys (per zone agent)
- `zone_location` (2,) — own zone centroid, normalized to [0,1]
- `zone_state` (3,) — fleet fraction (drivers in zone / fleet), idle fraction
  (within zone), pending-originating fraction (orders / max_pending)
- `all_zone_state` (num_zones, 2) — per-zone [fleet fraction, idle fraction]
- `pending_orders` (max_pending, 6) — origin_xy, dest_xy, pax/capacity,
  wait/timeout; sorted MOST-URGENT-FIRST
- `pending_mask` (max_pending,) — 1 if order exists
- `pending_passengers` (max_pending,) — pax/capacity
- `time_frac` (1,) — current_time / horizon
- `relocation_points` (num_zones, 2) — zone centroids (normalized)

### Action space (per zone agent)
- `0` = no zone action
- `1` to `max_pending` = dispatch pending order (i-1) to an idle driver in this zone
- `max_pending+1` to `max_pending+num_zones` = relocate one idle driver to zone (j - max_pending - 1)

---

## What to do next (in order)

### Phase 1: NYC data pipeline — DONE 2026-09-07 (all artifacts on disk)
Wrapper NYC support DONE 2026-09-08 (D13): the wrapper replays the splits via
`MultiWindowNYCOrderGenerator` and smoke-tests green on a train window.

### Phase 2: Training infrastructure
0. Rework wrapper to zone-controller granularity (D10) + NYC split support —
   **DONE 2026-09-08 (D13)**; smoke tests green on Bandra and NYC.
1. Implement IPPO — **DONE 2026-09-09** (machinery; see agents/ippo.py +
   experiments/train.py). Longer training + tuning = step 5.
2. Implement MAPPO (shared critic variant) — **DONE 2026-09-09**
   (machinery; see agents/mappo.py). Longer training + tuning = step 5.
3. Create training loop that uses PettingZoo API — **DONE 2026-09-09**
   (experiments/train.py drives the AEC API for both algos).
4. Add metric logging (served rate, wait time, empty km, reward)
5. Train on NYC base demand, verify agents learn something

### Phase 3: Baselines
1. Implement greedy nearest-idle dispatch (the floor) — **DONE 2026-09-11** (`baselines/greedy_baseline.py`)
2. Port LP rebalancer to the wrapper env — **DONE 2026-09-11** (`baselines/lp_baseline.py` + integrated in `experiments/evaluate.py`)
3. Tune all baselines with comparable effort to RL (D3 / RLSCALE-BENCH lesson)
   — **DONE 2026-09-16 (D27)**; sweep in `experiments/tune_lp_baseline.py`.
   Caveat: only the LP needed tuning (greedy has no parameters), and the tuned
   LP now leads every MAPPO run on service rate.

### Phase 4: Demand shift experiments (pending v1.4 retrain)
1. Shift injection (spatial, magnitude, burst, permute) — DONE
   (`experiments/shift_eval.py`; S0–S6 per `docs/demand-shift.md`).
2. Freeze trained policies, evaluate on shifted scenarios — **DONE
   2026-09-17** (5 seeds, S0–S6, both D28 policies + baselines;
   `results/shift_eval/`).
3. IPPO vs MAPPO head-to-head + H1–H4 writeup — **DONE 2026-09-17**:
   `docs/shift-results.md` (answer: H3, with H4-type residue on S1/S3;
   caveats: single training seed per algo).

---

## Files to read (in order)

1. `AGENTS.md` — rules and constraints
2. `INTENT.md` — research questions, hypotheses, non-goals
3. `docs/decisions.md` — all design decisions with rationale
4. `docs/demand-shift.md` — shift taxonomy (S0-S6)
5. `docs/literature.md` — what's known, what's not
6. `config.py` — all experiment parameters
7. `env/pettingzoo_env.py` — the wrapper (zone-controller, D13)
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
- ~~**Baseline understanding**: LP + greedy plan stands~~ — closed by D27: the
  LP was no-op'ing on a missing objective term, now repaired and tuned. Open
  follow-up: why it still relocates at ~7% of actions once 45 of 62 zones read
  as short (its shortfall estimate ignores the busy fleet freeing up inside the
  horizon).
- **Reward weights**: equal start, revisit if tuning stalls.

## Contact

If something is unclear, check `docs/decisions.md` first. If still unclear,
ask Prashil before making changes that affect the scientific claim.
