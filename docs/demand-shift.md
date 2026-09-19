# Demand Distribution Shift — taxonomy & protocol

Answers: "what exactly do we mean by demand distribution shift?" Each scenario
has an ID used in configs, logs, and figures. Base distribution B = the
training demand: NYC FHVHV train-split windows, `nyc_window_limit=500`
(D22 default), 50 drivers, 60-min episodes.

## Shift taxonomy

| ID | Type | Parameterization | Real-world story |
|----|------|------------------|------------------|
| S0 | In-distribution control | B resampled (new seeds) | Sanity check: no shift |
| S1 | Spatial permutation | Rotate/permute origin-destination intensities across zones, strength p ∈ [0,1] | Hotspots move (new mall, office district) |
| S2 | Magnitude scaling | Global arrival rate × λ, λ ∈ {0.5, 0.75, 1.25, 1.5, 2.0} | Demand surge / off-season |
| S3 | Temporal phase shift | Peak window shifted by Δt minutes | Schedule change, daylight saving, Ramadan |
| S4 | Cold-zone concentration | target_zone + target_zone_pct, zone verifiably cold in train split | Policy never rewarded in target zone |
| S5 | Event-like spike | Short high-intensity burst at random node/time | Match ending, sudden rain |
| S6 | Real held-out demand | Different day/window of real trip data | The non-synthetic case |

Mapping to the prof's four shift families (D12): higher demand → S2;
spatial shift → S1 (plus S4 cold-zone concentration); temporal shift → S3;
event-type shift → S5. S0 is the in-distribution control.

## Zone model (NYC study area)

62 NYC taxi-zone centroids falling inside the Region A bbox, sorted by
LocationID (see `NYCConfig`, `docs/nyc-pipeline.md`). One zone-controller
agent per zone (D10). `BandraConfig` (synthetic grid) is kept only for
quick local testing.

## Protocol (Fujimoto-style)

1. Train on B only. Freeze policy.
2. For each scenario ID and severity level: evaluate frozen policy over
   ≥ 5 seeds × N episodes.
3. Track KPIs **over episode time**, not just final values (degradation
   can be gradual, not a cliff).
4. Primary output: degradation curve = KPI vs shift severity, one line per
   method (IPPO / MAPPO / baseline / random).
5. Robustness metric: area-normalized performance retention =
   ∫ KPI(severity) d(severity) / KPI(S0). <TODO: finalize formula after
   first results.>

## Implemented parameterizations (Phase 4, 2026-09-14)

Concrete settings used in `experiments/shift_eval.py` (`nyc_window_limit=500`
unless noted, frozen policy, 5 episodes, `base_seed=1000`):

- S3 (temporal phase shift): `nyc_split="val"` — evaluation windows drawn
  from the validation split (different hours than the train split), simulating
  a shifted peak. Same order cap (500) as S0/S2c so only timing changes.
- S5 (event-like spike, D21 mechanism): `burst_share` of orders with
  request_time in [20, 30) min reassigned to `burst_zone 0`, timestamps
  preserved. Same total volume, time-localized (see redefined section).
- S6 (real held-out demand): `nyc_split="test"` — real FHVHV test-split
  windows never touched during training or tuning.

## Redefined spatial scenarios (2026-09-14 — SUPERSEDE S1a/S1b/S4a/S5 above for reporting)

Audit finding: S1a/S1b/S4a/S5 were all implemented as the same static
`(target_zone, pct)` injection, differing only in zone + percentage —
the taxonomy's promised mechanisms (permutation, unseen-ness, burst
timing) were not implemented (S5's `event` branch was literally `pass`).
The old labels are deprecated but their CSVs stay on disk. New definitions:

- S1 (spatial permutation, strength p ∈ {0.5, 1.0}): origin AND destination
  zones remapped through a fixed rotation (offset = num_zones // 2,
  documented seed) with probability p per order; otherwise unchanged.
  Hotspots move instead of intensifying. Same order count, same timestamps.
- S4 (cold-zone concentration, k = 1): 60% of orders originate in the
  coldest train-demand zone (measured below — genuinely near-zero training
  demand). Same mechanism as static concentration (ParameterizedZoneOrderGenerator),
  but the target zone is verifiably unseen in training.
- S5 (event burst): `burst_share` of the orders with request_time inside
  [burst_start, burst_end] = [20, 30] min reassigned to `burst_zone 0`
  (centroid nodes, timestamps preserved). Same total volume, time-localized.
  Outside the window, demand is untouched base demand.

- Deprecated labels kept for backward-compat with old CSVs (all were the same
  static concentration-as-mechanism bug, now fixed):
  - S1a/S1b: static 70% concentration to zone 0 / zone 30 (not permutation).
  - S4a: static 60% concentration to zone 60 (not a held-out cold zone).
  - S5a: static 20% concentration to zone 0 (not a time burst).
  (Their CSVs were purged 2026-09-16 per D22/D23; definitions retained so
  the labels stay interpretable. The reporting labels are S1/S4/S5.)
- Measured cold zones (train split, winlim 500 sampling, 84 windows, 42000
  sampled orders, seed=42): zone 1 (44 orders, 0.105%), zone 3 (57), zone 40 (63).
  Hottest: zones 55/47/33 (~1563/1381/1287 each, ~3.7%/3.3%/3.1%). **S4 uses zone 1.**

## Results

PURGED 2026-09-16 (D22): all pre-D22 eval data deleted. S2c (magnitude
×1.0) stays dropped — byte-identical config to S0, pure compute waste.
Repopulate per `docs/shift-results.md` after v1.4 retraining.

## Fair-comparison checklist

- [ ] Same seeds and same scenario instances across all methods.
- [ ] Baseline tuned on B with effort budget comparable to RL agents.
- [ ] Report baseline under shift too — maybe the heuristic is the robust one.
- [ ] Report random policy as the floor, never as the "baseline".