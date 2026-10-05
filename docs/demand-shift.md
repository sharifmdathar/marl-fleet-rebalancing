# Demand Distribution Shift — taxonomy & protocol

> **Status 2026-10-05 (D83).** The study runs on **16 zones / 26 drivers /
> `nyc_window_limit=130` / 60-min episodes** (D65), not the 62-zone figures that
> survive further down this file. Zone-demand figures and scenario zone ids have
> been re-measured on the current 16-zone build (`results/zone_demand_train16.json`).
> Held-out evaluation covers the whole split since D70; before that S3 and S6
> each saw a single window and their numbers are void.
>
> **One scenario is no longer valid on this build: S4.** See "Why S4 needs
> rethinking" below — the 16 busiest zones have no cold zone.

Answers: "what exactly do we mean by demand distribution shift?" Each scenario
has an ID used in configs, logs, and figures. Base distribution B = the
training demand: NYC FHVHV train-split windows, `nyc_window_limit=130`
(D65), 26 drivers, 60-min episodes, 84 train / 12 val / 12 test windows.

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

**16 NYC taxi-zone centroids** falling inside the Region A bbox, chosen as the
16 busiest by train-window volume (D65; `scripts/build_zone_subset.py
--num-zones 16`). One zone-controller agent per zone (D10), action space 33,
`max_pending` 16. `BandraConfig` (synthetic grid) is kept only for quick local
testing.

LocationIDs, in subset index order (the indices the shift scenarios use):

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| zone | 68 | 79 | 112 | 145 | 161 | 162 | 163 | 164 |

| idx | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| zone | 170 | 230 | 231 | 234 | 236 | 237 | 246 | 249 |

Driver and order caps keep the 62-zone pressure ratios (1.61 drivers/zone,
5 orders/driver).

**62-zone artifacts are not comparable to 16-zone results** (D65). The 62-zone
build remains in `data/nyc/` as reference.

## Protocol (Fujimoto-style)

1. Train on B only. Freeze policy.
2. For each scenario ID and severity level: evaluate frozen policy over
   ≥ 5 seeds × N episodes.
3. Track KPIs **over episode time**, not just final values (degradation
   can be gradual, not a cliff).
4. Primary output: degradation curve = KPI vs shift severity, one line per
   method (IPPO / MAPPO / baseline / random).
5. Robustness metric: area-normalized performance retention =
   ∫ KPI(severity) d(severity) / KPI(S0). Still `<TODO>`.

### Preconditions before any shift run (D83)

These are not stylistic. Each one is a measurement this project already got
wrong once.

- **Freeze the evaluation policy definition first.** D72 found the trained
  (sampled) and evaluated (argmax) policies behave very differently at
  high beta — at beta 0.2 the argmax policy gave up 11 points of service
  relative to its own training rollouts. "The frozen policy" is not yet a
  settled question, and every number in this study inherits it.
- **Use the whole held-out split.** One episode per window, all 12 (D70).
  Before that fix, every evaluation saw window 0 only.
- **Service rate = completed / total**, not served / (served + cancelled).
- **Re-tune the LP on this environment and this build.** The re-tuned defaults
  are lookahead 3.0 / deficit penalty 4.0 (`scripts/tune_lp_objective.py`,
  chosen on train windows by objective R). Old defaults scored 53.9% where the
  tuned ones score 70.1%.
- **Do not train on a scenario used as an evaluation.** The two-zone
  competence test is a *diagnostic*, not a shift scenario, and no policy
  checkpoint reported in the log was trained on it.
- **Report IPPO and MAPPO from the same code.** As of D83 every IPPO checkpoint
  on disk predates the split-credit, chain-share, real-rows and window-coverage
  fixes, so there is currently **no valid IPPO arm**. Reporting a shift study
  with a MAPPO arm and a stale IPPO arm would compare two different
  environments.
- **≥5 seeds per reported cell.** Everything in `docs/decisions.md` through D82
  is 1 seed and labelled debugging output in the log itself.

## Implemented parameterizations (Phase 4, 2026-09-14; zone ids corrected D83)

Concrete settings in `experiments/shift_eval.py` (frozen policy,
`base_seed=1000`, `_WINDOW = 130`):

- S3 (temporal phase shift): `nyc_split="val"` — evaluation windows drawn from
  the validation split (different hours than the train split). Same order cap
  (130) as S0 so only timing changes. **Since D70 this covers all 12 val
  windows, not one.**
- S5 (event-like spike, D21 mechanism): `burst_share` of orders with
  request_time in [20, 30) min reassigned to `burst_zone`, timestamps preserved.
  Same total volume, time-localized.
- S6 (real held-out demand): `nyc_split="test"` — real FHVHV test-split windows
  never touched during training or tuning. All 12 windows since D70.
- S1 (spatial permutation): `perm_offset = num_zones // 2` = **8** on this
  build, so the rotation is idx → idx+8 mod 16. The 62-zone value was 31.

**Zone ids in the scenario table still need updating.** `shift_eval.py` carries
`S4 target_zone=1` and `S5 burst_zone=0` from the 62-zone build; on 16 zones
those are idx 1 (zone 79) and idx 0 (zone 68). Harmless as indices, but the S4
label is wrong for the reason above, and a burst into idx 0 (6.87%, near
average) is a weaker event than the 62-zone version.

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

**Caveat that survives all three redefinitions (D69).** The decision-time aim
check does not measure service. 70% of dispatches of a relocated car happen
*outside* the zone it was sent to, because mid-flight dispatch serves nearby
demand on the way — and forbidding it collapses service (D69: SR 32.8% vs
64.6%). So "aimed at the best legal zone" and "service improved" are different
claims. A scenario that moves a hotspot still gets followed by a policy that
aims well *and* by one that aims well but is scored on where cars land, so the
protocol must report service KPIs, not aim, as the headline.
- Measured per-zone train demand on the **16-zone** build (all 84 train
  windows, 304,393 orders, no sampling; `results/zone_demand_train16.json`):

  | idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | zone | 68 | 79 | 112 | 145 | 161 | 162 | 163 | 164 |
  | share | 6.87% | 5.99% | 6.02% | 5.03% | 7.33% | 4.84% | 5.41% | 5.63% |

  | idx | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
  |---|---|---|---|---|---|---|---|---|
  | zone | 170 | 230 | 231 | 234 | 236 | 237 | 246 | 249 |
  | share | 6.08% | 5.75% | 7.80% | 7.10% | 5.84% | 6.87% | 8.29% | 5.17% |

  Hottest **idx 14 (zone 246), 8.29%**. Coldest **idx 5 (zone 162), 4.84%**.
  Spread is only **1.7×**, and max single-zone share is 8.29%.

  This reproduces D73's independent observation (max single-zone share 9–12% per
  window against 90–100% in the two-zone test), and it is the fact that breaks
  S4.

## Why S4 needs rethinking

S4 is defined as concentration into a zone that is **verifiably cold in the
train split** — the point being that the policy was never rewarded there, so
moving demand there tests genuine out-of-distribution response rather than
re-testing a hotspot the policy already knows.

**That construction is unavailable on the 16-zone build.** The subset was
selected as the 16 *busiest* zones, so by construction there is no near-zero
zone in it: the coldest is 4.84% against a mean of 6.36%. The 62-zone build
had zone 1 at 0.105% — genuinely cold — which is why S4 worked there and why
the old text said "S4 uses zone 1".

Concentrating 60% of demand into idx 5 is therefore **not** a cold-zone test. It
is a 12× concentration into a mildly-below-average zone, and a policy that
learned "go where demand is" would score well on it without having demonstrated
anything about unseen demand.

Options, in order of preference:

1. **Re-run the whole study on the 62-zone build.** Restores S4 as defined, and
   the reason it was dropped was compute (D65), not principle. Revisit if the
   advisor requires full scale.
2. **Redefine S4 as a magnitude-and-location shift**: concentrate into idx 5
   *and* report it as such — "shift into a below-average zone", not "cold
   zone". Honest, but weaker than the original claim.
3. **Construct the cold zone by holdout**: exclude one zone's orders from
   training demand, then restore them at evaluation. This makes the zone
   genuinely unseen and is the faithful version of S4 on a subset that has no
   cold zone. **Not implemented; it changes the environment definition, so it
   needs a decision before it is used.**

Until one of these is chosen, **S4 must not be reported as a cold-zone
result.** `shift_eval.py` still hardcodes `target_zone=1` from the 62-zone
build, so running it today would silently test idx 1 (zone 79, 5.99%) and label
it "cold".

## Results

PURGED 2026-09-16 (D22): all pre-D22 eval data deleted. S2c (magnitude
×1.0) stays dropped — byte-identical config to S0, pure compute waste.
Regenerate from a fresh eval run; the pre-env-2.3 tables were deleted.

## Fair-comparison checklist

- [ ] Same seeds and same scenario instances across all methods.
- [ ] Baseline tuned on B with effort budget comparable to RL agents.
- [ ] Report baseline under shift too — maybe the heuristic is the robust one.
- [ ] Report random policy as the floor, never as the "baseline".