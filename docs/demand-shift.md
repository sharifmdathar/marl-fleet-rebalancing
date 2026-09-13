# Demand Distribution Shift — taxonomy & protocol

Answers: "what exactly do we mean by demand distribution shift?" Each scenario
has an ID used in configs, logs, and figures. Base distribution B = the
training demand (current: Poisson arrivals on Bandra West network, 400 orders /
30 min — treat as placeholder until we fix the base scenario).

## Shift taxonomy

| ID | Type | Parameterization | Real-world story |
|----|------|------------------|------------------|
| S0 | In-distribution control | B resampled (new seeds) | Sanity check: no shift |
| S1 | Spatial permutation | Rotate/permute origin-destination intensities across zones, strength p ∈ [0,1] | Hotspots move (new mall, office district) |
| S2 | Magnitude scaling | Global arrival rate × λ, λ ∈ {0.5, 0.75, 1.25, 1.5, 2.0} | Demand surge / off-season |
| S3 | Temporal phase shift | Peak window shifted by Δt minutes | Schedule change, daylight saving, Ramadan |
| S4 | New hotspot emergence | k new hotspot nodes with intensity h, unseen in training | Concert venue, new station opens |
| S5 | Event-like spike | Short high-intensity burst at random node/time | Match ending, sudden rain |
| S6 | Real held-out demand | Different day/window of real trip data | The non-synthetic case |

Mapping to the prof's four shift families (D12): higher demand → S2;
spatial shift → S1 (plus S4 hotspot emergence); temporal shift → S3;
event-type shift → S5. S0 is the in-distribution control.

## Zone partition over the Bandra West graph

The Bandra West OSMnx graph has 1464 nodes and 3281 edges (EPSG:4326).
We divide the graph into **12 zones** using a **spatial grid partition**:

- The graph's bounding box is partitioned into a 3×4 grid of rectangular zones
- Each zone contains approximately equal numbers of graph nodes (≈122 nodes/zone)
- Zone IDs are assigned based on the centroid latitude/longitude of each node
- This partition is deterministic, reproducible, and serves as the abstraction
  layer for S1 (spatial permutation) and S4 (new hotspot emergence)
- The zone mapping is persisted as `data/zone_partition.json` (node_id → zone_id)
- Order generators reference zones rather than individual graph nodes for
  demand injection, ensuring shifts operate at the zone level

<S1–S5 assume a zone/node abstraction over the OSMnx graph; define the
zone count and method. Zone count: 12 zones via 3×4 spatial grid partition.>

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
- S5 (event-like spike): `target_zone=0`, `target_zone_demand_pct=20` on the
  train split — a mild concentrated burst, distinct from S1's sustained 70%
  concentration and S4's 60% unseen hotspot.
- S6 (real held-out demand): `nyc_split="test"` — real FHVHV test-split
  windows never touched during training or tuning.

## Results

Phase 4 evaluation (2026-09-13/14) for all 11 shifts is recorded in
`docs/shift-results.md` (raw data: `results/shift_eval/summary.json`).
S2c (magnitude ×1.0) was dropped 2026-09-14: byte-identical config to S0,
verified reproducing S0 exactly — pure compute waste.

## Fair-comparison checklist

- [ ] Same seeds and same scenario instances across all methods.
- [ ] Baseline tuned on B with effort budget comparable to RL agents.
- [ ] Report baseline under shift too — maybe the heuristic is the robust one.
- [ ] Report random policy as the floor, never as the "baseline".