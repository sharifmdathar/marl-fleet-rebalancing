# LP Rebalancer Zero-Transfer Diagnosis

## When to check

- The `ZoneLPRebalancer` policy returns action 0 (no-op) on every step.
- Baseline evaluation shows identical performance to the greedy dispatch baseline.
- Shift evaluation results show the LP and greedy baselines have indistinguishable KPIs.

## Root cause: the LP objective had no value on filling shortfalls

The LP minimised travel cost subject only to *upper bounds*:

```
minimise  sum_ij c_ij x_ij
s.t.      sum_j x_ij <= surplus_i ,  sum_i x_ij <= deficit_j ,  sum_ij x_ij <= n_idle
```

Every `c_ij >= 0` (the diagonal was 0) and every constraint was `<=`, so `x = 0`
was feasible *and* optimal. The solver was firing and solving correctly — it was
just always right to move nothing. This is independent of any tunable knob:

| Knob | Effect on the zero plan |
|---|---|
| `speed_kmh` | None. It only rescales costs, and `x = 0` stays optimal at any scale. |
| `min_idle_to_rebalance` | None. It only gates whether the LP runs; the plan is still zero. |
| `lookahead_minutes` | None. It only changes the *magnitudes* of predicted demand. |

Verified directly: with `imbalances = [-50, -50, +100, +100]`, `n_idle = 100` and
uniform unit costs, `solve_rebalance_lp` returned an all-zero plan at cost scales
0.01, 1.0 and 100.

## Fix (D27)

Give the objective a term for the shortfall it fails to cover:

```
minimise  sum_ij c_ij x_ij  -  lambda * sum_j (sum_i x_ij)
lambda = deficit_penalty_scale * mean(off-diagonal c)
```

The reduced cost `c_ij - lambda` is the net value of a transfer, so a route is
used only when filling the destination's predicted shortfall is worth more than
the travel time it costs. `deficit_penalty_scale = 0` reproduces the old no-op
behaviour exactly (the solver call is skipped).

Two supporting fixes came with it:

- **`_update_demand_rates` units.** `env._pending_per_zone` is pending orders per
  zone *divided by fleet size* (D22 normalisation), but the rate estimate
  multiplied it by the total pending count, mixing units and shrinking every
  zone's estimate by `total_pending / num_drivers`. It now scales by
  `num_drivers`.
- **Reproducibility.** Fractional LP flows were rounded with the global
  `np.random` stream, so the same seed produced different plans between runs.
  The policy now owns a generator seeded from `reset(seed)`.

## Parameter sweep (S0 train + S3 val only, 5 seeds, base_seed 1000)

The reference floor for the same runs: greedy, 52.9% / 46.3% service rate,
115 km empty.

| lookahead | penalty scale | reloc % | empty km | SR S0 | SR S3 |
|---|---|---|---|---|---|
| 5 | 0.3 | 5.9 | 610 | 69.6% | 67.0% |
| **5** | **0.4** | **7.2** | **812** | **70.5%** | **70.6%** |
| 8 | 0.3 | 6.3 | 641 | 68.1% | 66.6% |
| 8 | 0.4 | 7.6 | 841 | 70.9% | 69.3% |

Adopted defaults: `lookahead_minutes = 5.0`, `deficit_penalty_scale = 0.4`.

Two findings worth keeping straight in the writeup:

1. **The horizon is not a "bigger is better" knob.** The intuitive fix — pick a
   horizon longer than the mean inter-zone travel time (8.6 min) — made things
   *worse*. Predicted demand grows linearly with the horizon while idle supply
   does not, so a 15–30 min horizon flags ~50 of 62 zones as short and the plan
   loses all spatial selectivity. 5 min won on both service rate and empty km.
2. **The service-rate gain is paid for in empty km — but it still wins.**
   Relocating costs 5–7× greedy's 115 km. This is not phantom shuttling: the
   extra distance buys more served orders than it costs, so the tuned LP also
   beats greedy on the environment's own reward (S0 +17.9 vs −112.7, S1 −139.1
   vs −276.8, S4 −331.9 vs −475.8, S6 +9.0 vs −184.0). Report empty km and
   reward alongside the service rate; do not quote the service rate alone.
   Average wait rises slightly (S0 1.05 vs 0.77 min) — buses in transit are not
   available for nearby pickups.

## Speed guidance is inverted — do not lower `speed_kmh`

The cost matrix is travel *time*: `c_ij = dist_km / speed_kmh * 60`. Lowering
`speed_kmh` **raises** every cost entry and makes the LP *less* willing to
relocate. Conversely, raising it would cheapen relocation, but that is the wrong
direction too: if the environment moves vehicles at 30 km/h, an LP that assumes
40 km/h under-prices every plan and over-relocates onto trips that arrive late.
Keep `speed_kmh` at the environment's true value and spend the tuning budget on
`deficit_penalty_scale`, which is an honest preference parameter rather than
fudged physics.

## Verification

- `pct_relocate > 0` on S0 (7.2% with the adopted defaults; was 0.0%).
- LP >= greedy service rate on every tuning shift (70.5/70.6 vs 52.9/46.3).
- `tests/test_baselines.py` asserts `deficit_penalty=0` is a no-op and that a
  penalty above the cheapest route cost moves exactly the surplus-constrained
  amount.

## Held-out shifts (5 seeds, scored once, after the defaults were frozen)

| Policy | S0 | S1 | S2a | S2b | S2d | S2e | S3 | S4 | S5 | S6 | Mean retention |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Greedy | 52.9% | 37.6% | 59.2% | 54.1% | 46.1% | 41.7% | 46.3% | 24.1% | 51.4% | 47.2% | 85.7% |
| LP (D27) | 70.5% | 56.0% | 82.5% | 77.1% | 63.9% | 53.6% | 70.6% | 32.0% | 69.1% | 69.3% | 90.5% |

Full KPIs in `results/shift_eval_lp_fixed/`. The LP now dominates greedy on
service rate in-distribution *and* on every held-out shift, with the largest
retention advantage on S1 (79.4% vs 71.1%), the spatial-permutation shift where
myopic dispatch is most exposed. Any earlier IPPO/MAPPO-vs-LP comparison is
invalid: the floor moved by 16+ points.

## Still open

- Why does the LP keep relocating at 7% of actions when 45 of 62 zones are
  nominally short? The plan ignores the busy fleet that frees up inside the
  horizon, so the shortfall estimate is probably biased upward. Worth one pass
  before treating 812 km as the baseline's honest operating point.
- `lookahead`/penalty were only swept on one demand level. Re-check the defaults
  if the fleet size or `nyc_window_limit` changes.
- Tuning runs live in `results/lp_tuning*/`; they are 3–5 seed, and the adopted
  defaults were chosen from the 5-seed table above.

## Related references

- `references/baselines.md` — Non-RL baselines for demand-shift evals (greedy/random design intent, KPI interpretation rules)
- `references/routing-and-measurement.md` — Single-switch routing guard-ordering pitfall, cold-zone measurement recipe
- `software-development/demand-shift-evaluation/SKILL.md` — Demand-shift evaluation protocol, always-on rules, mechanism verification
