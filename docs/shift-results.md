# Demand Shift Results

Protocol (unchanged): frozen policy, ≥ 5 seeds, mean ± std, identical scenario
instances across methods. Shift IDs per `docs/demand-shift.md` (S0–S6,
mechanism-labeled S1/S4/S5). Env config v1.4 (D22) + D24/D25/D26. Baseline
per D27. KPIs: service rate, avg wait, empty km, plus dispatch/relocate/idle
shares.

**Run of record:** `results/shift_eval/` — 5 seeds (`base_seed=1000`, seeds
1000–1004), 10 shifts, 6 policies. Reproduce with:

```
uv run python -m marl_fleet_rebalancing.experiments.shift_eval \
    --all-runs --include-baselines --episodes 5
uv run python -m marl_fleet_rebalancing.experiments.plot_shifts
```

---

## Head-to-head: service rate % (mean ± std over 5 seeds)

| Policy | S0 | S1 | S2a | S2b | S2d | S2e | S3 | S4 | S5 | S6 | Mean retention |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **mappo-185716** (100 it, global reloc) | 57.2 ± 2.0 | 43.4 ± 2.4 | 53.7 ± 2.8 | 54.9 ± 4.9 | 55.5 ± 2.7 | 48.7 ± 2.3 | 55.2 ± 1.2 | 22.8 ± 1.4 | 55.1 ± 1.7 | 55.1 ± 2.1 | 86.4% |
| **mappo-192802** (200 it, neighbor) | 54.7 ± 4.8 | 39.4 ± 4.8 | 59.2 ± 2.8 | 54.4 ± 1.7 | 53.5 ± 2.1 | 48.4 ± 2.7 | 50.1 ± 4.2 | 24.1 ± 1.7 | 53.9 ± 3.7 | 50.0 ± 4.7 | 88.0% |
| **mappo-202335** (400 it, neighbor) | 53.1 ± 4.0 | 39.0 ± 4.9 | 59.2 ± 2.8 | 53.9 ± 1.6 | 51.7 ± 1.1 | 47.7 ± 1.4 | 47.9 ± 3.1 | 24.1 ± 1.7 | 52.0 ± 3.4 | 48.2 ± 3.7 | 88.7% |
| **mappo-204712** (400 it, neighbor) | 52.8 ± 4.2 | 37.6 ± 4.6 | 59.2 ± 2.8 | 54.1 ± 1.6 | 46.8 ± 3.0 | 46.3 ± 1.7 | 46.3 ± 1.4 | 24.1 ± 1.7 | 51.4 ± 3.7 | 47.2 ± 3.1 | 86.9% |
| **Greedy-Baseline** (floor) | 52.9 ± 4.2 | 37.6 ± 4.6 | 59.2 ± 2.8 | 54.1 ± 1.6 | 46.1 ± 2.6 | 41.7 ± 2.8 | 46.3 ± 1.4 | 24.1 ± 1.7 | 51.4 ± 3.6 | 47.2 ± 3.1 | 85.7% |
| **LP-Rebalancer** (D27, tuned) | **70.5 ± 1.8** | **56.0 ± 1.8** | **82.5 ± 4.4** | **77.1 ± 1.7** | **63.9 ± 2.9** | **53.6 ± 1.7** | **70.6 ± 1.8** | **32.0 ± 1.5** | **69.1 ± 2.0** | **69.3 ± 3.4** | **90.5%** |

Empty km / episode and avg wait are in `results/shift_eval/degradation_table.md`.
Figures: `results/figures/degradation_{service_rate,empty_km,avg_wait_min}.png`.

## What this says

1. **The tuned baseline leads on every shift.** LP beats the best MAPPO by
   13.3 points on S0 and by 12.6–23.3 points on every other shift, at the
   highest retention (90.5%). No RL policy is currently competitive with a
   well-tuned rolling-horizon LP. This is the headline fact, not a footnote.
2. **The baseline's advantage is bought with repositioning.** LP burns 815 km
   of empty travel on S0 versus 119 km for greedy; MAPPO-185716 burns 1829 km
   for 13 fewer points of service rate, i.e. it relocates harder than the LP
   and gets much less for it.
3. **More training made it worse, and turned relocation off.** 100 → 400
   iterations drops S0 service rate 57.2% → 52.8%, and `pct_relocate` falls
   6.22% → 0.01%. The 400-iteration runs are effectively greedy dispatchers
   with a learned dispatch choice: D26's neighbor-only relocation plus D24's
   surplus-target masking leave relocation rarely available, so the policy
   learned to stop using it. Retention looks slightly *better* (86.9% vs
   86.4%) only because the in-distribution starting point fell.
4. **S1 and S4 are the discriminative shifts.** Spatial permutation costs the
   best MAPPO 13.8 points and greedy 15.3; S4 (cold-zone concentration) is
   where everything collapses (LP 32.0%, MAPPO 22.8–24.1%). S2a (demand
   halved) is the easiest for every policy — fewer orders means less shift
   pressure, so it is not evidence of robustness.
5. **The LP also wins the environment's own reward** (D27: S0 +17.9 vs
   greedy −112.7), so its lead is not an artifact of one KPI.

## Caveats and blocked work

- **No IPPO checkpoints exist** (`runs/` holds 4 MAPPO runs, all seed 0). H1–H4
  cannot be answered yet; the tables above are MAPPO-vs-baselines only.
- All MAPPO runs are **single-seed training** (seed 0). Eval seeds vary, but
  training-seed variance is unmeasured.
- The 400-iteration pair (202335, 204712) look like duplicates by parameter
  probes; confirm whether they differ before reporting them as two runs.
- The empty-km figure spans 64–2151 km, which visually flattens the
  greedy/LP comparison; consider a log axis before using it in the paper.
- `results/shift_eval_preD27/` holds the invalid pre-D27 3-seed run (LP row is
  a no-op floor clone). Kept only as a safety copy; do not cite, safe to delete.
- `INTENT.md` still says "dispatch is a fixed rule; agents learn only
  `relocate` (pending D5)", but D9 superseded D5 and the env implements joint
  dispatch+relocate. Reconcile before writing H1–H4.

## Status

- [x] Retrain/eval MAPPO on S0 (v1.4) — done, 4 runs.
- [ ] Train IPPO on S0 — **missing, blocks H1–H4**.
- [x] Tune baseline with comparable effort (D3 / D27).
- [x] Frozen-policy eval across S0–S6, 5 seeds → curves in `results/figures/`.
- [ ] Written H1–H4 answer with the curves as evidence.
