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
| **ippo-it75** (pre-collapse, see below) | 51.5 ± 3.9 | 33.4 ± 2.1 | 56.5 ± 2.4 | 53.2 ± 3.2 | 47.4 ± 2.7 | 44.6 ± 3.1 | 45.4 ± 1.8 | 21.4 ± 1.3 | 50.2 ± 3.2 | 46.9 ± 2.7 | 86.2% |
| **mappo-185716** (100 it, global reloc) | 57.2 ± 2.0 | 43.4 ± 2.4 | 53.7 ± 2.8 | 54.9 ± 4.9 | 55.5 ± 2.7 | 48.7 ± 2.3 | 55.2 ± 1.2 | 22.8 ± 1.4 | 55.1 ± 1.7 | 55.1 ± 2.1 | 86.4% |
| **mappo-192802** (200 it, neighbor) | 54.7 ± 4.8 | 39.4 ± 4.8 | 59.2 ± 2.8 | 54.4 ± 1.7 | 53.5 ± 2.1 | 48.4 ± 2.7 | 50.1 ± 4.2 | 24.1 ± 1.7 | 53.9 ± 3.7 | 50.0 ± 4.7 | 88.0% |
| **mappo-202335** (400 it, neighbor) | 53.1 ± 4.0 | 39.0 ± 4.9 | 59.2 ± 2.8 | 53.9 ± 1.6 | 51.7 ± 1.1 | 47.7 ± 1.4 | 47.9 ± 3.1 | 24.1 ± 1.7 | 52.0 ± 3.4 | 48.2 ± 3.7 | 88.7% |
| **mappo-204712** (400 it, neighbor) | 52.8 ± 4.2 | 37.6 ± 4.6 | 59.2 ± 2.8 | 54.1 ± 1.6 | 46.8 ± 3.0 | 46.3 ± 1.7 | 46.3 ± 1.4 | 24.1 ± 1.7 | 51.4 ± 3.7 | 47.2 ± 3.1 | 86.9% |
| **Greedy-Baseline** (floor) | 52.9 ± 4.2 | 37.6 ± 4.6 | 59.2 ± 2.8 | 54.1 ± 1.6 | 46.1 ± 2.6 | 41.7 ± 2.8 | 46.3 ± 1.4 | 24.1 ± 1.7 | 51.4 ± 3.6 | 47.2 ± 3.1 | 85.7% |
| **LP-Rebalancer** (D27, tuned) | **70.5 ± 1.8** | **56.0 ± 1.8** | **82.5 ± 4.4** | **77.1 ± 1.7** | **63.9 ± 2.9** | **53.6 ± 1.7** | **70.6 ± 1.8** | **32.0 ± 1.5** | **69.1 ± 2.0** | **69.3 ± 3.4** | **90.5%** |

**Read the ippo-it75 row with care.** It is the frozen iteration-75 checkpoint
(the pre-collapse peak window of `ippo-nyc-s0-95it-ck5`), evaluated on the
shift suite. Its *training* service rate at that point was 74.4% — but its
frozen evaluation S0 is only 51.5%, below greedy. Two things explain the gap:
training metrics are measured on stochastic rollouts in the training window
mix, while evaluation is deterministic on fresh seeds; and the checkpoint was
selected by peak training SR, not by validation SR, so it may already sit
partway into the collapse. The honest statement is: **no IPPO checkpoint has
yet been produced whose frozen evaluation beats greedy**, while MAPPO-100it
(57.2%) does. Whether that is an IPPO-vs-MAPPO difference or a
checkpoint-selection artifact cannot be settled until early stopping selects
checkpoints by validation performance for both algos. Until then, H1-H4 have
no answer, and the ippo-it75 row is a data point, not a comparison.

Empty km / episode and avg wait are in `results/shift_eval/degradation_table.md`.
Figures: `results/figures/degradation_{service_rate,empty_km,avg_wait_min}.png`.

## What this says

1. **The tuned baseline leads on every shift.** LP beats the best RL policy by
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
   with a learned dispatch choice. Retention looks slightly *better* (86.9% vs
   86.4%) only because the in-distribution starting point fell.

   **Diagnosis (2026-09-16, from metrics.jsonl):** this is not an exploration
   or masking failure. A greedy walk of the v1.8 env shows relocation is
   *always* available when it could matter: 55% of zone-turns hold ≥1 idle
   driver, and every one of those turns has a legal relocation target (mean
   5.7 of 8 neighbors pass the D24 surplus test). The collapse is learned and
   driven by the environment's own economics: D25 prices empty travel at
   −0.15/km (≈ −0.75 per 5 km hop) while the immediate +1.0 dispatch credit
   pays *now*, so a relocation costs more than half a served order with only
   speculative future payoff. The training metrics show it happen gradually:
   `pct_relocate` decays 14.6% → 3.1% (200 it) → 0.3% (400 it) as service rate
   simultaneously falls 73.6% → 49.7% — the policy first strips out relocation,
   then loses the demand-anticipation value that relocation was supporting.

   **Update (same day): this is NOT MAPPO-specific and NOT slow.** IPPO on the
   identical env (`runs/ippo-nyc-s0-100it`) trained *healthily for ~80
   iterations* — 74.0% training service rate at 13.5% relocation, the best
   training numbers of any run — then **collapsed to the same attractor inside
   ~15 iterations**: iteration 81 SR 74.0% / reloc 13.3% → iteration 91 SR
   63.3% / reloc 4.9% → iteration 100 SR 52.8% / reloc 0.5%. Both algos, same
   env, same failure mode: once the policy discovers that suppressing empty
   travel yields an immediate reward improvement, PPO's advantage estimates
   push every agent off relocation and there is nothing left to stabilize the
   coordination value it was providing. The end state (SR 52.8%, empty 172 km)
   is exactly the greedy-dispatch floor.

   Implication for RQ2: any late-training checkpoint is a policy that has
   *unlearned* rebalancing, so "train longer" is anti-productive under the
   current reward. Options, in order of cost: (a) **early stopping** — select
   the checkpoint at peak *validation* service rate (this run: ~it 41–81);
   note the saved `checkpoint.pt` is the final iteration, so retrain with
   `--checkpoint-interval 5` and pick, or add best-checkpoint tracking to
   `train.py`; (b) reshape the relocation payoff so the delay between paying
   for travel and collecting the service is shorter or the travel discount
   smaller — this changes the reward function and therefore requires a
   decisions.md entry plus config version bump (AGENTS.md); (c) report the
   collapse itself as a finding — "both PPO variants unlearn proactive
   rebalancing under a pay-now/pay-later reward asymmetry" is a legitimate
   RQ2-adjacent result if documented with the training curves. Do not report
   late-training checkpoints as either algo's best effort.
4. **S1 and S4 are the discriminative shifts.** Spatial permutation costs the
   best MAPPO 13.8 points and greedy 15.3; S4 (cold-zone concentration) is
   where everything collapses (LP 32.0%, MAPPO 22.8–24.1%). S2a (demand
   halved) is the easiest for every policy — fewer orders means less shift
   pressure, so it is not evidence of robustness.
5. **The LP also wins the environment's own reward** (D27: S0 +17.9 vs
   greedy −112.7), so its lead is not an artifact of one KPI.

## Caveats and blocked work

- **The first IPPO row exists but is not a fair comparator yet.** Both PPO
  variants collapse late in training (see finding 3); `ippo-it75` was selected
  by peak *training* SR, not validation SR, and evaluates below greedy despite
  a 74.4% training SR. Fair H1–H4 requires validation-selected checkpoints for
  both algos (train.py now writes versioned `checkpoint_it####.pt` files, but
  has no best-checkpoint tracking yet).
- **MAPPO's fair comparator is also missing.** mappo-185716 (100 it) predates
  the collapse and is the best RL row, but it was trained under
  `neighbor_relocation_only=False` (global relocation) — a different action
  space from the 200/400-it runs and from ippo-it75. A MAPPO run under the
  current v1.8 env with early stopping is needed before any IPPO-vs-MAPPO
  claim.
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
- [x] Train IPPO on S0 — done (`ippo-nyc-s0-95it-ck5`); **but** both PPO
  variants collapse late in training, so checkpoint selection by validation SR
  is now the blocking item for H1–H4.
- [x] Tune baseline with comparable effort (D3 / D27).
- [x] Frozen-policy eval across S0–S6, 5 seeds → curves in `results/figures/`.
- [ ] Written H1–H4 answer with the curves as evidence.
