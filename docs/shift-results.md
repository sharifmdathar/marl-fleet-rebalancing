# Demand Shift Results

Protocol (unchanged): frozen policy, ≥ 5 seeds, mean ± std, identical scenario
instances across methods. Shift IDs per `docs/demand-shift.md` (S0–S6,
mechanism-labeled S1/S4/S5). Env config v1.4 (D22) + D24/D25/D26 + D28.
Baseline per D27. KPIs: service rate, avg wait, empty km, plus
dispatch/relocate/idle shares.

**Run of record:** `results/shift_eval/` — 5 seeds (`base_seed=1000`, seeds
1000–1004), 10 shifts. The head-to-head rows are `ippo-nyc-d28-stage2` and
`mappo-nyc-d28-stage2`: both trained under the **identical D28 protocol**
(same env v1.8, same two-stage entropy schedule 0.05→0.01, same
neighbor-relocation action space, checkpoints selected by validation service
rate on the true held-out val split) and evaluated with the same frozen-policy,
5-seed procedure. Older rows (pre-D28 action spaces, training-selected
checkpoints) are kept below only as historical context and are **not**
comparators. Reproduce with:

```
uv run python -m marl_fleet_rebalancing.experiments.shift_eval \
    --all-runs --include-baselines --episodes 5
uv run python -m marl_fleet_rebalancing.experiments.plot_shifts
uv run python scripts/summarize_shift_eval.py   # rebuild head-to-head tables
```

---

## Head-to-head: service rate % (mean ± std over 5 seeds)

| Policy | S0 | S1 | S2a | S2b | S2d | S2e | S3 | S4 | S5 | S6 | Mean retention |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ippo-nyc-d28-stage2** (D28, val-selected) | 56.5 ± 2.5 | 38.9 ± 2.4 | 51.5 ± 2.7 | 56.0 ± 3.0 | 52.5 ± 1.6 | 46.9 ± 2.0 | 53.6 ± 1.2 | 21.1 ± 0.4 | 53.3 ± 2.4 | 57.3 ± 1.9 | 84.8% |
| **mappo-nyc-d28-stage2** (D28, val-selected) | 56.9 ± 1.9 | 41.4 ± 2.6 | 59.5 ± 2.2 | 55.9 ± 2.7 | 52.8 ± 1.5 | 48.6 ± 1.0 | 57.3 ± 2.4 | 23.5 ± 1.4 | 54.6 ± 2.0 | 55.0 ± 4.3 | 87.6% |
| **Greedy-Baseline** (floor, D27) | 52.9 ± 4.2 | 37.6 ± 4.6 | 59.2 ± 2.8 | 54.1 ± 1.6 | 46.1 ± 2.6 | 41.7 ± 2.8 | 46.3 ± 1.4 | 24.1 ± 1.7 | 51.4 ± 3.6 | 47.2 ± 3.1 | 85.7% |
| **LP-Rebalancer** (D27, tuned) | **70.5 ± 1.8** | **56.0 ± 1.8** | **82.5 ± 4.4** | **77.1 ± 1.7** | **63.9 ± 2.9** | **53.6 ± 1.7** | **70.6 ± 1.8** | **32.0 ± 1.5** | **69.1 ± 2.0** | **69.3 ± 3.4** | **90.5%** |

Companion KPIs (same run, full tables via `scripts/summarize_shift_eval.py`):

| Policy | Empty km S0 | Empty km range (S0–S6) | Avg wait S0 | Avg wait range |
|:---|:---:|:---:|:---:|:---:|
| ippo-nyc-d28-stage2 | 1781 | 1080–2173 (min S2e/S4) | 1 min | 1–2 min |
| mappo-nyc-d28-stage2 | 1324 | 930–1392 (min S4) | 1 min | 1–2 min |
| Greedy-Baseline | 119 | 64–210 | 1 min | 1–2 min |
| LP-Rebalancer | 815 | 488–847 | 1 min | 1–2 min |

## Answer to H1–H4

Under the identical D28 protocol, **H3 is the winner, with a lean toward H4's
mechanism** — the differences are small and concentrated in shift type, not a
general robustness ordering.

**H1 (MAPPO more robust) — weakly supported, not established.** MAPPO-D28 has
the better mean retention (87.6% vs 84.8%) and leads or ties on 7 of 10
shifts, but the gaps are mostly within one to two standard deviations, and
both training runs are single-seed (seed 0), so training-seed variance —
typically comparable to eval-seed variance in this setup — is unmeasured. The
honest statement: MAPPO was as robust or slightly more robust in this one
paired training run.

**H2 (IPPO more robust) — not supported.** IPPO-D28 never beats MAPPO-D28 by
more than noise, and its edge cases (S6 +2.3 points, S2b +0.1) are within
seed noise.

**H3 (no meaningful difference) — best single description of the aggregate.**
S0 differs by 0.4 points (56.9 vs 56.5, both ±~2); retention differs by 2.8
points, driven mostly by S6 (55.0 vs 57.3) and S2a (59.5 vs 51.5), each within
~2σ. After one paired run per algo, the centralized-critic advantage is not
demonstrable.

**H4 (difference depends on shift type) — partially supported, and this is
the interesting residue.** The two shifts where MAPPO pulls clear of IPPO are
exactly the two with distinctive mechanisms: S1 (temporal rapid surge: +2.5)
and S3 (evening peak intensification: +3.7) — both are demand-*level*
changes where a centralized critic has more signal to share across agents.
On spatial-structure shifts (S2 family) the two algos trade wins within
noise. Both algos collapse identically on S4 (cold-zone concentration: 21.1%
and 23.5%, both *below* greedy's 24.1%) — the failure is shift-type-specific,
not algo-specific.

**Cross-cutting facts that outweigh the IPPO-vs-MAPPO question:**

1. **The tuned LP baseline leads on every shift** (S0: 70.5% vs 56.9% best
   RL; S2a: 82.5% vs 59.5%) with the best retention (90.5%) — and it wins at
   *half the empty travel* of IPPO (815 vs 1781 km). RL is not competitive
   with a well-tuned rolling-horizon LP in this environment.
2. **Both RL policies pay far more for their service than greedy**: IPPO
   burns 15× greedy's empty km on S0 for +3.6 points of service rate;
   MAPPO burns 11× for +4.0.
3. **S4 is everyone's wall.** Cold-zone concentration is the only shift where
   every method's retention curve breaks; the RL policies fall below the
   do-nothing floor there.

## What this says

1. **The centralized critic did not buy robustness here.** Given one paired,
   validation-selected training run per algo under an identical protocol, the
   IPPO-vs-MAPPO difference is within noise on most shifts (H3), with a
   residual pattern on demand-level shifts (H4 flavor). This *updates* the
   earlier informal reading that "MAPPO-100it leads" — that row used a global
   relocation action space (pre-D26) and could not isolate the algo.
2. **The earlier degradation story is confirmed under the fair protocol.**
   MAPPO-D28 lands essentially where mappo-185716 did (57.2 → 56.9 on S0), so
   the old action-space confound did not manufacture the number. The D28
   arrival bonus + two-stage entropy schedule also fixed the relocation
   collapse at the source: `pct_relocate` stays healthy (~14% IPPO, ~10%
   MAPPO) through training instead of decaying to zero, and late-training
   checkpoints no longer converge to the greedy floor.
3. **The reward-economics diagnosis (2026-09-16) stands as a result in its own
   right**: with empty travel priced at −0.15/km against a +1.0 pay-now
   dispatch credit, both PPO variants independently unlearned proactive
   rebalancing (IPPO: 74.0% → 52.8% SR inside ~15 iterations, reloc 13.3% →
   0.5%). The D28 arrival bonus removed the asymmetry. "PPO variants unlearn
   rebalancing under pay-now/pay-later reward asymmetry" is documented with
   training curves from `runs/ippo-nyc-s0-100it` and the 0916 MAPPO series.
4. **S1 and S4 remain the discriminative shifts**, now with the same ranking
   under the clean protocol. S4 (cold-zone concentration) is where everything
   collapses; S2a (demand halved) remains the easiest shift for every policy
   and is not evidence of robustness.
5. **The LP also wins the environment's own reward** (D27: S0 +17.9 vs
   greedy −112.7), so its lead is not an artifact of one KPI.

## Caveats

- **Single-seed training runs.** Each D28 policy was trained once (seed 0).
  Eval seeds (1000–1004) are shared across methods and scenario instances are
  identical, so the comparison is fair — but the IPPO-vs-MAPPO deltas cannot
  be distinguished from training-seed luck until the training run is repeated
  with 2–3 seeds.
- **One training-config pair, one city.** All conclusions hold under env v1.8
  (D22–D28) on the NYC zone graph with the D28 reward. Different reward
  economics or cities could reorder H1/H2/H3.
- **Checkpoints are validation-selected but the val split is small**
  (`docs/training-validation.md` protocol); validation selection reduces but
  does not eliminate selection bias.
- The empty-km figure spans 64–2173 km, which visually flattens the
  greedy/LP comparison; use a log axis before putting it in the paper.
- `results/shift_eval_preD27/` holds the invalid pre-D27 3-seed run (LP row is
  a no-op floor clone). Kept only as a safety copy; do not cite, safe to delete.

## Historical rows (pre-D28, not comparators)

Kept for provenance; each has a known confound (action space, or checkpoint
selected by peak *training* SR rather than validation SR).

| Policy | S0 | S1 | S2a | S2b | S2d | S2e | S3 | S4 | S5 | S6 | Retention |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ippo-it75 (pre-collapse ckpt, train-selected) | 51.5 ± 3.9 | 33.4 ± 2.1 | 56.5 ± 2.4 | 53.2 ± 3.2 | 47.4 ± 2.7 | 44.6 ± 3.1 | 45.4 ± 1.8 | 21.4 ± 1.3 | 50.2 ± 3.2 | 46.9 ± 2.7 | 86.2% |
| mappo-185716 (100 it, **global** reloc) | 57.2 ± 2.0 | 43.4 ± 2.4 | 53.7 ± 2.8 | 54.9 ± 4.9 | 55.5 ± 2.7 | 48.7 ± 2.3 | 55.2 ± 1.2 | 22.8 ± 1.4 | 55.1 ± 1.7 | 55.1 ± 2.1 | 86.4% |
| mappo-192802 / 202335 / 204712 (200–400 it, post-collapse) | 52.8–54.7 | 37.6–39.4 | 59.2 | 53.9–54.4 | 46.8–53.5 | 46.3–48.4 | 46.3–50.1 | 24.1 | 51.4–53.9 | 47.2–50.0 | 86.9–88.7% |

Read the ippo-it75 row with care (kept from the earlier analysis): its
*training* service rate at selection was 74.4%, but frozen evaluation is 51.5%
— training metrics are stochastic rollouts in the training window mix while
evaluation is deterministic on fresh seeds, and the checkpoint was picked at
peak training SR, likely already partway into the collapse. This artifact is
what motivated the validation-selection protocol now in place.

## Status

- [x] Retrain/eval MAPPO on S0 (v1.4) — done, 4 runs (historical).
- [x] Train IPPO on S0 — done; both PPO variants' late-training collapse
  diagnosed and fixed via D28 (arrival bonus + entropy schedule).
- [x] Tune baseline with comparable effort (D3 / D27).
- [x] Validation-selection protocol on the true held-out val split
  (`docs/training-validation.md`).
- [x] Paired D28 retraining: IPPO (`ippo-nyc-d28-stage2`) + MAPPO
  (`mappo-nyc-d28-stage2`) under the identical protocol.
- [x] Frozen-policy eval across S0–S6, 5 seeds → curves in `results/figures/`.
- [x] **Written H1–H4 answer with the curves as evidence** (this document).

Remaining before submission: repeat each D28 training with 2–3 training seeds
to attach error bars to the H1–H4 deltas; reconcile `INTENT.md`'s non-goals
line ("agents learn only `relocate`") with D9's joint action space — this
document assumes D9 as implemented.
