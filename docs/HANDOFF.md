# HANDOFF — MARL fleet rebalancing

**Read `docs/decisions.md` for the full decision record (D35–D64). This file is
the map: where things stand, what was tried, what broke, and what to do next.**

Everything below was measured in this repo. Numbers are labelled
`MEASURED` / `INFERRED` / `TODO`. If a claim here has no measurement attached,
treat it as a hypothesis.

---

## 1. Bottom line

**There is no usable model yet. Do not start the demand-shift study.**

The two RL arms are **dominated by the LP baseline** on the efficiency metric
that `AGENTS.md` requires alongside service rate.

| arm | SR% | empty km | SR/100km |
|---|---|---|---|
| greedy | 52.73 | 78.1 | **68.0** |
| **LP** | **67.29** | **154.3** | **43.7** |
| IPPO | 70.95 | 483.9 | 14.7 |
| MAPPO | 74.99 | 468.8 | 16.0 |

`scripts/eval_tradeoff.py`, 6 paired seeds (12000–12005), val split, argmax
actions. LP gets 67.29% at 154 km; MAPPO gets 74.99% at 469 km — **3.0× the
deadhead for +7.7pp. RL is 2.7× worse than LP per empty km.**

Measuring degradation with this would measure a policy the baseline already
beats. That is the user's explicit stop condition.

---

## 2. The one defect that blocks everything

**D63 Cause 1, unfixed.** The relocating zone is paid a **flat 0.65 regardless
of whether it sends a car to a zone with 1 rider or 10.**

MEASURED, D62 IPPO, dispatch rate by aim class:

| aim class | aims | paid | rate |
|---|---|---|---|
| best legal target | 148 | 84 | **56.8%** |
| other non-empty zone | 91 | 49 | **53.8%** |
| empty zone | 167 | 55 | **32.9%** |

```
gap best vs other-nonempty :  +2.9 pp   <- RANKING signal: missing
gap non-empty vs empty     : +20.9 pp   <- EMPTINESS signal: learned
```

The reward is **7× more sensitive to "did you aim somewhere empty" than to "did
you aim at the best one".** MAPPO is worse (+0.2 pp ranking).

So RL learns half the decision and cannot learn the other half. That is why
MAPPO reaches **49%** precision while a supervised probe on **identical
features** reaches **87.5%** (MEASURED, `scripts/diag_target_learnability.py`,
7.5–9.6× chance across 4 seeds).

**The policy is being paid to move cars without being told where.**

### Proposed fix, not yet implemented

Scale the relocator credit by the deficit it relieved:

```
credit = relocator_share * min(pending_at_target / reference, 1)
```

- Keep the existing **dispatch proof-gating** — the credit still fires only when
  the relocated car is actually dispatched. This is *not* D28's arrival bonus,
  which paid for the arrival itself and was switched off for paying without
  proof.
- `reference` should be **measured**, not guessed: mean demand at the best
  available target is **1.69–1.77 riders** (MEASURED, D62).
- **Risk to watch:** over-aggressive scaling makes the policy chase one hotspot
  and neglect breadth. Test against the current 49% and the 87% ceiling.

---

## 3. What actually works

**Targeting went from broken to good.** MEASURED, argmax, decision-time
(`scripts/diag_targeting_at_decision.py`):

| stage | precision | chance | lift |
|---|---|---|---|
| D51 (pooled encoder) | 6.40% | 8.40% | 0.76× — *below chance* |
| **D64 MAPPO** | **49.15%** | 9.03% | **5.44×** |
| D64 MAPPO it35 | 49.08% | 8.00% | **6.14×** |
| D64 IPPO | 10.24% | 8.18% | 1.25× — **regressed** |

D64 MAPPO is robust across iterations (it10 44.37%, it30 38.98%, it35 49.08%),
so it is not one lucky checkpoint. **One seed each** — `AGENTS.md` requires ≥5
seeds for a *reported* result, so this is debugging output.

**IPPO regressed** 37.19% → ~13% consistently. Most likely arm-specific: the
higher relocator credit rewards relocating more, MAPPO's centralized critic can
evaluate whether a relocation will pay so it learns to aim, IPPO's weaker critic
cannot so the credit buys indiscriminate relocating. **Unproven — needs seeds.**

---

## 4. Measurement discipline — read before trusting any number

**Four of my own reported numbers were wrong**, each caught by testing a
premise rather than accepting a conclusion. Do not repeat them.

| wrong number | correct value | why |
|---|---|---|
| relocator margin 3.01× | **2.54×** | hardcoded 2.49 km leg; re-measured 2.95 km |
| relocation chance 4.1% | **8.4%** (13.5 legal targets, not 1.12) | confounded by mid-journey dispatch; D44 predated the D45 mask change |
| "network ignores demand" | demand reaches logits, L1 2.9–15.3 | my probe wrote an unreachable observation state |
| targeting 13–20% | **51–55%** at argmax | `deterministic=False` in the test — sampling discarded most of the policy |

**Rules that follow:**
- Always pass `--deterministic` for decision-quality measurements. It is
  recorded in the JSON artifact (`action_selection`).
- **Legal targets are ~13.5, not 1.12.** Chance is `1 / mean legal targets`,
  recomputed per run.
- Evaluate **paired** on identical seeds (`docs/eval-noise.md`).
- Use `scripts/watch_plateau.py` to detect plateau rather than running to the
  nominal iteration count.
- The competence test (`scripts/competence_two_zone.py`) concentrates all demand
  in one zone and is **near-degenerate** — greedy scores 0.8%. It is a targeting
  probe, not a performance benchmark.

---

## 5. Reward function as it stands

```
R_zone = 0.8 * R_local + 0.2 * mean(all zones' R_local)
```

| term | value | charged to | when |
|---|---|---|---|
| served | **+1.0** | dispatching zone | on ride completion |
| relocator credit | **+0.65** | zone that relocated the car | on **dispatch** (D48) |
| empty km | **−0.06/km** | zone that commanded the move | per turn as flown |
| waiting | **−0.2** × Σwait/timeout | zone containing the waiting rider | every turn |
| cancellation | **−1.0** | origin zone + idle-holding neighbours | on timeout |

`NYCConfig`: 16 zones, 26 drivers, `max_pending` 16, action_dim 33.

**Two invariants that must be asserted, not assumed:**

```
# scripts/diag_reloc_invariant.py — CONDITIONAL, the one that matters
share * gamma * P(dispatch | relocated) > beta * km_total
# 0.65 * 1.0 * 0.568 = 0.3692 > 0.06 * 4.36 = 0.2616   -> HOLDS at 1.41x
```

`AGENTS.md`'s literal form (`share > beta * km`) **passed while every relocation
was net-negative** — it ignores that the credit only fires ~57% of the time.
D64 raised `share` 0.45 → 0.65 to restore it.

---

## 6. Feasibility — the scenario is solvable

MEASURED inputs (`scripts/diag_feasibility.py`), INFERRED verdict. The 0.83×
figure assumes a 10.0 min job (ride 6.55 min + deadhead 1.46 km at an implied
0.42 km/min ≈ 3.5 min). The same inputs give 0.55× to 1.09× depending on how
much deadhead is charged, so the verdict is a range, not a single measured
number. Treat "feasible with headroom" as a hypothesis until deadhead is
measured directly.

| quantity | value |
|---|---|
| fleet / demand | 26 cars, 130 orders/hour, 16 zones |
| mean trip distance | **2.77 km** (p90 5.63) |
| mean ride duration | 6.55 min |
| deadhead per completed ride | 1.46 km (53% of ride time) |
| implied speed | 0.42 km/min (25.3 km/h) |
| **feasibility** | **0.83× fleet — feasible with headroom** |

Trip distance is logged per episode (D57 follow-up). Order `origin`/
`destination` are in **degrees**, so distance needs haversine — dividing by 1000
as if metres silently reports `0.000 km`.

Per-episode outcome: 59 completed, 55 cancelled, 16 still open at cutoff.

---

## 7. Map of the repo

| path | what it is |
|---|---|
| `docs/decisions.md` | **the record** — D35–D64, every change with its measurement |
| `src/.../config.py` | all experiment parameters; reward weights + rationale |
| `src/.../env/pettingzoo_env.py` | reward computation, obs, action mask |
| `src/.../networks/encoder.py` | D54 destination-conditioned relocation head |
| `tests/` | 41 tests, incl. D54 head and D59 cost attribution |
| `scripts/eval_tradeoff.py` | **the three-metric check** (SR / empty km / wait) |
| `scripts/diag_targeting_at_decision.py` | **precision at argmax — primary metric** |
| `scripts/diag_reward_rank_resolution.py` | proves whether reward can rank destinations |
| `scripts/diag_reloc_invariant.py` | conditional expected-value invariant |
| `scripts/diag_feasibility.py` | Little's Law feasibility |
| `scripts/watch_plateau.py` | plateau detection |
| `scripts/diag_demand_sensitivity.py` | causal demand intervention |
| `scripts/export_shift_videos.py` | 2-zone video, `--hubs 10 12` |
| `results/videos/` | `shift_{greedy,ippo,mappo}.mp4`, zones 10→12 |
| `results/tradeoff/*.json` | recorded SR/empty/wait per arm |

**Videos** (zones 10→12, 3 hops, ~6.9 km, shift verified 1.00→1.00):
`fleet@B` is the share of fleet in zone B per phase — a policy that follows the
shift shows it **rising** after minute 25. Greedy's falls (15.4% → 4.8%).

---

## 8. Suggested order of work

1. **Implement deficit-scaled relocator credit** (§2). This is the binding
   defect. Test against 49% precision and the 87% ceiling.
2. **Empty km** is the separate, unmoved wall. Everything so far improves *where*
   cars go; nothing reduces *how many* move 2.9 km. `beta` is capped at ~0.156
   by the conditional invariant — raising it alone breaks profitability. Likely
   needs a fleet-size change or an explicit invariant break. **User's call.**
3. **≥5 seeds** on whatever config wins, then ≥2 more for the IPPO/MAPPO
   divergence.
4. **Only then** the demand-shift study.

---

## 9. Open questions I could not settle

- **Why did IPPO regress under D64?** Arm-specific critic quality is a
  hypothesis, not measured.
- **Is 49% precision enough?** Unknown. No reference point for "good" —
  LP's precision was never measured under the same protocol.
- **Should the reward rank by deficit, or should the observation change?** Both
  are plausible; I only tested the reward side.
- **Logit scale.** The relocation block sits at **93–94% of maximum entropy**
  (uniform over ~10.9 legal targets) while dispatch is near-deterministic, and
  `entropy_coef` has no leverage on it (MEASURED,
  `scripts/diag_entropy_by_block.py`). The destination logits span only ~0.4
  nats. A temperature on `dest_logits` would sharpen them **without changing the
  argmax**, so it is free. **Untested.**