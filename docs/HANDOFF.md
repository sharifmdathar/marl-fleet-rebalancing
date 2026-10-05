# HANDOFF — MARL fleet rebalancing

**Read `docs/decisions.md` for the full record (D35–D82). This file is the map:
where things stand, what was tried, what broke, what to do next.**

Everything is labelled `MEASURED` / `INFERRED` / `TODO`. A claim here with no
measurement attached is a hypothesis.

**Rewrite note (D83).** The previous version of this file was measured on a
single validation window and on an LP that had never been tuned. Both defects
were found later (D70) and invalidated every table in it, so the stale tables
were **deleted rather than annotated** — per `AGENTS.md`, a stale number is
worse than a missing one because it gets cited. The withdrawn figures live in
`docs/decisions.md`, which keeps corrections in place.

---

## 1. Bottom line

**Do not start the demand-shift study yet.** Two things must be true first, and
neither is yet.

MEASURED, 12 val windows (one episode per window), paired, argmax actions,
service rate = completed / total (`scripts/diag_objective_gap.py`, seeds
12000–12011):

| arm | SR % | empty km | wait min | objective R | SR/100km |
|---|---|---|---|---|---|
| greedy, relocation masked off | 41.6 ± 7.9 | 64.9 | 2.47 | −93.0 | 64.1 |
| LP, old defaults | 53.9 ± 7.6 | 141.2 | 2.58 | −48.5 | 38.2 |
| **LP, re-tuned** | **70.1 ± 3.3** | 339.9 | 2.68 | **+4.4** | 20.6 |
| RL untrained (1 PPO update) | 66.9 ± 3.7 | 352.5 | 2.78 | −8.7 | 19.0 |
| RL flat bonus (pre-D71) | 67.2 ± 4.5 | 472.3 | 2.67 | −13.0 | 14.2 |
| RL split credit (D71) | 69.4 ± 3.3 | 468.1 | 2.71 | −3.5 | 14.8 |

Two readings that replace everything previously believed:

1. **RL with relocation masked off is identical to greedy** (41.6%, 64.9 km).
   Everything RL adds comes from relocation, and *almost any* relocation helps:
   the **untrained** policy gains 25 points of service over greedy. So "can RL
   learn to relocate" was never the open question — it already answers yes.
2. **40 iterations of training bought nothing.** Paired against untrained:
   +0.3 ± 1.0 points of service for +120 ± 13 km.

### What must be true before the study

| precondition | status |
|---|---|
| A valid **IPPO** arm on current code | **MISSING.** Every IPPO checkpoint predates the split-credit, chain-share, real-rows and window-coverage fixes. There is no valid IPPO result. |
| **≥5 seeds** on the reported config | **MISSING.** All of D65–D82 is 1 seed, labelled debugging output in the log itself. |
| Frozen-policy definition settled (argmax vs sampled) | **OPEN.** D72 found the two behave very differently at high beta. |
| RL competitive with LP on the frontier | **PARTIAL.** Ties LP on service (69.4 vs 70.1), 128 km worse on distance. |

---

## 2. The two open defects

### 2a. The frontier gap is not closed by beta

MEASURED, seed 1, split credit + targeting bonus 0.1 + distance term +
real-rows update, 12 val windows, argmax (D79 → D80 → D81):

| iteration | SR % | empty km | wait | effective distance weight |
|---|---|---|---|---|
| 40 | 68.0 | 461 | 2.77 | −0.17 |
| 80 | 67.1 | 416 | 2.84 | |
| 120 | 65.8 | 332 | 2.80 | −0.80 |
| 160 | 61.2 | 218 | 2.90 | |
| 200 | 60.5 | 234 | 3.02 | |
| 240 | 59.9 | 204 | 2.91 | −1.61 |
| 120 (+ inbound, D81) | 64.4 | 281 | 2.84 | |

**This is the central negative result.** With distance priced, the policy *does*
learn to drive less — the learned weight goes −0.006 → −0.80 → −1.61. But it
never settles near LP. It keeps trading service away, and its points sit
**below LP's own trade-off curve** (LP 67.6% at 259 km, 58.9% at 151 km; RL
65.8% at 332 km, 59.9% at 204 km — indicative, different seeds and split).

**Beta moves *where on the curve* RL sits. It does not move the curve.** No value
of it closes the gap: 0.06, 0.12 and 0.2 all measured (D72, D76, D79–D81).

Where the extra empty km comes from, MEASURED (D75):

| | RL | LP |
|---|---|---|
| relocations / episode | 141 | 112 |
| mean km per relocation | 3.17 | 2.13 |
| moves ≥4 km / <2 km | 26% / 22% | 6.7% / 54% |
| idle car with nothing to dispatch: relocate | **100%** | 26% |
| driver-minutes relocating | 722 | 479 |

~70% is longer hops, ~30% is never holding a car.

### 2b. Aim: solvable, and solved, but it does not buy service

The reward could not rank destinations (D63): best-vs-other-nonempty **+2.9 pp**
against nonempty-vs-empty **+20.9 pp** — 7× more sensitive to "did you aim
somewhere empty". A supervised probe on identical features reached 87.5%, so
the features were sufficient.

**Three fixes, all measured:**

- **D66 deficit-scaled credit** (what HANDOFF previously proposed): implemented,
  **did not help**. Kept behind a flag.
- **D67/D73 centred targeting bonus** (`--targeting-bonus`): works as designed.
  Aim 34% → 74% at bonus 0.2; two-zone fleet-at-B 5% → 27% where LP gets
  8% → 30%. On ordinary demand it costs service and wait, and ties LP at 0.1.

  | bonus | SR vs LP | empty km vs LP | wait vs LP | aim | two-zone ph2 | fleet@B |
  |---|---|---|---|---|---|---|
  | 0 | −0.6 ± 1.1 | +128 ± 11 | +0.03 | 34% | 13.7% | 5% → 7% |
  | 0.05 | −1.2 ± 0.6 | +115 ± 12 | +0.22 | 44% | 20.9% | 6% → 22% |
  | **0.1** | **−0.4 ± 1.4** | +118 ± 12 | +0.14 | 52% | 31.3% | 5% → 23% |
  | 0.2 | −3.1 ± 0.9 | +103 ± 12 | +0.37 | 74% | 24.3% | 5% → 27% |

  **No value touches the empty-km gap.**
- **D75 distance term**: the old head's distance was a linear layer on
  (target − own), and own position is constant across targets so it cancelled —
  the policy *could not* prefer a nearer zone. Fixed to a real centroid
  distance, zero-init, old checkpoints load unchanged.

**Caveat that survives all of it (D69):** aim ≠ service. 70% of dispatches of a
relocated car happen outside the zone it was sent to, because mid-flight
dispatch serves nearby demand on the way. Forbidding it collapses service
(SR 32.8% vs 64.6%). Never report aim as a service result.

---

## 3. Measurement discipline

**Six numbers this project reported were wrong**, each caught by testing a
premise rather than accepting a conclusion. They cost days; don't repeat them.

| wrong | correct | why |
|---|---|---|
| relocator margin 3.01× | 2.54× | hardcoded 2.49 km leg; re-measured 2.95 km |
| relocation chance 4.1% | 8.4% | confounded by mid-flight dispatch; D44 predated the D45 mask change |
| "network ignores demand" | demand reaches logits, L1 2.9–15.3 | probe wrote an unreachable observation state |
| targeting 13–20% | 51–55% at argmax | `deterministic=False` — sampling discarded the policy |
| **every validation number D47–D69** | re-measured on 12 windows | seeded reset rewound the held-out cursor; **one** window, not twelve |
| **LP "dominates RL"** | RL ties LP on service | LP was never tuned on this env; untuned it scored 53.9% |

Rules that follow:
- `--deterministic` for any decision-quality measurement; recorded in the JSON.
- **One episode per held-out window, all 12.** Print the window index before
  trusting a held-out number.
- Service rate = **completed / total**.
- Evaluate paired on identical seeds (`docs/eval-noise.md`); per-window demand
  varies 36–69%.
- **Sum the zone rewards and compare with the objective.** D49's on-top relocator
  bonus made the shared policy optimise R + 0.65 per moved car (~+40/episode);
  training raised that and lowered R.
- **Check convergence before diagnosing.** An iteration-30 policy was analysed
  for hours before anyone checked it had converged.
- `scripts/watch_plateau.py` rather than running to the nominal iteration count.
- Order coords are **degrees**; use haversine. Dividing by 1000 as if metres
  reports 0.000 km.

---

## 4. The environment as it stands

`NYCConfig`: 16 zones, 26 drivers, `nyc_window_limit` 130, `max_pending` 16,
action space **Discrete(33)**, observation length **9**, 84 train / 12 val /
12 test windows.

```
R_zone = 0.8 * R_local + 0.2 * mean(all zones' R_local)
```

| term | value | charged to | when |
|---|---|---|---|
| served | +1.0 | dispatching zone | on ride completion |
| relocator credit | **split out of gamma**, not on top (D71) | zones that moved the car, shared along the chain (D82) | on **dispatch** (D48) |
| empty km | −`beta`/km | zone that commanded the move | per turn as flown |
| waiting | −`alpha` × Σwait/timeout | zone containing the waiting rider | every turn |
| cancellation | −1.0 | origin zone + idle-holding neighbours | on timeout |
| targeting bonus | 0 by default (D67) | zone that commanded the move | at execution |

Weights: `alpha` 0.2, `beta` 0.06, `gamma` 1.0, `cancel` 1.0, `share` 0.65,
`team_share` 0.2, `order_timeout` 5.0.

**The reward must sum to the objective.** `tests/test_reward_sums_to_objective.py`
asserts it, covering split + chain sharing. D71 exists because it didn't.

**D78 audited the plumbing and it is sound.** Every observation field matched
simulator truth to float precision over 1920 decisions; 278/278 relocations and
196/199 dispatches executed as chosen; buffer contents matched what was acted
on. An imitation-trained copy of the net reaches 100% on stay-vs-move and 92%
on target. **The failure is the learning signal, not the implementation** —
per-decision advantage sd ~1.15 against a hold-vs-move difference of ~0.1,
split-half gradient cosine 0.16, and 85% of PPO rows were forced (zero gradient
but still in the batch mean and advantage normalisation — fixed by D78's
real-rows update, which also un-scaled the distance input ×10).

---

## 5. Repo map

| path | what it is |
|---|---|
| `docs/decisions.md` | **the record** — D35–D82, every change with its measurement |
| `docs/demand-shift.md` | shift taxonomy S0–S7, protocol, preconditions |
| `docs/eval-noise.md` | why paired evaluation is mandatory |
| `docs/spaces-freeze.md` | current 33/9 spaces + the superseded 62-zone freeze |
| `docs/nyc-pipeline.md` | data stages; `data/nyc16` is the study build |
| `src/.../config.py` | every parameter, with rationale |
| `src/.../env/pettingzoo_env.py` | reward, observation, action mask |
| `src/.../networks/encoder.py` | `ZoneAttentionEncoder`, D54 head, D75 distance |
| `src/.../baselines/lp_baseline.py` | LP rebalancer (tuned: lookahead 3.0, penalty 4.0) |
| `tests/` | **65 tests**, incl. reward-sums-to-objective, chain credit, real rows |
| `scripts/eval_tradeoff.py` | three-metric check (SR / empty km / wait) |
| `scripts/diag_objective_gap.py` | **all 12 windows, paired, vs the objective** |
| `scripts/diag_targeting_at_decision.py` | decision-time aim precision |
| `scripts/diag_reloc_outcomes.py` | do relocated cars serve their target zone? |
| `scripts/diag_reloc_invariant.py` | conditional expected-value invariant |
| `scripts/diag_feasibility.py` | Little's Law feasibility |
| `scripts/tune_lp_objective.py` | LP sweep on **train** windows by R |
| `scripts/watch_plateau.py` | plateau detection |
| `scripts/export_shift_videos.py` | videos, `--hubs 10 12` |
| `scripts/competence_two_zone.py` | two-zone diagnostic (**not** a benchmark) |

**The two-zone test is near-degenerate and caps phase-2 service near 53%** for
every policy: ~47% of zone B's road nodes lie beyond the 2 km dispatch radius.
Use it as a targeting probe, never as a performance number.

---

## 6. Order of work

1. **Train IPPO on current code.** Biggest single gap — there is no valid IPPO
   arm, and IPPO-vs-MAPPO is the actual research question. Everything so far is
   MAPPO only.
2. **D82 (chain sharing) has never been trained on.** Implemented and tested,
   zero training runs. Cheap, and 36% of relocations are overwritten before
   dispatch (D78), so it should matter.
3. **Settle argmax vs sampled as the frozen policy** (D72). Everything inherits
   it.
4. **≥5 seeds** on whatever config wins. Nothing before this is reportable.
5. **Then the shift study**, against the preconditions in
   `docs/demand-shift.md`.

**Open, user's call:**
- **The value of `beta`.** It sets where on the frontier RL sits, and the
  study's degradation conclusion depends on that choice. 0.06 / 0.12 / 0.2 all
  measured; none closes the frontier gap (§2a).
- **Empty km needs a bigger fleet or a deliberate break of the relocation
  invariant.** Nothing in the reward has moved it in 15 decisions.
- **Whether to return to 62 zones.** The only reason to leave was compute
  (D65). It is also the only build where S4 (cold-zone concentration) is
  meaningful — see `docs/demand-shift.md`.

**Open questions I could not settle:**
- Why the two arms diverge under identical reward changes (D64: MAPPO 49% aim,
  IPPO regressed 37% → 13%). Arm-specific critic quality is a hypothesis.
- Is 49% aim precision *enough*? LP's precision was never measured under the
  same protocol, so there is no reference point.
- `entropy_coef` is a dead lever: the relocate block sits at 93–94% of maximum
  entropy (D-diagnostic), so the term already dominates the policy loss.
- Destination logits span only ~0.4 nats. A temperature would sharpen them
  without changing the argmax — free, and untested.
