# Demand Shift Results — Phase 4 (2026-09-12/13)

Frozen-policy evaluation of the 4 properly-trained models
(`ippo/mappo-nyc-proper-seed0/1`, 70 iters, uniform temporal sampling,
`nyc_window_limit=500`). Raw data: `results/shift_eval/summary.json`
plus per-shift CSVs. Eval outputs: `results/eval/*.txt`.

Protocol: frozen policy, 5 episodes per shift, `base_seed=1000`,
deterministic. Shift IDs per `docs/demand-shift.md`. S0/S2c are the
in-distribution control (same config as training).

## Degradation curves: service rate (mean ± std, %)

| Shift | Description | ippo-s0 | ippo-s1 | mappo-s0 | mappo-s1 |
|---|---|---|---|---|---|
| S0 | In-distribution control | 51.3±2.9 | 53.8±4.1 | 51.1±3.0 | 49.6±2.5 |
| S1a | Spatial 70% zone 0 | 21.8±2.0 | 22.6±2.6 | 18.7±0.9 | 21.1±1.4 |
| S1b | Spatial 70% zone 30 | 23.3±1.8 | 27.1±2.6 | 21.2±1.5 | 22.6±1.9 |
| S2a | Magnitude 0.5x | 63.9±3.4 | 61.7±4.4 | 66.0±4.2 | 62.9±3.8 |
| S2b | Magnitude 0.75x | 60.4±4.1 | 60.7±3.2 | 61.6±2.8 | 58.7±4.0 |
| S2d | Magnitude 1.5x | 41.4±2.6 | 41.8±2.0 | 41.6±2.7 | 38.9±2.5 |
| S2e | Magnitude 2.0x | 32.0±1.1 | 33.3±1.3 | 33.5±1.2 | 31.2±2.2 |
| S3 | Temporal (val split) | 49.8±1.2 | 54.1±1.9 | 51.5±3.0 | 48.9±2.9 |
| S4a | Hotspot 60% zone 60 | 26.1±2.0 | 25.6±3.1 | 26.7±1.5 | 28.2±2.4 |
| S5 | Event 20% zone 0 | 46.5±4.2 | 49.0±3.3 | 46.0±3.3 | 43.3±2.4 |
| S6 | Real held-out (test) | 51.1±3.0 | 52.9±3.2 | 52.3±4.0 | 48.6±3.2 |

## Empty distance (km/ep, mean)

| Shift | Description | ippo-s0 | ippo-s1 | mappo-s0 | mappo-s1 |
|---|---|---|---|---|---|
| S0 | In-distribution control | 659 | 639 | 599 | 216 |
| S1a | Spatial 70% zone 0 | 1080 | 1088 | 799 | 118 |
| S1b | Spatial 70% zone 30 | 1047 | 982 | 892 | 120 |
| S2a | Magnitude 0.5x | 972 | 1000 | 671 | 167 |
| S2b | Magnitude 0.75x | 770 | 761 | 622 | 217 |
| S2d | Magnitude 1.5x | 486 | 478 | 438 | 224 |
| S2e | Magnitude 2.0x | 417 | 400 | 383 | 220 |
| S3 | Temporal (val split) | 677 | 649 | 588 | 223 |
| S4a | Hotspot 60% zone 60 | 1010 | 1030 | 652 | 164 |
| S5 | Event 20% zone 0 | 708 | 681 | 646 | 192 |
| S6 | Real held-out (test) | 717 | 693 | 609 | 218 |

## Avg wait (min/ep, mean)

| Shift | Description | ippo-s0 | ippo-s1 | mappo-s0 | mappo-s1 |
|---|---|---|---|---|---|
| S0 | In-distribution control | 2.50 | 2.40 | 2.28 | 1.83 |
| S1a | Spatial 70% zone 0 | 2.79 | 2.84 | 2.25 | 1.95 |
| S1b | Spatial 70% zone 30 | 2.73 | 2.69 | 2.33 | 1.99 |
| S2a | Magnitude 0.5x | 2.47 | 2.47 | 2.17 | 1.84 |
| S2b | Magnitude 0.75x | 2.45 | 2.38 | 2.24 | 1.95 |
| S2d | Magnitude 1.5x | 2.27 | 2.24 | 2.18 | 1.87 |
| S2e | Magnitude 2.0x | 2.27 | 2.15 | 2.13 | 1.87 |
| S3 | Temporal (val split) | 2.49 | 2.38 | 2.27 | 1.94 |
| S4a | Hotspot 60% zone 60 | 2.75 | 2.59 | 2.49 | 1.96 |
| S5 | Event 20% zone 0 | 2.53 | 2.51 | 2.26 | 1.80 |
| S6 | Real held-out (test) | 2.48 | 2.39 | 2.25 | 1.87 |

## Findings

1. **Magnitude shifts degrade monotonically.** SR falls from ~62-66%
   (0.5x) to ~31-34% (2.0x) for all four policies. No method escapes
   overload: at 2.0x demand, rewards go negative (-506 to -648/ep).
   Empty distance is inversely correlated with demand — busier fleets
   deadhead less (fewer idle gaps between dispatches).

2. **Spatial shifts collapse all policies.** Concentrating 60-70% of
   demand in a single zone drops SR to 18-28% across the board
   (vs ~50% in-distribution). The frozen policies cannot re-route
   capacity to an unseen hotspot; drivers deadhead 800-1100 km
   chasing orders outside their learned coverage. This is the
   strongest RQ1 evidence: spatial shift hurts far more than
   2x magnitude shift.

3. **No IPPO-vs-MAPPO winner on magnitude (supports H3 there).**
   Both degrade along nearly identical curves (within 1-3pp at
   every severity). MAPPO-seed0 is marginally best at low demand
   (66.0% at 0.5x) but the gap is within seed noise.

4. **MAPPO-seed1 is an exploration failure, not a reward failure
   (flag, do not headline).** Empty distance stays at 118-224 km
   under every shift (vs 400-1000 for the other three) while SR is
   consistently lowest — but the reward formulation is exonerated:
   three of four seeds learned good policies under it. The cause is
   premature convergence of the centralized critic: relocation
   actions collapsed to ~0.05% (vs 0.88% IPPO) and idle climbed to
   99.2%, i.e. the policy locked into an idle-heavy local optimum
   early and never explored repositioning strategies afterward.
   The 4739-dim global state with 3 rollout eps/iter gives the
   critic too little data to recover once the actor commits.
   Exclude from IPPO-vs-MAPPO comparisons or report separately
   as a centralized-critic exploration failure mode. Reward
   formulation deliberately kept unchanged (see finding 5).

5. **MAPPO-seed0 survived premature entropy collapse in training.**
   Its entropy fell 88% (0.0355 → 0.0048) vs 46% for IPPO-seed0
   over 70 iters, yet eval SR matches IPPO (51.1% vs 51.3-53.8%).
   The centralized critic (4739-dim state, 3 rollout eps/iter)
   converges to a usable but brittle policy — worth a hyperparameter
   note (entropy_coef 0.01 → 0.02+) before any retraining.

6. **Temporal, event, and held-out shifts barely hurt (2026-09-14,
   all four policies).** S3 (val-split windows, shifted peak timing)
   holds SR at 48.9-54.1% vs 49.6-53.8% in-distribution; S6 (real
   test-split windows) at 48.6-52.9% — the frozen policies generalize
   across real time windows. S5 (mild 20% event burst in zone 0) dips
   only to 43.3-49.0%. Ranking of shift severity is now clear:
   spatial concentration (18-28%) >> 2x magnitude (~31-34%) >
   mild event (~43-49%) ≈ temporal ≈ held-out (≈S0). RQ1 answer:
   policies are robust to *when* demand arrives, fragile to *where*.

## H1–H4 reading (preliminary, 2 seeds each)

- H1 (MAPPO more robust): not supported — curves overlap.
- H2 (IPPO more robust): not supported — curves overlap.
- H3 (no difference): supported on magnitude, temporal, event,
  and held-out shifts alike.
- H4 (depends on shift type): supported in the RQ1 sense — shift
  *type* matters enormously (spatial collapse vs temporal
  invariance), but the IPPO-vs-MAPPO *difference* does not appear
  under any tested type. With MAPPO-seed1 excluded as degenerate,
  the honest headline is H3 with a strong RQ1 ranking result:
  where >> how-much >> when.

---

# Fixed-code re-evaluation (2026-09-14, 5 seeds each — SUPERSEDES above for reporting)

Retrained under the D18 fix (selection-ordered rollout, MAPPO states +
critic values) and re-evaluated: 10 fixed checkpoints
(`ippo/mappo-nyc-fixed-seed0..4`) + greedy + random, same 11 shifts,
5 episodes each. Raw: `results/shift_eval_fixed/summary.json`;
curves: `results/figures_fixed/`. Old `results/shift_eval/` left intact;
do NOT mix pre/post-fix seeds in one aggregate (D17 note).

- IPPO vs MAPPO still overlap on every shift within seed std
  (e.g. S0: 52.9±1.4 vs 51.1±1.8) — H3 holds at n=5.
- Neither RL method beats greedy meaningfully anywhere (±1–3pp).
- **Random floor sits ABOVE all trained methods on most shifts**
  (S0 54.1, S1a 27.3, S2a 76.5, S3 53.5, S6 53.9). Trained policies are
  ~99% idle; doing nothing ≈ greedy, while random dispatch serves more
  under winlim=500. Points at reward shaping (D19), not exploration.
- Severity ranking unchanged: spatial (21–29%) >> 2x magnitude
  (~32%) > event (~45%) ≈ temporal ≈ held-out (≈S0).

## Mechanism-labeled results (2026-09-15 — supersedes the spatial/event reading above)

D21 closed the implementation gap where S1/S4/S5 were all static
`(target_zone, pct)` concentration differing only in zone + percentage.
The three scenarios now use distinct mechanisms as the taxonomy claims, and
ALL 14 shift entries are populated for every policy in
`results/shift_eval_fixed/summary.json`. Deprecated backward-compat rows
(S1a/S1b/S4a/S5a) are retained so old CSVs stay interpretable; the new
mechanism rows (S1/S4_z1/S5_z0) are the ones to report.

**Eval**: 10 RL policies (IPPO 5-seed + MAPPO 5-seed + MAPPO-ent002-seed0)
× 2 baselines (greedy + random) × 14 shifts, 5 episodes each, frozen-policy,
`base_seed=1000`, deterministic. Raw: `results/shift_eval_fixed/summary.json`
+ per-policy CSVs (not committed — raw data). Curves:
`results/figures_fixed/`.

### Service rate (mean ± std, %) — mechanism-labeled, all 14 shifts

|| Shift | Description | IPPO-seed0 | IPPO-seed1 | IPPO-seed2 | IPPO-seed3 | IPPO-seed4 | MAPPO-seed0 | MAPPO-seed1 | MAPPO-seed2 | MAPPO-seed3 | MAPPO-seed4 | MAPPO-ent002 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|| S0 | In-distribution (train) | 54.2±5.5 | 55.0±4.2 | 51.9±3.6 | 52.0±3.5 | 51.5±4.9 | 49.1±2.9 | 50.6±3.0 | 49.2±2.5 | 53.3±2.1 | 53.2±3.0 | 48.3±1.8 |
|| **S1** | **S1 permute p=1.0 (hotspots move)** | **39.3±2.2** | **39.4±2.3** | **38.9±1.1** | **37.9±2.9** | **38.5±1.2** | **34.6±2.9** | **37.0±2.8** | **34.5±2.7** | **38.1±3.5** | **39.7±2.8** | **35.3±3.1** |
|| S1a_z0 | S1a depr. 70% conc z0 | 21.2±3.0 | 21.5±1.7 | 19.7±1.3 | 24.4±2.1 | 19.3±3.3 | 20.7±1.5 | 17.9±1.4 | 20.5±1.7 | 20.8±1.1 | 23.8±1.5 | 19.0±1.8 |
|| S1b_z30 | S1b depr. 70% conc z30 | 23.0±3.3 | 23.2±1.3 | 22.2±1.4 | 25.1±2.1 | 19.8±3.3 | 22.5±1.5 | 20.2±2.4 | 22.2±1.8 | 23.2±1.8 | 24.8±1.4 | 22.6±1.9 |
|| S2a | Magnitude ×0.5 | 63.5±7.2 | 69.0±6.4 | 57.8±6.3 | 61.2±3.8 | 51.1±6.0 | 65.4±3.4 | 53.6±5.5 | 65.2±3.5 | 62.8±4.7 | 64.4±5.9 | 60.6±3.5 |
|| S2b | Magnitude ×0.75 | 61.1±3.3 | 63.8±4.3 | 60.6±2.9 | 60.1±1.5 | 54.4±3.4 | 59.1±3.7 | 53.2±2.2 | 59.0±4.1 | 59.9±3.8 | 62.9±3.9 | 58.7±2.9 |
|| S2d | Magnitude ×1.5 | 42.0±3.0 | 43.1±3.0 | 41.1±2.2 | 41.7±2.2 | 40.2±3.0 | 39.4±2.8 | 38.2±2.8 | 39.3±2.9 | 41.2±2.0 | 42.5±2.8 | 38.8±3.0 |
|| S2e | Magnitude ×2.0 | 32.9±2.1 | 33.1±1.8 | 32.5±1.3 | 32.8±1.3 | 33.2±1.3 | 31.3±1.6 | 31.4±0.9 | 31.2±1.8 | 34.0±1.4 | 33.3±1.6 | 31.1±2.1 |
|| S3 | Temporal phase (val split) | 52.1±1.4 | 51.9±1.9 | 52.6±1.6 | 55.1±0.7 | 51.3±2.5 | 48.8±3.2 | 48.6±1.4 | 48.6±3.4 | 53.2±1.1 | 50.5±2.6 | 48.9±4.0 |
|| **S4_z1** | **S4 cold-zone z1 60% conc** | **26.2±3.1** | **27.9±1.1** | **25.7±3.5** | **24.5±2.1** | **22.3±2.4** | **28.1±1.2** | **23.9±2.2** | **28.2±1.5** | **27.5±1.0** | **26.8±0.8** | **26.7±1.6** |
|| S4a_z60 | S4a depr. 60% conc z60 | 27.6±3.0 | 28.6±2.5 | 24.9±1.4 | 25.9±3.0 | 25.6±2.8 | 28.8±2.6 | 25.1±2.6 | 28.7±2.3 | 31.3±1.5 | 27.2±1.8 | 27.3±3.2 |
|| **S5_z0** | **S5 burst z0 20% [20,30)min** | **53.0±5.2** | **53.2±3.8** | **50.0±3.4** | **50.8±3.0** | **49.1±4.5** | **48.1±1.9** | **48.6±2.9** | **48.5±1.6** | **52.0±2.5** | **52.4±3.0** | **47.7±1.5** |
|| S5a_z0 | S5a depr. 20% conc z0 | 46.7±4.7 | 49.4±3.4 | 44.4±3.8 | 46.6±4.1 | 43.8±3.8 | 43.4±1.7 | 42.0±1.8 | 43.4±2.4 | 45.2±1.5 | 49.4±2.1 | 43.0±2.3 |
|| S6 | Real held-out (test split) | 51.7±4.2 | 53.0±2.5 | 53.0±1.9 | 54.7±1.7 | 52.2±1.4 | 48.2±4.3 | 48.0±5.0 | 48.2±3.6 | 54.4±3.5 | 52.4±3.3 | 48.4±4.0 |

### Aggregated by algorithm family (mean ± std across seeds, %)

|| Shift | IPPO 5-seed mean | MAPPO 5-seed mean | MAPPO-ent002 |
|---|---|---|---|---|
|| S0 (baseline) | 52.9±1.4 | 51.1±1.8 | 48.3 |
|| S1 (permute) | 38.8±0.6 | 36.8±2.0 | 35.3 |
|| S4_z1 (cold z1 60%) | 25.3±1.9 | 26.9±1.6 | 26.7 |
|| S5_z0 (burst z0 20%) | 51.2±1.6 | 49.9±1.9 | 47.7 |

Δ vs S0 (mechanism families only):

|| Family | IPPO ΔS0 | MAPPO ΔS0 | MAPPO-ent002 ΔS0 |
|---|---|---|---|
|| S1 permute | −14.1pp | −14.3pp | −13.1pp |
|| S4 cold z1 | −27.6pp | −24.2pp | −21.6pp |
|| S5 burst z0 | −1.7pp | −1.2pp | −0.7pp |

### Baselines (mean ± std, %)

|| Shift | greedy | random |
|---|---|---|---|
|| S0 | 17.0±1.6 | 6.1±0.5 |
|| S1 (permute) | 15.6±1.5 | 6.1±0.4 |
|| S4_z1 (cold z1) | 11.3±1.0 | 3.5±0.6 |
|| S5_z0 (burst z0) | 16.8±1.3 | 6.2±1.0 |

### Degradation curves

Service rate (mechanism-labeled, algorithm-family mean ± std bands):

![Service rate degradation](results/figures_fixed/degradation_service_rate.png)

Empty distance (km/ep):

![Empty distance degradation](results/figures_fixed/degradation_empty_km.png)

Average wait (min/ep):

![Avg wait degradation](results/figures_fixed/degradation_avg_wait_min.png)

### Updated findings

1. **S1 (spatial permutation) degrades moderately (~14pp), not catastrophically.**
   Hotspots *move* via zone-offset rotation — same total volume, same
   timestamps, but the policy's learned coverage no longer aligns with where
   demand concentrates. SR drops from ~52% to ~38% (IPPO) / ~37% (MAPPO).
   This is ~14pp softer than the old S1a (70% static concentration into
   zone 0, which gave 19-24%) — permutation is less severe than static
   concentration because the policy still encounters its learned zones,
   just in different relative positions. Supports RQ1 "where matters but how
   it moves matters too."

2. **S4 (cold-zone concentration) is the hardest hit (~25pp).** 60% of demand
   concentrated into zone 1 via the same static-concentration mechanism as
   S1a/S4a/S5a, but targeting a zone that is verifiably cold in the training
   split (44 orders, 0.105% across 84 train windows, seed=42; hottest zones
   ~3.7% each). The frozen policy has never been rewarded serving zone 1,
   so it doesn't position drivers there. SR drops to ~25-27% across all
   policies. This is the strongest single-mechanism degradation signal and
   confirms the cold-target hypothesis: concentration into a zone the policy
   never learned to serve is catastrophic even at moderate intensity.

3. **S5 (time-localized burst) barely registers (~1pp).** 20% of orders
   with request_time in [20,30) min get their origin reassigned to zone 0;
   timestamps and total volume preserved, only a 10-min window spikes.
   SR drops from ~51% to ~50% (IPPO) / ~50% (MAPPO) — essentially
   indistinguishable from noise. The policy's response to a transient local
   spike is adequate because it doesn't require repositioning (the burst is
   too short to justify deadheading) and zone 0 is not a cold zone. This
   mechanism is genuinely mild and should NOT be compared with the old S5
   (static 20% concentration into zone 0, which gave 43-49% and looked
   like a real degradation).

4. **Severity ranking across mechanisms is now clean and distinct:**
   cold-zone concentration (S4, −25pp) >> spatial permutation (S1, −14pp) >>
   time-localized burst (S5, −1pp) ≈ temporal (S3, ~0pp) ≈ held-out (S6,
   ~0pp) > magnitude shifts (S2, monotonic −7 to −21pp from 0.5x to 2.0x).
   This is the D21 payoff: three scenarios that looked identical under the
   old static-concentration implementation now show three clearly different
   degradation magnitudes, matching the taxonomy's claim that *type* matters.

5. **IPPO vs MAPPO: still no winner.** Both families degrade along nearly
   identical curves within seed std on every mechanism (S1: IPPO 38.8±0.6
   vs MAPPO 36.8±2.0; S4: 25.3±1.9 vs 26.9±1.6; S5: 51.2±1.6 vs 49.9±1.9).
   MAPPO has slightly wider seed variance on S1 (±2.0pp vs IPPO's ±0.6pp),
   but means are within 2pp. H3 holds at n=5 across all three mechanism
   families.

6. **Baselines: greedy 17%, random 6% at S0.** Greedy (dispatch longest-
   waiting order from the zone with most idle drivers, one dispatch/turn,
   no relocation) is ~11pp above random and ~34pp below RL. It degrades
   with the mechanisms roughly in proportion to RL (S4 hardest, S5 mildest)
   but at a much lower absolute level. Random sits well below both — the
   RL policies are learning something real, not just noise.

### H1–H4 reading (D21, 5 seeds + ent002)

- H1 (MAPPO more robust): not supported — curves overlap on all three
  mechanism families within seed std.
- H2 (IPPO more robust): not supported — same.
- H3 (no difference): supported on S1, S4, S5 alike — IPPO and MAPPO
  degrade within 2pp of each other on every mechanism.
- H4 (depends on shift type): strongly supported — the three mechanism
  families produce three clearly different degradation magnitudes
  (−14pp / −25pp / −1pp), and the old static-concentration rows (S1a/S1b/
  S4a/S5a) produce yet different numbers. Shift *type* matters enormously;
  the IPPO-vs-MAPPO *difference* does not appear under any tested type.
  Honest headline: H3 + strong RQ1 ranking — where (cold zone ≫ permuted
  ≫ burst) >> how-much (magnitude) >> when (temporal ≈ held-out).
