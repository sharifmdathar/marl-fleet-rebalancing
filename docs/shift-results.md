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
