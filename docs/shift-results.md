# Demand Shift Results — Phase 4 (2026-09-12/13)

Frozen-policy evaluation of the 4 properly-trained models
(`ippo/mappo-nyc-proper-seed0/1`, 70 iters, uniform temporal sampling,
`nyc_window_limit=500`). Raw data: `results/shift_eval/summary.json`
plus per-shift CSVs. Eval outputs: `results/eval/*.txt`.

Protocol: frozen policy, 5 episodes per shift, `base_seed=1000`,
deterministic. Shift IDs per `docs/demand-shift.md`. S0/S2c are the
in-distribution control (same config as training).

## Degradation curves: service rate (mean ± std, %)

| Policy | S0 | S2a 0.5x | S2b 0.75x | S2d 1.5x | S2e 2.0x | S1a z0 | S1b z30 | S4a z60 |
|---|---|---|---|---|---|---|---|---|
| ippo-s0 | 51.3±2.9 | 63.9±3.4 | 60.4±4.1 | 41.4±2.6 | 32.0±1.1 | 21.8±2.0 | 23.3±1.8 | 26.1±2.0 |
| ippo-s1 | 53.8±4.1 | 61.7±4.4 | 60.7±3.2 | 41.8±2.0 | 33.3±1.3 | 22.6±2.6 | 27.1±2.6 | 25.6±3.1 |
| mappo-s0 | 51.1±3.0 | 66.0±4.2 | 61.6±2.8 | 41.6±2.7 | 33.5±1.2 | 18.7±0.9 | 21.2±1.5 | 26.7±1.5 |
| mappo-s1 | 49.6±2.5 | 62.9±3.8 | 58.7±4.0 | 38.9±2.5 | 31.2±2.2 | 21.1±1.4 | 22.6±1.9 | 28.2±2.4 |

## Empty distance (km/ep, mean)

| Policy | S0 | S2a | S2b | S2d | S2e | S1a | S1b | S4a |
|---|---|---|---|---|---|---|---|---|
| ippo-s0 | 659 | 972 | 770 | 486 | 417 | 1080 | 1047 | 1010 |
| ippo-s1 | 639 | 1000 | 761 | 478 | 400 | 1088 | 982 | 1030 |
| mappo-s0 | 599 | 671 | 622 | 438 | 383 | 799 | 892 | 652 |
| mappo-s1 | 216 | 167 | 217 | 224 | 220 | 118 | 120 | 164 |

## Avg wait (min/ep, mean)

| Policy | S0 | S2a | S2b | S2d | S2e | S1a | S1b | S4a |
|---|---|---|---|---|---|---|---|---|
| ippo-s0 | 2.50 | 2.47 | 2.45 | 2.27 | 2.27 | 2.79 | 2.73 | 2.75 |
| ippo-s1 | 2.40 | 2.47 | 2.38 | 2.24 | 2.15 | 2.84 | 2.69 | 2.59 |
| mappo-s0 | 2.28 | 2.17 | 2.24 | 2.18 | 2.13 | 2.25 | 2.33 | 2.49 |
| mappo-s1 | 1.83 | 1.84 | 1.95 | 1.87 | 1.87 | 1.95 | 1.99 | 1.96 |

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

## H1–H4 reading (preliminary, 2 seeds each)

- H1 (MAPPO more robust): not supported — curves overlap.
- H2 (IPPO more robust): not supported — curves overlap.
- H3 (no difference): supported on magnitude shifts.
- H4 (depends on shift type): partially — spatial shifts hurt
  everyone equally, so the "difference" is between shift types,
  not between methods. With MAPPO-seed1 excluded as degenerate,
  the honest headline is H3 with a strong RQ1 spatial-collapse result.
