# INTENT — why this project exists

(Read this before writing or changing any experiment code.)

## The questions (refined with advisor, 2026-09 — see D12)

- **RQ1**: How much does fleet-rebalancing performance degrade as passenger
  demand moves away from the training distribution?
- **RQ2**: Do IPPO and MAPPO degrade *differently* under the same demand shifts?

Central question: *How robust are independent and centralized-training MARL
approaches for dynamic fleet rebalancing under passenger-demand distribution
shift?*

Framing (advisor): the research problem is demand distribution shift;
IPPO-vs-MAPPO is one investigation method, not the problem itself; the tuned
baseline exists for credibility, not as a research direction.

## Why it matters

- Ride-hailing demand shifts constantly (seasonality, events, new hotspots);
  retraining daily is expensive, so frozen-policy robustness is the
  deployment-relevant property.
- The MARL community benchmarks in-distribution; the rebalancing literature
  handles uncertainty at training time. The evaluation gap between them is
  unmeasured. (Full gap statement: `docs/literature.md`.)

## Hypotheses (all four outcomes are publishable — do not steer)

- H1: MAPPO more robust (centralized critic learns demand-invariant structure)
- H2: IPPO more robust (centralized critic overfits training demand)
- H3: No meaningful difference
- H4: Difference depends on shift type (see taxonomy in `docs/demand-shift.md`)

## Non-goals (if you're doing these, you're off course)

- No new algorithm. No "RobustMAPPO-X." This is a measurement study.
- No headline claims from in-distribution performance.
- No learning the dispatch policy — dispatch is a fixed rule; agents learn
  only `relocate` (pending decision D5 in `docs/decisions.md`).
- No sim-to-real or deployment claims. One city, simulated demand.
- No beating an untuned baseline. Baseline tuning is part of the work.

## Definition of done (what "answered" looks like)

1. Degradation curves (KPI vs shift severity) for IPPO, MAPPO, tuned baseline,
   and random floor, across all scenarios in `docs/demand-shift.md`.
2. ≥ 5 seeds, mean ± std, identical scenario instances across methods.
3. KPIs tracked over episode time, not just finals.
4. At least one non-synthetic scenario (real held-out demand window).
5. A written answer to H1–H4 with the curves as evidence.

## When this file changes

Any change here is a scope change: log it in `docs/decisions.md` and tell the
advisor. Agents must refuse work that contradicts this file and ask instead.