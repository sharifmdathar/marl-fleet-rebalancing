# INTENT — why this project exists

(Read this before writing or changing any experiment code. Restored 2026-10-04
from the pre-purge version in `bc53c9c^`, with the KPI requirement made explicit.)

## The questions

- **RQ1**: How much does fleet-rebalancing performance degrade as passenger
  demand moves away from the training distribution?
- **RQ2**: Do IPPO and MAPPO degrade *differently* under the same demand shifts?

Central question: *How robust are independent and centralized-training MARL
approaches for dynamic fleet rebalancing under passenger-demand distribution
shift?*

Framing (advisor): the research problem is demand distribution shift;
IPPO-vs-MAPPO is one investigation method, not the problem itself; the tuned
baseline exists for credibility, not as a research direction.

## Scope (confirmed 2026-10-04, see D65)

NYC, 16 busiest taxi zones (train-window selection), 26 drivers, real FHVHV
demand. The full 62-zone build is kept as reference only.

## Why it matters

- Ride-hailing demand shifts constantly (seasonality, events, new hotspots);
  retraining daily is expensive, so frozen-policy robustness is the
  deployment-relevant property.
- The MARL community benchmarks in-distribution; the rebalancing literature
  handles uncertainty at training time. The evaluation gap between them is
  unmeasured. (Full gap statement: `docs/literature.md`.)

## Target metrics (all three, always together)

Every result reports **service rate, empty distance, and average waiting time**
together, mean ± std over ≥5 seeds:

- **Service rate** = completed / total orders (higher is better).
- **Empty distance** = km driven without passengers, including relocation legs
  (lower is better). Required alongside service rate because service rate can
  be bought with deadhead.
- **Average waiting time** = pickup time − request time over completed orders
  (lower is better). Required because it is the rider-side cost that service
  rate hides.

A gain on one metric is not a result unless the cost on the other two is shown
next to it. Efficiency is reported as service rate per 100 km of empty distance.

## Hypotheses (all four outcomes are publishable — do not steer)

- H1: MAPPO more robust (centralized critic learns demand-invariant structure)
- H2: IPPO more robust (centralized critic overfits training demand)
- H3: No meaningful difference
- H4: Difference depends on shift type (see taxonomy in `docs/demand-shift.md`)

## Non-goals (if you're doing these, you're off course)

- No new algorithm. No "RobustMAPPO-X." This is a measurement study.
- No headline claims from in-distribution performance.
- No separate dispatch algorithm. Zone agents choose both dispatch and
  relocation (user decision 2026-10-05; the earlier "dispatch is a fixed rule"
  line did not match the code).
- No sim-to-real or deployment claims. One city, simulated demand.
- No beating an untuned baseline. Baseline tuning is part of the work.

## Definition of done (what "answered" looks like)

1. Degradation curves (all three target metrics vs shift severity) for IPPO,
   MAPPO, the tuned LP baseline, and the greedy floor, across all scenarios in
   `docs/demand-shift.md`.
2. ≥ 5 seeds, mean ± std, identical scenario instances across methods.
3. KPIs tracked over episode time, not just finals.
4. At least one non-synthetic scenario (real held-out demand window).
5. A written answer to H1–H4 with the curves as evidence.

## When this file changes

Any change here is a scope change: log it in `docs/decisions.md` and tell the
advisor. Agents must refuse work that contradicts this file and ask instead.
