# Decision log

Versioned entries for changes to environment dynamics, reward, or the
observation/action space. Each entry records what changed and why, with the
measurement that justified it. Numbers here are from runs logged in this repo;
see `docs/eval-noise.md` for the paired-comparison rule.

Reset when the pre-env-2.3 artifacts were purged (`bc53c9c`). Earlier entries
survive only in git history.

---

## D35 — SUPERSEDED by D36, retained for the record

**Change:** `require_destination_demand = True` — a relocation destination must
have at least one waiting rider (`_legal_relocate_targets`).

**Why it was made:** RL relocated into empty zones. Precision was 13.5% (LP:
38%), and 87% of relocations went to zones with no unmet demand.

**Why it was wrong:** it deleted the decision instead of penalising the bad
outcome. With demand-saturating destinations filtered out before the mask, the
menu collapsed to **2.25 legal targets per turn** and **81.0%** of turns offered
targets with *identical* pending demand — on those turns "pick the best" and
"pick at random" are literally the same action.

Measured consequence on NYC16 base S0, 5 paired seeds, timeout 2.0:

| arm | paired SR vs dispatch-first |
|---|---|
| relocate-first-best | −3.00 pp |
| relocate-first-rand | **−1.83 pp** |
| relocate-first-first | −2.55 pp |

Random beat "best". Targeting had no measurable value because there was no
targeting decision left.

---

## D36 — `require_destination_demand = False` (env 3.0)

**Change:** restore demand-free zones as legal relocation destinations. The
mask no longer pre-filters on destination demand.

**Why:** the relocation *action* was never inert — earlier "0 served" results
came from probes that could not dispatch (they fell back to action 0, which is
noop; the slice bug made `flatnonzero` return a noop index). With corrected
probes, relocating at all is worth **+18.70 pp for +365 empty km**. The part
that had collapsed was *where* to relocate, and D35 had removed it.

Restoring the choice re-opens the decision the study is about:

| arm | SR% | empty km | paired vs dispatch-first |
|---|---|---|---|
| **relocate-first-best** | **80.53** | 467.6 | **+15.74 ± 1.80 (5W/0L)** |
| relocate-first-rand | 76.63 | 480.5 | +11.84 ± 5.04 (5W/0L) |
| relocate-first-first | 60.50 | 528.9 | −4.29 ± 3.23 (0W/4L) |
| random-legal | 71.86 | 479.9 | +7.07 ± 4.75 (4W/1L) |
| dispatch-first | 64.79 | 519.3 | — |
| dispatch-only | 50.96 | 71.7 | −13.83 ± 5.21 (0W/5L) |

NYC16 base S0, 5 paired seeds, `order_timeout=5.0`. `best > rand > first`, best
wins 5/5 with the tightest spread of any arm. Under D35 those three were
indistinguishable. `relocate-first-first` scoring *below* random is the control:
index correlates with distance from demand, so the ordering is not an artifact.

Menu diagnostics (`scripts/diag_menu_choice.py`, 2 episodes):

| config | menu | spread | null turns |
|---|---|---|---|
| D35 (demand required) | 2.25 | 0.21 | 81.0% |
| D36 (demand-free ok) | 7.39 | 1.40 | 15.7% |

**Risk accepted:** D35 existed to stop a weak policy relocating into empty
zones, and that protection is now removed. The difference is that the policy no
longer has to be weak, because the choice exists. `claim_relocation_targets`
(D34) is deliberately left ON so at most `pending` cars arrive per zone per
turn. Do not relax both guards at once without re-measuring precision.

---

## D35a — `order_timeout` 2.0 → 5.0 minutes

**Change:** patience before an unassigned order cancels, all three configs.

**Why:** the 2.0 value was picked arbitrarily, not from any operational source,
and sat below the travel time of a mean relocation leg (2.49 km at 30 km/h ≈
5.0 min). A repositioned car routinely arrived after its rider had cancelled.

Anchors found (all NYC, all ride-hailing):

- platform average **total** wait 5.49 min (HVFHV, 2026)
- TLC WAV benchmark: **<10 min for 80%** of trips, <15 min for 90%
- accept-to-board pickup time, mean **303 s**

**Inferred, not derived:** those sources measure *total* wait (accept + en route
+ boarding); this field is *unassigned-order patience*. They bound the plausible
range, they do not measure this quantity. 5.0 is the primary arm; 2.0 is
retained as a documented sensitivity arm.

Service rate rises with patience under every policy arm tested, e.g. base S0
dispatch-first 62.8% (2.0) → 64.8% (5.0), and 80.0% at 8.0.

---

## D37 — per-channel normalisation of `all_zone_state` and `demand_hist` (env 3.1)

**Change:** each of the four `all_zone_state` columns is divided by its own
running max-abs scale; `demand_hist` likewise. Raw per-zone totals are now kept
(`_zone_total_counts`) so `observe()` can form ratios without de-normalising.

**Why:** the demand signal was an order of magnitude too weak to learn from.
Measured over 400 turns on NYC16, before:

| channel | max | std |
|---|---|---|
| col 0 fleet total / 26 | 0.3846 | 0.1016 |
| col 1 idle / 26 | 0.0385 | 0.0127 |
| **col 2 pending / 26** | **0.1538** | **0.0402** |
| col 3 deficit | 0.1154 | 0.0359 |

One waiting rider in a 26-car fleet reads `1/26 = 0.0385`. So the difference
between "one rider waiting" and "nobody waiting" was 0.04, on a channel whose
neighbouring column swings by 0.38. Spread ratio across columns: **8.0x**.

After: every column occupies the full Box range, demand std **0.0402 → 0.2610
(6.5x stronger)**, spread ratio **8.0x → 1.27x**. `demand_hist` max 0.1538 → 1.0.

**Why this matters:** it explains three measurements that were otherwise
puzzling.

- RL relocated to demand-free zones **81.6%** of the time (LP: 36.2%). A policy
  that cannot resolve demand at the provided resolution will not resolve it.
- Raising `beta` 0.06 → 0.15 changed relocation volume not at all (10.1% → 9.7%).
  A uniform cost cannot fix a missing signal.
- Sampled policies beat argmax by ~11pp. A stochastic draw samples across the
  7.39-option menu and blunders into demand-having zones by chance more often
  than the mode does — noise compensating for an absent feature.

**Bug found while implementing:** de-normalising the scaled matrix to recover raw
counts produced `idle_frac` up to **1.5**, outside the `Box(-1, 1)` bound. Caught
by PettingZoo `api_test`'s out-of-bounds assertion, fixed by storing
`_zone_total_counts` directly. 24/24 tests pass.

**Not yet verified:** whether a retrained policy actually relocates to demanded
zones less often. The diagnosis is measured; the fix's effect is not.

---

## D38 — `state()` covers every observation channel (MAPPO critic was blind)

**Change:** `state()` now returns 257 dims (critic input 259 with zone identity),
up from 177. Added `demand_hist` (64) and zone adjacency (16).

**Why it was a bug:** `state()` was written at D4. `demand_hist` and adjacency
arrived at D30 as observation-only additions, and `state()` was never revisited.
So MAPPO's centralized critic — the entire point of the arm — was missing **80 of
the 279 observation dims**.

IPPO's decentralized critic sees those channels in its own observation. MAPPO's
did not. **MAPPO was starting with less information than IPPO**, which inverts
RQ2: the study asks whether centralized training helps or hurts, and this made
the answer an artifact of missing inputs rather than of the training paradigm.

Nothing failed loudly, because `test_mappo_conditioned_critic` only asserted
`value()` returned a finite scalar of the right shape. It never checked the
critic could *see* the state. Same class as D35 silently deleting the
relocation decision: a component that was correct became wrong, and no test
covered the property that broke.

**Limitation, deliberate:** the actor's `neighbor_mask` is the ACTING zone's
legal-target set, which differs per agent. A shared critic cannot hold one mask
per zone without an O(num_zones^2) tensor, so it receives **static adjacency**
instead — the zone-independent part. The live per-zone mask remains with the
actor, and zone identity is appended to the state.

**Regression-locked:** two tests in `tests/test_fixes_and_demand.py`, both
verified to fail when `state()` is reverted:

- `test_state_covers_every_observation_channel` — every obs key reaches the
  critic, and the critic accepts the width
- `test_state_critic_sees_demand_history` — perturbs `demand_hist` and asserts
  the state actually changes, so allocated-but-zero dims cannot pass

26 tests pass (was 24).

**Checked and found correct, recorded so they are not "fixed" later:**

- the store/critic round trip: `g_state` is captured once per turn
  (`train.py:308`) and the same vector is stored and fed to the critic
- `explained_var` (`mappo.py:211`) reads `buffer.values`, which for MAPPO *is*
  the centralized critic (`train.py:351`) — an earlier claim that this was wrong
  was itself wrong

---

## Consequence for existing artifacts

D35a and D36 both change environment dynamics. **Every result measured before
them is void**, per the `AGENTS.md` rule that a stale number is worse than a
missing one:

- `logs/lp_tuning_nyc16/` (LP hyperparameter sweep)
- `runs/paired_comparison.json` (the RL-vs-LP gate table)
- `runs/env30-*/` and `runs/m29-*/` (trained under env 2.9)
- the LP 59.5% / 179 km reference the gate is measured against

All must be regenerated. LP must be re-tuned on the new env before the gate can
be judged, since a baseline tuned on different dynamics is not a fair
comparison.
## D44 — relocation is at chance, and the destination is mostly unexpressible

Measured, not hypothesised. Full method in the session handoff; the three
load-bearing results:

**Targeting is a coin flip, and a perfect forecast does not fix it.** Injecting
an oracle demand forecast into the trained checkpoints (no retraining, no obs-dim
change) moved nothing. On 4,236 relocation legs:

| arm | precision |
|---|---|
| uniform random (control) | **4.1%** |
| IPPO blind | 3.9% |
| IPPO + oracle forecast | 4.2% |
| MAPPO blind | 2.8% |
| MAPPO + oracle forecast | 3.1% |

And it is not blindness: a per-channel sensitivity test shows `demand_hist` is the
**most influential input to the relocation logits** (16% of the legal-logit spread,
ahead of `all_zone_state` at 9.5%). The actor reads demand and does not act on it.

**CORRECTION to earlier framing.** The "12.5% chance over 8 legal targets" premise
was wrong twice: the legal set averages **1.12** targets per turn, not 8, and the
true chance rate measured by the random arm is **4.1%**. RL is therefore **at**
chance, not below it. Any writeup calling relocation targeting "below chance" or
"anti-correlated with demand" is an artifact of the bad denominator.

**The destination is largely unexpressible before the switch.** Measured directly
with demand 100% at zone A and zone B holding none (localhub, t < 20 min):

    mean legal relocation targets per turn : 0.83
    turns where B is a legal target        : 3 / 640  (0.5%)

So a policy **cannot pre-position toward B even in principle**. A zone with no
*current* demand offers almost no legal inbound targets. This makes the oracle
result coherent: the forecast is not merely ignored, the action space barely
offers the destination to aim at.

**Relocation does pay, but far too late.** Payback ledger, IPPO, 3 episodes:

    legs                         606
    ever paid back               231 (38.1%)
    never paid                   61.9%
    mean turns to payback        39.1  (horizon 121)
    cost/leg 0.1106 vs paid 0.45
    net per leg                  +0.0609

The "relocation is priced out" hypothesis is **dead** for the current constants —
the ledger is net positive. But zero legs pay within 5 turns and the median is
21–40 turns, against a 5-minute order timeout.

**Net:** two mechanisms compound rather than compete. The actor sees the shift
but (a) is not rewarded for acting early, 0% payback inside 5 turns, and (b)
cannot legally express the aim, 0.5% reachability. Fixing either alone should not
be expected to fix targeting.

**Left open:** whether widening the legal-target set, paying relocation credit on
arrival rather than completion, or both are the right move. Not decided here.

---

## D47 — relocation credit is paid to one zone, undivided (no relay chains)

**Change:** `_driver_relocator` goes from `dict[int, list[int]]` to
`dict[int, int]`, and the completion credit becomes
`relocator_share * gamma` paid in full to that single zone. The previous form
was `relocator_share * gamma / len(chain)`.

**Why it was made:** the "relay chain" was not a relay. An entry was appended
every time ANY zone relocated a car that already had one, and the list was
cleared only when that car completed an order. So `len(chain)` counted *how
often a car had been reused since it last earned*, not how many zones staged a
journey.

Measured directly (`scripts/diag_chain_semantics.py`, 504 relocations, 26
cars, 4 episodes, dispatch-first greedy):

| gap between two hops of the same car | share of hop pairs |
|---|---|
| <=2 turns (a real relay) | **0.4%** |
| 3-10 | 14.9% |
| 11-30 | 56.9% |
| >30 | 27.8% |

median gap **20 turns**; the same zone commanded both hops only **6%** of the
time; and **87%** of hop pairs had the car complete an order *in between*.
A relay means the car is in transit through zones. Here the car was working and
was moved again later by an unrelated decision.

Consequence in the arithmetic. Cost per leg is `beta * 2.49km = 0.149`:

| hops | old effective share | net |
|---|---|---|
| 1 | 0.450 | +0.301 |
| 2 | 0.225 | +0.076 |
| 3 | 0.150 | **+0.0006** |
| 4 | 0.1125 | **-0.037** |
| 5 | 0.090 | **-0.059** |

23.8% of legs paid at or below 1.5x cost and 2.8% paid below cost outright, so
the `AGENTS.md` invariant (`relocator_share > beta * km / gamma`) was being
violed by the divisor itself. `scripts/diag_reloc_invariant.py` now asserts it
from the live config at every hop count: margin is a constant **3.01x**.

**What this is NOT.** This does not make relocation pay *sooner*. The credit is
still paid on ride completion, 21-40 turns after the decision, against an
immediate cost. The relocation asymmetry is untouched; only the divisor is gone.

Measured alongside it, for the next decision: of 131 relocation legs that
arrived, 44.3% were ever dispatched to a rider, and **98.3% of those were
dispatched on the arrival turn itself** (`scripts/diag_reloc_time_to_use.py`).
So the evidence that a relocation worked is available almost immediately, and
55.7% of arrivals produce no evidence at all.

*Caveat:* both measurements are under a dispatch-first greedy policy, so the
distributions are policy-dependent. The chain mechanism is not -- it is in the
env, and the reward reads it identically whatever the policy does.

**Supersedes:** the relay-chain split introduced in `303c7a5` ("pay every zone
that relayed a car, not just the last hop"). That change was made to stop the
originating zone earning nothing on multi-hop repositioning. The measurement
says the originating zone earned nothing because the divisor was dividing, not
because it went untracked, so the fix was treating a coincidence as a relay.

---

## D48 — the relocating zone is paid when its car is DISPATCHED, not on completion

**Change:** `relocator_share * gamma` moves from the ride's completion
(section 1) to the dispatch of the relocated car (new section 1c). Section 1
then pays the dispatcher only `1 - relocator_share` for that ride, so the pair
still pays exactly one gamma between them. A cancellation reverses the
relocator share.

**Why:** the relocation decision pays `-beta * km = -0.149` on the turn it is
commanded. Evidence that it was right should arrive on a comparable timescale.
Measured over 131 relocation legs that reached their destination
(`scripts/diag_reloc_time_to_use.py`): **44.3% were ever dispatched, and 98.3%
of those were dispatched on the arrival turn itself** -- median 0 turns, not the
21-40 turns a completion credit took. The 21-40 figure is the completion lag,
not the time until the car finds work, and paying on completion threw away a
signal that was already available.

At `gae_lambda=0.95` a reward 60 turns back retained 2.5% of its magnitude at
the originating action (D46), so a completion-gated credit was arriving
effectively as noise. This removes the delay rather than trying to propagate
through it.

**Why this is not D28.** D28's arrival bonus paid for the *arrival*, which
happens whether or not the car is wanted, and was switched off for exactly that
reason. This pays for the *dispatch*, which requires a waiting rider: the 55.7%
of arrivals that are never dispatched earn nothing.

**Two accounting bugs found while verifying, both mine:**

1. *Double payment.* Section 1 read `_driver_relocator`, which 1c already pops,
   so the lookup returned `None` and paid the dispatcher the full gamma **on top
   of** the relocator share. One ride paid 1.45 gamma. Section 1 now reads
   `_order_relocator_paid`.
2. *Clawback sign.* The cancellation reversal first used `+=`, crediting the
   relocator instead of reversing it, overpaying every relocated cancellation by
   0.90. It showed up as a suspiciously constant `+0.90` per episode in
   `scripts/diag_served_credit_total.py`. Now `-=`.

The clawback is also no longer nested inside `if cancel_penalty > 0.0`. It is
part of the served-credit accounting, not a penalty; gating it on the penalty
weight silently disabled it whenever that weight was 0.

**Verified.** `scripts/diag_served_credit_total.py`, 3 episodes: emitted 243.10
against expected 243.10 (235 completed x 1.0 + 18 in-flight-at-cutoff x 0.45),
difference **+0.00**. A control run with `relocator_share = 0` emits exactly the
completed-ride count (86.00, 74.00, 75.00 on three seeds), confirming the
dispatcher term is never overpaid. `scripts/diag_reloc_invariant.py`: the
relocator margin is a constant 3.01x at every hop count. 33 tests pass.

**Not yet measured.** Whether this makes relocation *learn*. The acceptance test
is unchanged: relocation precision against the 4.1% chance baseline measured by
the random control arm in D44. Everything above establishes that the credit is
correct and prompt, not that a policy exploits it.

---

## D49 — the dispatcher is paid the full gamma; the relocator share is a bonus

**Change:** section 1 pays the dispatcher `gamma` unconditionally. Previously
(D48) it paid `1 - relocator_share` whenever the car had been relocated.

**Why:** D48's split was a mistake in attribution, not just arithmetic. The
dispatching zone did identical work either way -- it sent a car to a rider and
the ride completed -- so it earned 45% less for using the car another zone had
pre-positioned for it. That penalised precisely the outcome rebalancing exists
to produce: zone B was better off ignoring the car zone A thoughtfully sent it
and using one it already had.

The penalty was not diluted by cooperation either. At `team_share=0.2` the local
term carries 80% of the weight, so the 0.45 shortfall was nearly all real.

The split existed to stop one ride paying two gamma. But the relocator's share
is not a slice of the served credit -- it pays for a *different action*. Zone A
moved a car to where it was needed; zone B put it to work. Both happened, so
both are paid. A ride served by a rebalanced car is now worth 1.45 rather than
1.0, and that is the honest price of the two actions.

**Economics, unchanged where it matters.** The relocator's own decision is
still `+0.45` against `beta * 2.49km = 0.149`, a **3.01x** margin at every hop
count (`scripts/diag_reloc_invariant.py`). D49 does not make relocation more
profitable to the zone that decides it; it stops the *receiving* zone being
punished for the decision.

**Verified.** `scripts/diag_served_credit_total.py`, 3 episodes: emitted 301.60
against expected `235 x 1.0 + 148 x 0.45 = 301.60`, difference **+0.00**. 34
tests pass, including a direct regression that a zone earns the same whether it
uses a rebalanced car or its own.
