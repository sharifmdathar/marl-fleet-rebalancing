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

Consequence in the arithmetic. Cost per leg is `beta * km`. Using the 2.49 km figure that was current when D47
was written, that is `0.149`; re-measured on the current env the mean leg is
**2.95 km**, so the cost is `0.177`. The arithmetic below is shown at 2.49 km
because that is what the decision was made against -- the margin is 2.54x, not
3.01x. See the D50 entry.

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
still `+0.45` against `beta * km = 0.177` at the re-measured 2.95 km mean leg, a
**2.54x** margin at every hop count (`scripts/diag_reloc_invariant.py`). D49 does not make relocation more
profitable to the zone that decides it; it stops the *receiving* zone being
punished for the decision.

**Verified.** `scripts/diag_served_credit_total.py`, 3 episodes: emitted 301.60
against expected `235 x 1.0 + 148 x 0.45 = 301.60`, difference **+0.00**. 34
tests pass, including a direct regression that a zone earns the same whether it
uses a rebalanced car or its own.

---

## D50 — audit of the reward path: three findings, none of them fixed here

Run before training rather than after. Every number is from
`scripts/diag_reward_audit.py`, 3 episodes, seeds 3000-3002, dispatch-first
greedy, plus an independent line-by-line audit of `_compute_zone_rewards`,
`_flush_turn` and `reset`.

**1. The wrapper credits dispatches RideGym refuses.** `_flush_turn` records
`_order_dispatcher[oid]`, sets `_driver_commander[did] = zone_idx` to the
*dispatching* zone, and appends to `dispatched_this_turn` -- all before
`env.step()`. RideGym drops bids on non-pending orders silently, and
`_cancel_timed_out()` runs at the top of its `step()`, so an order in the last
`dt` of patience is advertised in the candidate menu, masked legal, and already
expired when picked. Measured void rate **1.1-2.5%** of recorded dispatches.
Consequences: the in-flight empty km of the abandoned leg is charged to the
dispatching zone, and `_driver_relocator.pop(did)` destroys the sending zone's
credit for a relocation that was never cancelled. Cross-zone attribution damage
measured **0** under a greedy policy, because the voided bids mostly hit
drivers already at their destination; the phantom records and lost credits are
still real.

**2. The break-even invariant was checked against a stale distance.**
`scripts/diag_reloc_invariant.py` hardcoded 2.49 km per relocation, carried from
an older run, and `config.py` quoted it as an environment property. Re-measured
per leg by `scripts/diag_reloc_leg_distance.py` over 367 legs: mean **2.95 km**,
median 3.21, p90 5.25. The asserted 3.01x margin is really **2.54x** at the mean
and 1.43x at p90. The invariant still HOLDS, so no price change is implied, but
every "3.0x" in D47-D49 was optimistic. The script now measures instead of
hardcoding and accepts a distance on argv; `config.py` records both numbers.

**3. Partial legs are paid in full.** `beta * km` is charged per turn for
distance actually flown, but `relocator_share * gamma` is paid in full at
dispatch. With `allow_relocating_dispatch=True`, dispatching a car mid-flight is
normal rather than exceptional. Measured: **98-100% of relocation credits exceed
the deadhead actually charged for the leg flown**. So the effective margin on
most relocations is larger than the config arithmetic implies, and the
documented invariant does not bind on those events.

**Deliberately not fixed here.** Each of these is a judgement call about what
the reward *should* say, and all three change the economics:

- (1) is a bug and should be fixed, but the fix is to verify dispatch against
  RideGym's outcome, which touches the dispatch path rather than the reward.
- (3) is arguably correct as-is -- a relocation that gets a car working sooner
  *should* pay, since the car did earn. Making credit proportional to distance
  flown would penalise opportunistic dispatch, which is good behaviour.
- (2) needs no code change; the numbers are now measured.

**Also ruled out**, so these are not open: no order is counted twice in the
waiting term; the waiting zone attribution matches the acting zone; team
blending is applied to the final reward and no baseline reads an unblended
value; `reset()` clears all seven tracking structures; `_newly_completed_ids`
cannot return an id twice; the `events` dict cannot survive a turn, so no
distance is charged twice; `_prev_served` is dead code (written, never read).

---

## D51 — a dispatch is credited only if RideGym honours it

**Change:** `_flush_turn` records dispatch intent in `_pending_dispatch` and
commits `_order_dispatcher`, `_driver_commander` and the
`_driver_relocation_target` pop only after `env.step` returns, and only for
orders whose status is ASSIGNED/ONBOARD/COMPLETED **and** whose
`assigned_driver` matches the requested driver. `dispatched_this_turn` is built
from the confirmed set, not the requested set.

**Why:** fixes D50 finding 1. RideGym's `_cancel_timed_out()` runs at the top of
its `step()`, so an order in the last `dt` of patience is advertised in the
candidate menu, masked legal, and already expired when the agent picks it. Its
assignment step then drops the bid silently. The wrapper was recording the
dispatch regardless, which produced three phantom attributions per refusal:

1. `_order_dispatcher[oid]` named a zone that never dispatched;
2. `_driver_commander[did]` was overwritten to the *dispatching* zone, so the
   abandoned relocation leg's remaining deadhead was billed to the wrong zone --
   and it persisted, since nobody re-commands a driver already in flight;
3. `_driver_relocator.pop(did)` destroyed the sending zone's credit for a
   relocation that was never cancelled.

**Measured.** Void rate **1.1-2.5% of recorded dispatches -> 0.0%** over 3
episodes (`scripts/diag_reward_audit.py`). Cross-zone misattribution was already
0 under a greedy policy because most voided bids hit cars already at their
destination; the phantom records and the lost credits were the real damage, and
both are gone.

**Regression test.** `scripts/diag_dispatch_void.py` rewinds a completed order to
CANCELLED -- the state a timed-out order is in when the wrapper offers it -- and
asserts `_order_dispatcher`, `_driver_commander` and `_driver_relocator` are all
untouched.

**Unchanged and re-verified.** `scripts/diag_served_credit_total.py`: emitted
302.50 vs expected `235 x 1.0 + 150 x 0.45 = 302.50`, difference **+0.00**.
`scripts/diag_reloc_invariant.py`: margin 2.54x at the measured 2.95 km mean leg.
34 tests pass.

**Still open, deliberately.** D50 findings 2 and 3. Finding 2 needs no code
change (the numbers are now measured). Finding 3 -- full credit for a partial leg
-- is arguably correct: a relocation that puts a car to work sooner did earn, and
making credit proportional to distance flown would penalise opportunistic
dispatch, which is desirable behaviour. Not changed without a decision.

---

## D52 — purge two documents that carried withdrawn claims

Deleted `docs/relocation-quality.md` and `docs/HANDOFF.md`. Neither was a live
result; both were still quotable, which is the failure mode `AGENTS.md` warns
about ("a stale number is worse than a missing one, because it gets cited").

**`relocation-quality.md`** was the more dangerous of the two:

- It taught the break-even invariant using `relocator_share = 0.35` and a 2.49 km
  leg. Both are superseded -- the share is 0.45 (D47) and the measured leg is
  2.95 km (D50 finding 2). Following that document's own advice would have led
  to a beta that breaches the invariant.
- It asserted relocation targets at "12.5% chance over 8 neighbours" and that
  argmax is **below** chance at 6.6%. D44 formally withdrew both: the legal
  target set averages ~1.12 per turn, the true chance rate measured by a random
  control arm is **4.1%**, and RL is therefore **at** chance, not below it. The
  document was the canonical source for a claim the decision log had already
  retracted.
- Its scenario table is env 2.7-era.

**`HANDOFF.md`** described env 2.3 and decisions D28-D34, roughly fifteen
entries behind, and closed with tuning instructions ("re-tune LP on env 2.3")
premised on invalidated measurements.

**Kept.** `decisions.md` deliberately retains the withdrawn numbers. D44's
correction is only meaningful next to the claim it corrects, and `AGENTS.md`
requires a correction to be recorded where the original was made rather than
quietly superseded.

**Dangling references fixed.** `scripts/beta_sweep.py` computed its safe/breach
thresholds from the stale 2.49 km, and so reported `beta=0.15` as safe; at the
measured 2.95 km it is a 1.02x margin, effectively break-even. It now defaults
to 2.95 and accepts an override. `diag_reloc_invariant.py`,
`competence_two_zone.py` and a `config.py` comment pointing at the deleted files
were repointed at live sources. The `config.py` comment also asserted that
`nyc_window_limit` was an "open decision that MUST be settled before training";
it was settled (`NYCConfig` sets 130) and the comment was misleading.

---

## D53 — relocation targeting is demand-blind, and it is not the reward

Measured, not hypothesised. Three probes, each ruling out one explanation.

### 1. The earlier precision numbers were confounded

`diag_reloc_leg_outcome.py`: **7 of 7** relocation legs were cut short by an
opportunistic dispatch (`allow_relocating_dispatch`, D14). The car is handed to a
rider mid-journey, so the zone it *ends up in* reflects where a rider was, not
where the policy aimed. Every earlier figure that compared aimed-zone to landed-
zone -- including D44's 4.1% chance baseline -- was therefore measuring the wrong
thing.

`diag_targeting_at_decision.py` measures at the instant of the action instead:

| | IPPO | MAPPO |
|---|---|---|
| mean legal targets per move | **13.55** | 13.77 |
| chance (1/legal targets) | **8.40%** | 7.60% |
| chose best legal target | 6.40% | 6.08% |
| mean demand at chosen target | **0.30** | 0.31 |
| mean demand at best available target | **1.69** | 1.77 |
| misses where the neediest zone WAS offered | **91.9%** | 91.4% |

Two corrections fall out of this. D44's "legal targets average 1.12/turn" is
**stale** -- it predates D45, which removed the neighbour-only mask; the real
figure is 13.5 of 15. And D44's conclusion that "the destination is largely
unexpressible (0.5% reachability)" **no longer holds**: 91.9% of misses had the
right zone on the menu.

`diag_reloc_opportunity.py` also killed the "cannot" explanations: IPPO relocates
on **84.7%** of zone-turns where an idle car and a legal target coincide, MAPPO
on 71.4%. The low relocation *rate* is because a zone holds an idle car on only
8.8% of zone-turns.

### 2. The reward does reward good targeting

`diag_reward_targeting_sensitivity.py`, MAPPO and IPPO across two seeds:

| aim class | paid |
|---|---|
| best legal target | 58.6% / 45.5-66.7% |
| non-best but non-empty | 58.1% / 53.8-60.9% |
| empty zone | **37.8% / 37.2-41.4%** |

Aiming at a needy zone is paid ~20pp more often than aiming at an empty one. So
there is gradient being left on the table, and the policy is not optimising
correctly.

### 3. The network reads demand magnitude but not direction

`diag_logit_sensitivity.py`, finite differences on the real network:

| input | IPPO L1 | MAPPO L1 |
|---|---|---|
| `all_zone_state` col2 (pending) | 3.02 | 2.90 |
| `all_zone_state` col3 (deficit) | 1.83 | 2.78 |
| `demand_hist` | 3.80 | **15.26** |
| `neighbor_mask` | 0.005 | 0.009 |

The logits move substantially -- the signal is read. But placing 100% of demand
in each of 16 zones in turn, the argmax relocation target **never changed**:
1/16 correct, chance 6.2%.

The cause is in `ZoneAttentionEncoder.forward`. `zone_pool` is a softmax-weighted
**sum** of per-zone embeddings, and every action logit is produced from that one
pooled vector. A weighted sum is permutation-invariant over zones: it can encode
"there is demand somewhere" but cannot bind that demand to a particular
destination. **Magnitude is read; direction is discarded.**

### Ruled out as external failures

| check | result |
|---|---|
| `obs_to_vector` order vs `obs_dim` | match, 279 = 279 |
| demand column recoverability from the vector | **608/608 exact** |
| `demand_hist` live during episodes | 98.3% of agent-steps |
| demand reaches the logits | yes, L1 2.9-15.3 |

No plumbing bug. The loss is architectural.

### The precondition for fixing it

`diag_target_learnability.py`: a supervised probe on the *existing*
observation, given the legal-target menu, recovers the best target at:

| seed | top-1 | chance | lift |
|---|---|---|---|
| 8000 | 71.6% | 9.4% | **7.6x** |
| 8100 | 64.6% | 8.6% | **7.5x** |
| 8200 | 75.3% | 9.1% | **8.3x** |
| 20000-sample run | 87.5% | 9.1% | **9.6x** |

**The information is present and learnable from the current observation.** The
architecture, not the features, is what loses it. That is what made D54 worth
building rather than changing the observation.

---

## D54 — destination-conditioned relocation head (attention arch only)

**Change.** In `ZoneAttentionEncoder.forward`, the relocation columns of the
action logits are no longer produced solely from the pooled vector `h`. Each
candidate destination `j` now gets its own logit:

    dest_logits[j] = <W_dest · zone_emb[j], W_self · self_ctx> / sqrt(H)
                   + w · raw_feats[j]              # demand/car counts, direct
                   + g · (centroid[j] − self_centroid)

and those are added to the (zero-initialised) relocation columns of the pooled
head. `zone_emb` is scored per destination and **never pooled on this path**.

**Why.** D53 established the policy reads demand magnitude but cannot bind it to
a destination: 16 controlled demand placements produced an identical argmax
target, because `zone_pool` is a softmax-weighted sum and therefore
permutation-invariant over zones. The fix is to stop destroying zone identity
before the decision.

Three terms, each addressing a measured gap:

1. **Key/query score** keeps the structured zone embedding usable, but scored per
   destination instead of averaged.
2. **Direct raw-feature bias** gives the head demand and supply counts per
   destination *without* passing through an embedding that was pooled away. This
   is the term the probe showed is learnable.
3. **Travel cost** lets deadhead distance enter the choice rather than only being
   billed afterwards.

**Scope.** Attention arch only; the flat MLP is untouched and still builds.
Dispatch and no-op logits are unchanged, so the **action space, observation
space and rollout buffer are all unchanged** -- no env version bump. The pooled
head's relocation rows are zeroed at init so the two heads cannot compete.

**Tests.** `tests/test_destination_conditioned_head.py`, 4 tests. They pin the
*structural* property -- the relocation logit vector responds to where demand is,
each destination's logit depends on its own features, and the pooled head does
not also own the relocation columns. They deliberately do **not** assert that the
argmax follows demand: at random init the per-zone constants legitimately
dominate one unit of demand, so that would be asserting convergence. Whether the
argmax now follows demand is the acceptance test
(`scripts/diag_targeting_at_decision.py`), run after training.

38 tests pass.

**Invalidates.** All D51 checkpoints -- the head is new, so their weights do not
load. Deleted with the runs.

---

## D54 acceptance test — pre-registered before results were seen

Written down before the D54 runs finished, so the outcome cannot be rationalised
after the fact. Three of today's reported numbers were artefacts (the 3.01x
margin, the 4.1% chance baseline, "output ignores demand"), each caught only by
checking. The bar is fixed here in advance.

**Primary test.** `scripts/diag_targeting_at_decision.py`, decision-time
comparison, >=2 seeds per arm, checkpoint at plateau.

| arm | chance baseline | D51 (pooled head) | pass requires |
|---|---|---|---|
| IPPO | 8.40% | 6.40% | > 8.40% |
| MAPPO | 7.60% | 6.08% | > 7.60% |

Chance is `1 / mean legal targets`, measured on the same run, not assumed.

**Secondary tests**, each independently informative:

| test | what it rules out |
|---|---|
| mean demand at chosen target vs best available (D51: 0.30 vs 1.69) | coarse anti-targeting |
| share of misses where the neediest zone WAS offered (D51: 91.9%) | residual expressibility |
| `diag_logit_sensitivity.py` argmax across 16 demand placements (D51: 1/16) | whether the head binds direction at all |
| relocation rate conditional on having a car + target (D51: 84.7% IPPO) | regression to "won't relocate" |

**Interpretation fixed in advance:**

- **Pass** = precision above chance on both arms AND argmax tracks demand. D54
  is supported.
- **Fail on precision, argmax tracks** = the head now expresses the choice but
  PPO did not learn it in 60 iterations. That is a training-budget or
  optimisation result, NOT evidence against D54, and I will report it as such.
- **Fail on argmax too** = the head is still effectively permutation-invariant,
  or the travel-cost / raw-bias terms dominate. D54 did not work; revert it.
- **Precision falls below D51** = the new head actively hurt; revert and treat
  the pooled head as the better prior.

I will also report the IPPO-vs-MAPPO ordering, but neither arm beating the other
is evidence about D54 -- they differ only in the critic.

**Not claimed in advance:** that D54 improves service rate. It is a targeting
fix; if targeting improves and service rate does not, that is a real and
interesting result about the two being separable, not a failure.

---

## D54 RESULT — targeting fixed; passes the pre-registered test on 3 seeds

Run at 60/60. Best val SR: IPPO 73.7% @ it15, MAPPO 74.6% @ it30.

**Primary criterion, `checkpoint_best.pt`, 3 seeds:**

| arm | chance | D51 | D54 seed 8000 | 8100 | 8200 | verdict |
|---|---|---|---|---|---|---|
| IPPO | 7.96-8.72% | 6.40% | **13.04%** | **18.22%** | **15.40%** | **PASS** |
| MAPPO | 7.88-8.16% | 6.08% | **15.22%** | **17.67%** | **20.61%** | **PASS** |

Both arms above measured chance on every seed, where both were below it before.
Chance is recomputed per seed rather than assumed.

**Secondary criterion, argmax tracks demand over 16 controlled placements:**
D51 1/16 (chance 6.2%) -> D54 **8/16 IPPO, 5/16 MAPPO**.

**Corroborating:** mean demand at chosen vs best target 0.18/0.17 -> **0.31-0.39**.

**Controlled stress test** (`scripts/diag_targeting_stress.py`), because a
1920-turn average cannot resolve a change confined to ~8% of turns:

| scenario | greedy | IPPO | MAPPO |
|---|---|---|---|
| 1 zone, 100% demand there | 0.8% | **28.8%** | **43.6%** |
| 2 zones, hard switch at 25min | 4.1% | **17.2%** | **29.3%** |

(service rate; greedy never relocates and is the floor)

**Why aggregate service rate did not move.** Arithmetic, not a null result:
relocation is ~8% of zone-turns, so ~154 relocations per episode; a +10pp
targeting gain is ~15 better aims; at the measured 44% dispatch conversion that
is ~7 extra dispatches, ~+3 reward on a base of ~235. **~1.3% of total reward**,
below the noise floor of a service-rate mean. The stress test removes the
attenuation by concentrating the problem.

**Caveat, recorded:** greedy scoring 0.8% under 100% single-zone demand shows
that scenario is near-degenerate in absolute terms. It is a *targeting*
benchmark, not a performance claim.

**Caveat, recorded:** IPPO's selected checkpoint is iteration 15 of a run that
then degraded to 63.7% by iteration 60. IPPO's numbers therefore come from a
barely-trained policy and are not a fair IPPO-vs-MAPPO comparison.

### Why the ratio is still 0.33 and not 1.0

`scripts/diag_dest_head_decomposition.py`, over 61 real relocation-opportunity
observations:

| term | share of logit variation | argmax agreement with full head |
|---|---|---|
| key/query | **61.4% IPPO / 48.5% MAPPO** | **100%** |
| raw bias | 5.8% / 5.9% | 86.9% |
| travel | 13.5% / 14.7% | 3.3% |

**The key/query path decides the destination on its own.** The raw-feature term
-- the one the learnability probe showed is *sufficient* on its own (87.5% top-1,
9.6x chance) -- contributes under 6% of the variation, because it is a single
`Linear(8, 1)` while the probe used an MLP.

So the head is no longer representation-limited: it routes demand through the
Tanh zone embedding, which is why it works at all. But the direct path the probe
proved sufficient is starved. `travel` burns 14% of the variation and agrees
with the full head only 3.3% of the time, so it is close to inert.

---

## D54 CORRECTION — the head was never the bottleneck; my acceptance test sampled

**The planned D55 change was cancelled before making it**, because the premise
was tested first and failed.

### What I was about to do, and why it was wrong

The D54 decomposition showed the raw-feature path contributing under 6% of
logit variation while the key/query path decided the destination alone. I
proposed replacing `Linear(8, 1)` with an MLP on the theory that the raw path
was capacity-limited, since the learnability probe reached 87.5% with an MLP.

That theory has two competing versions:

  (A) CAPACITY -- the raw path is too small to express what the probe expressed,
      so widening it helps.
  (B) OPTIMISATION -- the raw path is already good and the optimiser simply
      preferred key/query, so widening it changes nothing.

`scripts/diag_raw_path_quality.py` separates them, over 406 real relocation
opportunities, scoring each term's argmax by how needy the chosen zone is:

| selector | demand ratio | top-1 |
|---|---|---|
| random legal target | 0.29 | 12.1% |
| **full head** | **0.99** | **82.0% IPPO / 80.5% MAPPO** |
| key/query alone | 0.98 | 79.3% / 82.0% |
| raw bias alone | 0.95 | 66.0% |
| travel alone | 0.28 | 15.8% |

**The raw path is already at ratio 0.95 and 66% top-1 on its own.** It was never
capacity-limited. Hypothesis (B) holds, so the MLP change would have been
cosmetic at best. Cancelled.

### The real problem: my acceptance test measured sampling, not the policy

The full head reaches **82% top-1 / 0.99 ratio** under argmax. Yet the
pre-registered test reported 13-20% and ratio 0.33. Both numbers were correct
and they were measuring different things:

`diag_targeting_at_decision.py` called `policy.act(..., deterministic=False)`,
so it recorded **sampled** actions. The decomposition took **argmax**.

Re-running the same test with `--deterministic`:

| arm | chance | sampled | **argmax** | ratio sampled | **ratio argmax** |
|---|---|---|---|---|---|
| IPPO | 7.74% | 13.04% | **51.42%** | 0.31 | **0.94** |
| MAPPO | 7.23% | 15.22% | **55.06%** | 0.34 | **0.90** |

**The policy's PREFERRED destination is right 51-55% of the time, against ~8%
chance -- and picks zones holding 1.50-1.55 unmet riders where 1.59-1.72 were
available.** That is close to the 87.5% the supervised probe achieved on the
same features, and it is a completely different picture from the one I reported.

### What this means

1. **D54 worked, and better than the pre-registered test could show.** The
   original test was sound as a test of *executed* behaviour, and it did what it
   was designed to do -- it caught that D51 was below chance. But it understated
   D54 because it scored a stochastic policy.
2. **No architecture change is needed.** The head extracts what the features
   offer; the gap to the probe is small and is not representational.
3. **The remaining gap is action selection, not representation.** A policy whose
   argmax is 51-55% correct but whose samples are 13-20% correct is
   under-confident: its relocation distribution is too flat. That is an entropy
   or training-budget question, and possibly an evaluation-protocol question.
4. **`--deterministic` is now the default for decision-quality tests**, and
   sampling is reported separately as executed behaviour. Both are legitimate;
   conflating them is not.

**Method note.** This is the fourth time today a number I reported measured the
wrong thing (3.01x margin, 4.1% chance, "output ignores demand", and now
sampling-vs-argmax). Every one was caught by testing a premise rather than
accepting a conclusion. The MLP change was cancelled *because* the premise was
checked first.

---

## D56 — the waiting penalty is ~90% of the reward and dwarfs the rest

Asked directly: are the number of cars, the distance cost and the waiting time in
good proportion? **No.** Measured per episode under each trained policy
(`scripts/diag_reward_balance.py`, 3 episodes, seeds 9000+):

| arm | served | empty km | deadhead | cancel | wait | NET |
|---|---|---|---|---|---|---|
| greedy | 177.0 | 259.2 | −15.6 | −180.0 | **−3991.4** | −4009.9 |
| IPPO | 263.0 | 1386.4 | −83.2 | −88.0 | **−3991.4** | −3899.6 |
| MAPPO | 279.0 | 1382.6 | −83.0 | −77.0 | **−3991.4** | −3872.3 |

Share of total absolute reward movement:

| arm | served | deadhead | cancel | **wait** |
|---|---|---|---|---|
| greedy | 4.1% | 0.4% | 4.1% | **91.5%** |
| IPPO | 5.9% | 1.9% | 2.0% | **90.2%** |
| MAPPO | 6.3% | 1.9% | 1.7% | **90.1%** |

**The waiting penalty is ~15x the served credit and ~45x the deadhead cost.**
Serving a rider -- the thing the project is about -- is **6% of the signal.**
Deciding *where* to send a car moves a term worth 1.9%.

### Why: it is an integral, not a weight

```python
local_rewards[zi] -= cfg.alpha * sum(waiting_times_in_zone) / order_timeout
```

Charged **every turn** a rider waits, and summed over all pending riders in all
zones. Served credit is charged **once**, at completion. So waiting is a running
integral and service is a one-shot reward; their ratio is set by *how many
riders are pending on average*, not by `alpha` vs `gamma`.

Per-rider the constants look sane (`alpha/gamma = 0.2`), which is why this was
never caught. But one rider left unserved accumulates `alpha * sum(t)/timeout`
over 10 turns to **0.90**, against **1.00** for serving them -- and there are
pending riders in most zones on most turns.

### This is not the env-2.7 failure, and the difference matters

`relocation-quality.md` recorded served and deadhead cancelling almost exactly,
which made the good-move/wasted-move signal vanish. That no longer holds:
served + deadhead is **+180 to +196**, comfortably positive. **Deadhead is not
cancelling service.** The problem is the opposite one -- everything else is
being drowned out.

### Consequence

The policy is overwhelmingly optimising "reduce accumulated rider-minutes",
not "serve riders" or "don't drive empty". That is a defensible objective in
spirit, but it means:

- the served credit, and therefore the relocation credit derived from it, is a
  **6% perturbation** on the objective;
- the deadhead discipline the `AGENTS.md` invariant depends on is a **1.9%
  perturbation**;
- a policy can improve its reward substantially by reducing time-to-assignment
  without relocating better, which is consistent with the modest aggregate
  service-rate response to D54.

### Options, not decided here

1. **Cap or discount the waiting term** so it stops integrating without bound --
   e.g. charge per-rider once at timeout rather than every turn, or clip
   `sum(waiting)` per zone per turn.
2. **Rescale `alpha`** so the four terms sit in comparable proportion. Crude, and
   fleet-size dependent, which is what D37 was written to avoid elsewhere.
3. **Leave it**, and report every result as a statement about
   minimise-waiting rather than maximise-service.

Option 3 is defensible but changes what the headline metric means, so it is a
reporting decision as much as a technical one. This is the user's call: it is a
reward change, it would bump the env version, and it invalidates every D54
checkpoint.

---

## D57 — feasibility check: the fleet is at or slightly over capacity

Asked as a physical question, separate from the reward weighting in D56: are the
number of cars, the number of orders, the distance to cover and the car speed in
proportion such that the goals are actually achievable?

`scripts/diag_feasibility.py`, 3 episodes, val split:

| quantity | value |
|---|---|
| zones | 16 |
| drivers (fleet) | **26** (1.62 per zone) |
| orders per episode | 130.0 |
| orders per hour | 130.0 |
| completed per episode | 59.0 |
| cancelled per episode | 55.0 |
| still open at cutoff | 16.0 |
| mean ride duration | **6.55 min** (median 5.50, p90 12.00) |
| empty km per completed ride | **1.46 km** |

Little's Law, `cars_required = arrival_rate x time_per_job`:

| time per job | cars needed | vs fleet of 26 | verdict |
|---|---|---|---|
| ride time only (6.6 min) | 14.2 | **0.55x** | feasible |
| ride + 1x ride of deadhead (13.1 min) | 28.4 | **1.09x** | **infeasible** |
| ride + 2x ride of deadhead (19.7 min) | 42.6 | **1.64x** | infeasible |

### The answer: marginal, and it tips on deadhead

**The scenario is feasible only if deadhead is cheap.** Deadhead is 1.46 km per
completed ride against a mean ride of 6.55 min. Converting at the ride's own
speed:

| if the mean trip is | implied speed | deadhead time | as % of ride |
|---|---|---|---|
| 2.0 km | 0.32 km/min | 4.59 min | **73%** |
| 3.0 km | 0.48 km/min | 3.06 min | **49%** |
| 4.0 km | 0.64 km/min | 2.29 min | **37%** |

At a 2 km mean trip the fleet is **9% short** and the task is physically
impossible. At 4 km it has comfortable headroom. The scenario sits right on the
boundary, and **which side depends on a parameter never recorded.**

This is consistent with everything observed today and previously unexplained:

- **55 cancelled vs 59 completed per episode.** Cancellations are not a policy
  failure at 5-minute patience; they are queue overflow.
- **16 orders still open at cutoff**, in a 60-minute episode, on a fleet with
  26 cars.
- **Service rate stuck near 77%** regardless of reward or architecture. A
  throughput ceiling does not move when you improve decisions.
- **The competence test scoring 0.8-8%** under 100% single-zone demand. That
  concentrates all 130 orders into one of 16 zones, making the effective arrival
  rate at that zone ~16x, and the scenario categorically infeasible. It was never
  a fair benchmark.

### What is missing, and it is a config gap not a result

The trip distance distribution and the vehicle speed are **not in the config and
not logged**. `empty_km` and ride duration are recorded, which is why the deadhead
burden can be bounded but not resolved. Until mean trip distance is recorded, the
feasibility of the scenario is a range (0.55x to 1.09x fleet), not a fact.

Recording mean trip distance per episode is a small, non-behavioural logging
addition and would settle this permanently.

### Consequence for results reported so far

Every service-rate number this session -- 77% on natural demand, 43.6% on the
single-zone stress test -- is a **policy measured against a throughput-capped
target**. That does not invalidate the targeting results, which are ratios against
measured chance and therefore unaffected by the ceiling. It does mean service
rate should not be read as "how good the policy is", only as "how much of the
achievable throughput it captured".

---

## D58 RESULT — alpha rescale did not fix the tradeoff; beta is the binding weight

### What was done

1. **Trip distance logging** (D57 follow-up). Order `origin`/`destination` are
   in DEGREES, so distance needs haversine; the first version divided by 1000
   as if metres and silently reported 0.000 km. Now recorded per episode:
   **mean trip 2.77 km, p90 5.63 km**.

   **Feasibility closed: 0.83x the fleet** (job = 10.0 min, 21.7 cars needed vs
   26 available). The scenario is feasible with modest headroom -- not the
   1.09x infeasibility the unbounded range allowed for. Recorded deadhead is
   1.46 km/ride = **53% of ride time** at the implied 0.42 km/min.

2. **`alpha` 0.2 -> 0.0132.** Waiting was 90% of all reward movement (measured
   -3991/episode against served +263). New shares: served 37.7%, wait 37.7%,
   cancel 12.6%, deadhead 11.9%.

### The result: the tradeoff is unchanged

Paired, 6 identical seeds, val split:

| arm | SR% | empty km | avg wait | SR/100km | reloc |
|---|---|---|---|---|---|
| greedy | 52.73 ±4.38 | **78.1 ±8.1** | **2.56 ±0.18** | 68.0 | 0 |
| lp | 67.29 ±3.97 | 154.3 ±9.6 | 2.67 ±0.27 | 43.7 | 64 |
| **IPPO (D58)** | 72.25 ±4.20 | 475.8 ±13.0 | 3.00 ±0.20 | 15.2 | 160 |
| **MAPPO (D58)** | 71.70 ±4.43 | 480.8 ±13.3 | 2.92 ±0.16 | 14.9 | 165 |

Against D54 (same seeds): IPPO 74.32 -> 72.25 SR, 470.4 -> 475.8 km. **The
rescale cost ~2pp of service rate and bought nothing on deadhead.**

### Why: deadhead is the cheap term, and alpha was never the problem

Relative to deadhead, the other terms are enormous:

    served / deadhead  = 263 / 83   = 3.16x
    wait   / deadhead  = 263 / 83   = 3.16x
    deadhead           = 11.9% of total movement

**One served ride buys 11 relocations worth of deadhead** (`beta * 2.9 km = 0.174`
against a served credit of 1.0). Relocating is so cheap that the policy does it
160 times an episode and pays almost nothing for the privilege. Draining the
waiting term only removed a term that was *reinforcing* that behaviour, since
waiting was 90% of the signal and the policy was already minimising it.

### The actual constraint: beta, bounded above by the AGENTS.md invariant

`beta` cannot simply be raised, because `relocator_share > beta * km` is the
invariant `AGENTS.md` records from env 2.8:

| beta | deadhead per 2.9 km relocation | rides to pay back | invariant (share 0.45) |
|---|---|---|---|
| **0.06 (current)** | 0.174 | **5.7** | 2.59x -- satisfied |
| 0.10 | 0.290 | 3.4 | 1.55x -- satisfied |
| **0.15** | 0.435 | 2.3 | 1.03x -- marginal |
| 0.20 | 0.580 | 1.7 | **0.78x -- VIOLATED** |

**`beta` can rise only to ~0.156** before a perfect relocation becomes
net-negative and the policy learns to churn -- the exact env-2.8 failure.

So the two requirements are in direct conflict:
- make deadhead expensive enough to discipline relocation -> needs beta >= ~0.15
- keep a perfect relocation profitable -> needs beta < ~0.156

They meet in an extremely narrow band. **Raising `beta` alone cannot fix this**,
and neither can `alpha`. The lever that can is `relocator_share`: raising it
lifts the floor that `beta` is bounded by, opening the range.

This is recorded, not acted on -- it is a reward change with a real design
tension, and the choice of which term to move is the user's.

---

## D60 — evaluate with argmax, not sampled actions

**Change:** none to the env or the algorithm. `eval_tradeoff.py` and the
targeting probes gained `--deterministic`, and every decision-quality
measurement now defaults to argmax.

**Why.** D54's correction established that the destination-conditioned head's
argmax is **51-55% correct** against ~8% chance, while its *sampled* actions are
only 13-20%. The policy knows which zone it wants and discards roughly two thirds
of that at sampling time. Sampling is correct for training -- it is what
explores -- but wrong for measuring a policy, because it reports what the policy
occasionally does rather than what it prefers.

**Result**, same D59 checkpoints, same 6 paired seeds, sampled -> argmax:

| arm | | SR% | empty km | avg wait | reloc |
|---|---|---|---|---|---|
| IPPO | sampled | 70.98 | 482.0 | 3.02 | 162 |
| IPPO | **argmax** | **74.87** | **468.8** | **2.83** | **131** |
| MAPPO | sampled | 73.98 | 467.1 | 3.00 | 161 |
| MAPPO | **argmax** | 73.70 | **450.8** | **2.78** | **130** |

| arm | d SR | d empty km | d wait |
|---|---|---|---|
| IPPO | **+3.89pp** | **-13.2km** | **-0.19min** |
| MAPPO | -0.28pp | **-16.3km** | **-0.22min** |

**This is the only intervention today that moves all three in the right
direction.** Empty km and waiting fall for both arms, and IPPO gains 3.89pp of
service rate as well. It costs nothing: no retraining, no reward change, no
hyperparameter. The policy was always capable of this; the evaluation was
sampling away the evidence.

Relocation count drops 162 -> 131 and 161 -> 130, so part of the deadhead saving
is simply relocating less. That is the entropy bonus doing its job: confident
policies churn less.

**Still not the project's bar.** Against greedy the tradeoff persists (argmax
MAPPO: +20.97pp SR, +372.7km empty, +0.22min wait). This improves the *policy*
on all three; it does not move the *frontier*. See D61.

---

## D62 — alpha restored to 0.2; D58 reversed

**Change:** `alpha` 0.0132 -> 0.2. A direct reversal of D58.

**Why D58 was wrong.** D58 measured the MAGNITUDES of the four reward terms and
found waiting was 90% of all reward movement, then lowered it. It never asked
what each term TEACHES. Magnitude is not importance.

The waiting penalty is charged to the zone **containing the waiting rider**, so
it is the only term that distinguishes *which* zone needs a car. Served credit
pays for any completion regardless of where the car came from, so it cannot
teach targeting at all. D58 therefore deleted most of the targeting signal while
"fixing" a proportional imbalance.

**Measured**, NYC16 val, 3 paired seeds, argmax actions, precision PRIMARY:

| alpha | precision | chance | lift | SR% | empty km | wait |
|---|---|---|---|---|---|---|
| 0.0132 (D58) | 18.73% | 7.28% | 2.57x | 76.49 | 444.8 | **2.63** |
| 0.10 | 28.53% | 8.38% | 3.41x | 79.45 | 454.2 | 2.92 |
| **0.20** | **31.36%** | 7.78% | **4.03x** | **80.85** | **439.2** | 2.89 |

Higher alpha improves **targeting, service rate and empty km together**. Only
average wait is slightly worse. Against the D58 setting: precision +12.6pp,
service rate +4.36pp, empty km -5.6km.

The earlier 55.06% figure was D54 (alpha 0.2, before D59's deadhead
attribution), so the two effects are not fully separated; both moves are on the
same side and D62 is the configuration actually measured above.

**Also recorded: the entropy sweep failed and is not adopted.** `entropy_coef`
0.01 -> 0.003 -> 0.001, argmax eval on 6 paired seeds:

| coef | SR% | empty km | wait | precision |
|---|---|---|---|---|
| 0.01 | 73.70 | **450.8** | 2.78 | 20.95% |
| 0.003 | 73.69 | 476.7 | 2.67 | -- |
| 0.001 | **75.04** | 475.4 | **2.51** | **11.65%** |

ec0.001 has the best service rate and waiting of the sweep and the **worst
targeting measured anywhere** -- it serves riders by relocating
indiscriminately. Adopting it would have traded away the thing D54 fixed.

**Why the entropy lever does nothing here**
(`scripts/diag_entropy_by_block.py`, 200 relocation decisions per checkpoint):

| coef | relocate top-1 | relocate entropy | % of max | dispatch top-1 |
|---|---|---|---|---|
| 0.01 | 14.2% | 2.30 | 94.1% | 98.5% |
| 0.003 | 17.0% | 2.30 | 92.7% | 99.5% |
| 0.001 | 14.8% | 2.31 | 93.0% | 99.5% |

The relocation block sits at **93-94% of maximum entropy** (uniform over ~10.9
legal targets) while dispatch is near-deterministic. A 10x cut in the
coefficient moves top-1 by 0.6pp. At coef 0.01 the entropy term is 0.0026
against a policy loss of 0.00058 -- the bonus DOMINATES and still cannot
concentrate the distribution, because the destination logits span only ~0.4
nats across 10 destinations. Softmax over logits that tightly clustered is
uniform by construction. The lever that fits this evidence is the SCALE of the
destination logits (a temperature on `dest_logits`), not the entropy bonus --
and it would not change the argmax, so it is free. Not tried yet.

---

## D63 — RCA: why RL reaches 37% precision when a probe reaches 87%

Two distinct causes, measured. The second is a defect I introduced in D59.

### Cause 1 — the reward flags emptiness but does not rank destinations

`scripts/diag_reward_rank_resolution.py`, D62 checkpoints, 3 episodes, argmax.
Classifies every relocator payout by aim quality:

**IPPO**

| aim class | aims | paid | paid rate |
|---|---|---|---|
| best legal target | 148 | 84 | **56.8%** |
| other non-empty zone | 91 | 49 | **53.8%** |
| empty zone | 167 | 55 | **32.9%** |

```
gap best vs other-nonempty :  +2.9 pp   <- RANKING signal
gap non-empty vs empty     : +20.9 pp   <- EMPTINESS signal
```

**The reward is 7x more sensitive to "did you aim at an empty zone" than to
"did you aim at the best zone".** MAPPO is worse: +0.2pp ranking vs +7.3pp
emptiness.

So half the decision is rewarded and learnable, and half has almost no
gradient. That is why precision is 37% rather than 87%: the probe is handed the
answer, whereas RL must infer it from pay, and pay barely distinguishes the
candidates it actually has to choose between.

**And the learnable half is not fully learned either: 41.1% of IPPO's aims
still land on empty zones** despite a 20.9pp penalty for doing so.

### Cause 2 — D59 BROKE the expected-value invariant (defect)

The `AGENTS.md` invariant is that a *perfect* relocation must be profitable:
`relocator_share * gamma > beta * km`. D59 raised what the relocating zone pays
without raising what it earns. It now pays for the whole job:

```
cost = beta * (reloc leg 2.9 km + pickup deadhead 1.46 km) = 0.2616
```

The invariant's literal arithmetic still holds (0.45 > 0.262). **But the credit
is conditional** -- it fires only when the relocated car is actually dispatched,
which happens 56.8% of the time even for a perfect aim:

| aim class | dispatch rate | expected credit | net |
|---|---|---|---|
| best | 56.8% | 0.2556 | **-0.0060** |
| other non-empty | 53.8% | 0.2421 | **-0.0195** |
| empty | 32.9% | 0.1481 | **-0.1135** |

**Expected value is negative for every class of relocation, including the
best.** The policy is therefore *correct* to be ambivalent about relocating, and
the config comment asserting profitability is wrong because it ignores the
conditionality.

This also explains why raising `beta` never worked (D58): the binding
constraint was never `beta` alone, it was credit-per-unit-cost where the credit
only arrives ~57% of the time.

### What follows

The invariant needs restating with the measured conditionality:

    EV = relocator_share * gamma * P(dispatch | relocated) > beta * km_total

At the measured 0.568 that requires `relocator_share * gamma > 0.461`, i.e.
`relocator_share > 0.46` against a cost of 0.2616 -- the current 0.45 is
marginally below it. That is a ~2% shortfall, which is why relocation is
net-neutral at best rather than profitable.

Not changed here: it is a reward edit and the user's call. Recorded with the
measurement that establishes it.

---

## D64 — `relocator_share` 0.45 -> 0.65, restoring the expected-value invariant

**Change:** `relocator_share` 0.45 -> 0.65 in `EnvConfig` and `NYCConfig`.

**Why.** D63 established that D59 had broken the expected-value invariant.
D59 made the relocating zone pay for the WHOLE job it caused
(`beta * (2.9 km leg + 1.46 km pickup) = 0.2616`) without raising what it earns,
while the credit only fires when the relocated car is actually dispatched --
measured at **56.8% even for a perfect aim**. So every relocation was
net-negative in expectation:

| share | expected credit | cost | net |
|---|---|---|---|
| 0.45 (old) | 0.2556 | 0.2616 | **-0.0060** |
| **0.65 (new)** | **0.3692** | 0.2616 | **+0.1076 (1.41x)** |

The `AGENTS.md` invariant `share > beta * km` still passed at 0.45 (0.45 >
0.262), because that form ignores the conditionality. It is a step-function
reward, not an expected-value one.

**`scripts/diag_reloc_invariant.py` rewritten** to assert the conditional form
`share * gamma * P(dispatch) > beta * km`, with the measured 0.568 as an
explicit input rather than a hidden constant. Verified it correctly reports
0.45 as VIOLATED and 0.65 as HOLDS at 1.41x. Reward accounting re-verified
exact: emitted 222.40 vs expected 222.40, difference -0.00. 41 tests pass.

## D64 RESULT — single-seed: MAPPO targeting up, IPPO down (NOT a reported result)

Correction (2026-10-04): the original heading said "works for MAPPO". That
was a single-seed (3-episode) observation and is not a result under AGENTS.md
(>=5 seeds). The body below already says so; the heading now does too.

3 episodes, argmax, `alpha=0.2`, `share=0.65`:

| run | precision | chance | lift | SR% | empty km | wait |
|---|---|---|---|---|---|---|
| **d64 MAPPO** | **49.15%** | 9.03% | **5.44x** | 79.39 | 453.3 | 2.63 |
| d64 IPPO | 10.24% | 8.18% | 1.25x | 70.39 | 487.8 | 2.59 |
| d62 IPPO (prev) | 37.19% | 7.33% | 5.07x | 78.68 | 447.9 | 2.79 |

Targeting by iteration, so this is not one lucky checkpoint:

| checkpoint | MAPPO | IPPO |
|---|---|---|
| it10 | 44.37% | 16.84% |
| it30 | 38.98% | 9.19% |
| it35 | **49.08%** | 14.07% |

MAPPO is robust at ~44% mean and reaches 49.08% (6.14x chance) -- the best
targeting measured in this project. **IPPO regressed from 37.19% to ~13%,
consistently across iterations**, so it is a real regression rather than
selection noise.

Most likely cause is arm-specific: a higher relocator credit rewards relocating
more, and MAPPO's centralized critic can evaluate whether a relocation will pay,
so it learns to aim. IPPO's weaker critic cannot, so the extra credit buys
indiscriminate relocating instead -- consistent with IPPO performing 136.7
relocations against MAPPO's 133.2 at worse targeting.

**One seed each.** "share 0.65 helps" is supported for MAPPO and unproven in
general. `AGENTS.md` requires >=5 seeds for a reported result; this is debugging
output.

---

## STATUS: no proper model yet -- do not start the shift study

**LP dominates both RL arms on the frontier** (`scripts/eval_tradeoff.py`,
6 paired seeds, argmax, D64):

| arm | SR% | empty km | SR/100km |
|---|---|---|---|
| greedy | 52.73 | 78.1 | **68.0** |
| **LP** | **67.29** | **154.3** | **43.7** |
| IPPO | 70.95 | 483.9 | 14.7 |
| MAPPO | 74.99 | 468.8 | 16.0 |

LP reaches 67.29% at 154km; RL reaches 74.99% at 469km -- 3.0x the deadhead for
+7.7pp. On the efficiency `AGENTS.md` requires alongside service rate, RL is
**2.7x worse than LP**. Measuring demand-shift degradation with this would be
measuring a policy the baseline already dominates.

**The binding defect is still Cause 1 of D63, and it is NOT fixed.** The
relocator receives a flat 0.65 whether it sends a car to a zone with 1 rider or
10, so the reward cannot separate good targeting from mediocre targeting:

```
gap best vs other-nonempty :  +2.9 pp   <- RANKING signal (missing)
gap non-empty vs empty     : +20.9 pp   <- EMPTINESS signal (learned)
```

We are paying cars to move without saying **where**. That is why MAPPO reaches
49% while the supervised probe reaches 87% on identical features.

**The next step, not yet done:** scale the relocator credit by the deficit it
relieved, `credit = relocator_share * min(pending_at_target / reference, 1)`,
with `reference` measured (~1.7 riders, the mean demand at the best available
target). Same dispatch proof-gating as now, so it is not D28's arrival bonus
which paid without proof. Risk to watch: chasing a single hotspot at the expense
of breadth.

## D65 — scope: 16 busiest taxi zones, 26 drivers (confirmed by user 2026-10-04)
- Date: 2026-10-04, Status: accepted (user sign-off)
- Change: NYC study moved from 62 zones / 100 drivers / max_pending 30 /
  action space 93 to 16 zones / 26 drivers / max_pending 16 / action space 33.
  Zones are the 16 busiest taxi zones by train-window volume only (no val/test
  information in the environment definition). Driver and order caps keep the
  62-zone pressure ratios (1.61 drivers/zone, 5 orders/driver).
- Why: compute (62 zones ~4x what the laptop can run) and the brief's 5-10 zone
  guidance. Recorded here because the original log (D1-D34) was purged in
  `bc53c9c`, and the commit that made the change (`b02f899`) left no decision
  entry behind.
- Consequence: 62-zone artifacts are not comparable to 16-zone results. Do
  not cite numbers across the two configs. The 62-zone build remains in
  `scripts/build_zone_subset.py`'s source directory `data/nyc/` as reference.
- Revisit if: the advisor requires the full 62-zone scale for the final study.

## D66 — deficit-scaled relocator credit (env 3.1)
- Date: 2026-10-04, Status: implemented, NOT yet evaluated (no training run)
- Problem (D63 Cause 1): the relocator was paid a flat 0.65 whatever the
  destination's demand, so the reward could not rank targets (+2.9pp gap
  between best and other non-empty zones vs +20.9pp empty vs non-empty).
- Change: relocator credit = relocator_share * gamma * min(waiting_at_dest /
  reference, 1). `waiting_at_dest` = pending riders in the zone that dispatched
  the car (the car's destination when it was relocated). Clawback on cancel
  reverses the exact amount paid (stored with the payout), not the flat share.
  NYCConfig reference = 1.7 (mean demand at best legal target, D62). EnvConfig
  default 0 = off, so Bandra and old configs are unchanged. Config 3.0 -> 3.1.
- Verified: tests/test_deficit_credit.py (disabled = flat; saturates at the
  reference). Full suite passes.
- Not verified: whether this improves precision, empty km or wait. That needs
  a training run and >=5 seeds, which has not been authorised.
- Revisit if: policy chases one hotspot and loses breadth (HANDOFF s.2 risk),
  or empty km rises without a matching service/wait gain.

## D67 — decision-time targeting bonus (flag, default off)
- Date: 2026-10-05, Status: implemented, evaluated only on 2 seeds (pending)
- Problem: the decision-time targeting check (scripts/diag_targeting_at_decision.py)
  puts relocation choice at chance: 7.6% vs 7.7% random; the chosen zone carries
  0.42 unmet demand vs 1.76 for the best legal zone. The only destination signal is
  the flat relocator credit, paid late (about 39 turns, D44), sparsely (about a
  quarter of relocations earn usable credit), and never tied to destination demand.
- Change: when a relocation is executed, the sender receives
  targeting_bonus * unmet(target) / max over legal targets of unmet, where
  unmet = pending riders per zone. Paid in the same reward step. Default 0 (off);
  the flag is --targeting-bonus on train.py.
- Why this value: a relocation costs about beta * 2.95 km = 0.18. With bonus 0.2 a
  move to the best zone roughly breaks even, a move to a zone with half the best
  demand loses about 0.08, and a move to an empty zone loses 0.18. So the bonus
  only pays for well-aimed relocations.
- Known risks (research report, not verified in this repo):
  - It is not potential-based: it depends on the chosen action, so it does not
    preserve the optimal policy. Expect a bias toward the largest hotspot.
  - It ignores the distance term at the moment it pays. The cost is still charged
    during the drive, so a long move to a hotspot can still net negative.
  - It is a shaping term, so any gain must be checked against service rate, empty
    km and wait together, and against the LP baseline.
- Revisit if: targeting precision does not rise above chance on 2 seeds, or
  empty km rises without a service-rate or wait gain.

## D68 — relocation outcome counters (measurement, no reward change)
- Date: 2026-10-05, Status: measured, debugging output (1 seed, 2 val episodes)
- Added counters: relocations executed, relocated cars later dispatched, and
  dispatches in the zone the car was sent to (env._reloc_out, _reloc_to).
  Script: scripts/diag_reloc_outcomes.py.
- Result, argmax, val episodes 8000-8001:

  | arm | relocations | later dispatched | dispatched in target | destination top-1 | HHI |
  |---|---|---|---|---|---|
  | no bonus (study2 s1) | 277 | 52% | 25% of dispatched | 38.5% | 0.30 |
  | targeting bonus 0.2 (D67 s0) | 284 | 67% | 30% of dispatched | 22.4% | 0.11 |

- Reading: the bonus raised dispatch of relocated cars and spread destinations
  (no collapse). But aim is right and the car still serves another zone most of
  the time (70% of dispatches are outside the target). Whether that is a design
  choice or an accounting artefact is under investigation.
- Not a result: one seed, two episodes.

## D69 — no mid-flight dispatch (negative result, not adopted)
- Date: 2026-10-05, Status: measured, rejected (1 seed, 40 iterations)
- Test: targeting bonus 0.2 with allow_relocating_dispatch=False (cars still
  relocating cannot be dispatched). Flag now reachable as --no-relocating-dispatch.
- Result (val, final iteration): SR 32.8%, empty 553 km, wait 2.01 min.
  Compare D67 seed 0: SR 64.6%, empty 482 km, wait 2.99 min.
- Reading: mid-flight pulls carry a large part of service. The low in-target
  share (30%) is therefore not waste to remove; cars sent to one zone often serve
  nearby demand on the way. The decision-time aim check does not measure service.
- Default stays allow_relocating_dispatch=True.

## D70 — every held-out evaluation so far used ONE window; fixed
- Date: 2026-10-05, Status: fixed, verified
- Defect: `MultiWindowNYCOrderGenerator.reseed` rewinds the val/test cursor to 0
  on every seeded reset. Every evaluation loop here seeds each episode
  (`train.py` validation, `eval_tradeoff.py`, `shift_eval.py`, `evaluate.py`, the
  `diag_*` scripts), so all of them only ever saw window 0 (Mon 08:00-09:00),
  re-subsampled. MEASURED: seeds 9000, 9001, 9002, 12000, 12001 all report
  window 0 on the val split.
- Consequence: every validation number from D47 to D69, the HANDOFF tables, and
  the LP tuning are single-window numbers. That includes "LP dominates RL". S3
  (val split) and S6 (test split) were also one window each.
- Fix: `UniformSampledNYCOrderGenerator.reseed` sets the window to
  `seed % n_windows` for val/test, so consecutive seeds walk the whole split and
  a seed always maps to the same window. Train mode is unchanged.
  `tests/test_val_window_coverage.py`. Validation now defaults to 12 episodes.
- Re-measured on all 12 val windows, paired, argmax, service rate = completed /
  total (`scripts/diag_objective_gap.py`, seeds 12000-12011):

  | arm | SR % | empty km | wait min | objective R |
  |---|---|---|---|---|
  | greedy | 41.6 ± 7.9 | 64.9 | 2.47 | -93.0 |
  | LP, old defaults (5.0 / 0.5) | 53.9 ± 7.6 | 141.2 | 2.58 | -48.5 |
  | LP, re-tuned (3.0 / 4.0) | 70.1 ± 3.3 | 339.9 | 2.68 | +4.4 |
  | RL untrained (1 PPO update) | 66.9 ± 3.7 | 352.5 | 2.78 | -8.7 |
  | RL flat bonus (study2 s1) | 67.2 ± 4.5 | 472.3 | 2.67 | -13.0 |
  | RL targeting bonus (D67 s2) | 68.5 ± 4.2 | 441.9 | 2.92 | -6.7 |

  R = gamma*served - beta*empty_km - alpha*wait - cancel, summed over zones with
  no shaping. One training seed per RL arm: debugging output, not a result.
- LP re-tune: 16 configs (lookahead 3/5/8/12 x penalty 0.5/1/2/4) on 8 TRAIN
  windows, ranked by R (`scripts/tune_lp_objective.py`,
  `results/lp_tuning/objective_sweep.json`). Defaults changed to 3.0 / 4.0.
  Val was not used for the choice.
- Two readings that replace earlier ones:
  1. RL with relocation masked off is identical to greedy (41.6%, 64.9 km), so
     everything RL adds comes from relocation, and almost any relocation helps:
     an untrained policy gains 25 points of service over greedy.
  2. 40 iterations of training added +120 ± 13 km and +0.3 ± 1.0 points of
     service over the untrained policy (paired). Training was buying nothing.

## D71 — relocator share is split out of gamma, not paid on top (env 3.2)
- Date: 2026-10-05, Status: adopted as NYC default; supported by 1 seed
- Problem: D49 paid the relocator 0.65 ON TOP of the dispatcher's gamma. The
  policy is parameter-shared, so what it optimises is the SUM of zone rewards,
  and that sum was R + 0.65 per dispatched relocated car: about +40 per episode
  for moving cars, whatever it achieved. MEASURED (D70 protocol): on the
  training reward the flat-bonus policy beats the untrained one by +2.2 ± 4.6,
  on R it is -4.3 ± 4.0 worse.
- Change: `relocator_share_is_split`. The relocator is still paid at dispatch;
  the dispatcher receives gamma minus that payment at completion. Zone rewards
  now sum to R exactly, apart from rides still in flight at the horizon
  (`tests/test_reward_sums_to_objective.py`). NYCConfig True, EnvConfig False.
- Result, seed 1, 40 iterations, paired vs untrained on 12 val windows:
  SR +2.6 ± 1.0, empty +116 ± 12 km, R +5.2 ± 4.2. First configuration where
  training improves service; the gain in R is about one standard error.
- 40 more iterations (warm restart) changed nothing: SR 67.9%, 468 km. The run
  is at a plateau, not cut short.
- Against the re-tuned LP (paired): SR -0.6 ± 1.1, empty +128 ± 11 km,
  R -7.9 ± 4.1. A tie on service, a clear loss on empty distance.
- D66 (shortage gate) and D67 (targeting bonus) stay off: both pay outside R.

## D72 — beta 0.2 probe: empty km is under-priced, and argmax differs from sampling
- Date: 2026-10-05, Status: measured, 1 seed, no default changed
- Why: at beta 0.06 the whole 128 km gap to LP is worth 7.7 reward per episode,
  so the reward barely asks for shorter or fewer moves.
- Run: split credit, beta 0.2, seed 1, 40 iterations.
  - Training rollouts (sampled, train split): SR 70.1%, 433 km at iteration 40.
  - Validation (argmax, 12 windows): SR 58.2%, 234 km, wait 2.92.
- Readings:
  1. beta does move empty km, so it is the lever for that KPI. At this setting
     the argmax policy gives up 11 points of service for it, and the LP frontier
     is better at the same service (58.9% at 151 km on train windows).
  2. The policy that is trained (sampled) and the policy that is evaluated
     (argmax, D60) behave very differently here. Which one is "the frozen
     policy" has to be fixed before the shift study.
- Open, user's call: the value of beta, and argmax vs seeded sampling.

## D73 — why RL does not follow a moved hotspot; centred targeting bonus (1 seed)
- Date: 2026-10-05, Status: measured, bonus stays OFF by default
- Two-zone test (`scripts/competence_two_zone.py`, 4 episodes, policy sampled):

  | policy | aim (best legal) | phase-1 SR | phase-2 SR | fleet at B |
  |---|---|---|---|---|
  | greedy | - | 3.7% | 6.9% | 20% -> 7% |
  | LP (tuned) | - | 56.0% | 34.0% | 8% -> 30% |
  | split credit (D71) | 33.7% | 47.0% | 13.7% | 5% -> 7% |
  | flat bonus + targeting bonus (D67 s2) | 61.3% | 54.9% | 32.0% | 5% -> 25% |
  | split + centred targeting bonus | 73.9% | 43.1% | 24.3% | 5% -> 27% |

- Cause: aim. The policies that pick the neediest zone move the fleet to the new
  hub; the one that does not, does not. The environment does not block it (LP
  uses the same mask and action interface). Training demand never shows this:
  max single-zone share is 9-12% per window against 90-100% in the test, with no
  within-episode drift beyond sampling noise, and 0-1 riders pending per zone.
- Change: `targeting_bonus_centred` (default True) subtracts the mean legal
  target's demand, so a random destination earns zero and the bonus does not
  pay for moving as such.
- Cost on ordinary demand (12 val windows, paired vs tuned LP): split + centred
  bonus is SR -3.1 ± 0.9, empty +103 ± 12 km, wait +0.37 ± 0.08, R -17.1 ± 4.0.
  Split alone is SR -0.6 ± 1.1, +128 km, wait +0.03, R -7.9 ± 4.1. So the aim
  signal buys the hub-following and loses service and wait on normal demand.
- Test caveat: about 47% of zone B's road nodes lie beyond the 2 km dispatch
  radius, so phase-2 service is capped near 53% for every policy.
- Not a result: one training seed per arm.

## D74 — centred targeting bonus, smaller values (1 seed each)
- Date: 2026-10-05, Status: measured, default stays 0
- Split credit, seed 1, 40 iterations. Paired vs tuned LP on 12 val windows, and
  the two-zone test (4 episodes):

  | bonus | SR vs LP | empty km vs LP | wait vs LP | aim | two-zone phase 2 | fleet at B |
  |---|---|---|---|---|---|---|
  | 0 | -0.6 ± 1.1 | +128 ± 11 | +0.03 ± 0.08 | 34% | 13.7% | 5% -> 7% |
  | 0.05 | -1.2 ± 0.6 | +115 ± 12 | +0.22 ± 0.08 | 44% | 20.9% | 6% -> 22% |
  | 0.1 | -0.4 ± 1.4 | +118 ± 12 | +0.14 ± 0.09 | 52% | 31.3% | 5% -> 23% |
  | 0.2 | -3.1 ± 0.9 | +103 ± 12 | +0.37 ± 0.08 | 74% | 24.3% | 5% -> 27% |

  LP on the two-zone test: phase 2 34.0%, fleet at B 8% -> 30%.
- Reading: 0.1 keeps the hub-following and ties LP on service. No value touches
  the empty-km gap.

## D75 — where the extra empty km comes from; distance term in the destination head
- Date: 2026-10-05, Status: defect fixed in code; no measured benefit yet
- Measured on 12 val episodes, argmax, RL = split credit (D71) vs tuned LP:

  | | RL | LP |
  |---|---|---|
  | relocations per episode | 141 | 112 |
  | mean km per relocation (centroid to centroid) | 3.17 | 2.13 |
  | moves of 4 km or more / under 2 km | 26% / 22% | 6.7% / 54% |
  | idle car, nothing to dispatch: relocate | 100% | 26% |
  | relocated again before any dispatch | 33% | 30% |
  | driver-minutes relocating | 722 | 479 |

  About 70% of the planned-distance gap is longer hops, about 30% is never
  holding a car. P(noop) on the stay-or-move decision is 0.4% (4 episodes).
- Defect: the head's "travel cost" was a linear layer on (target position - own
  position). The own position is the same for every target, so it cancelled, and
  a linear layer cannot form a distance. The policy could not prefer a nearer
  zone. Fixed: `dist_cost` on the real centroid distance (`zone_xy` buffer set
  from the env), zero-initialised, old checkpoints load unchanged.
- Result (seed 1, 40 iterations, bonus 0.1): learned distance weight 0.006,
  validation 67.1% / 470 km / 2.85. No change. At beta 0.06 nothing pushes the
  weight away from zero.

## D76 — beta 0.12 with the distance term (1 seed)
- Date: 2026-10-05, Status: measured, no default changed
- Split credit, bonus 0.1, distance term, beta 0.12, seed 1, 40 iterations:
  validation (12 windows, argmax) 66.9% / 425 km / 2.82 min. Distance weight
  -0.05 (right sign, small).
- Against beta 0.06 with the same setup (67.1% / 470 km / 2.85): about 45 km
  less at the same service. Tuned LP is 70.1% / 340 km / 2.68.
- With D72 (beta 0.2: 58.2% / 234 km): pricing distance does reduce it, and the
  policy moves slowly toward using distance, but in 40 iterations it does not
  reach LP's service and LP's distance together.

## D77 — nearest-k relocation option (measured, not adopted)
- Date: 2026-10-05, Status: implemented behind `relocation_nearest_k` (default 0)
- Applied at evaluation time only to an any-zone policy (12 val windows, paired
  vs tuned LP): nearest 2 gives 68.1% / 341 km / 2.48 min, nearest 4 gives
  71.1% / 410 km / 2.61.
- Trained under it (seed 1, 40 iterations, validation seeds 9000-9011, not
  paired with LP): nearest 2 gives 64.7% / 231 km / 2.68, nearest 4 gives
  72.5% / 441 km / 2.47. Tuned LP: 70.1% / 340 km / 2.68.
- User's call: a restriction of the action space works around the failure
  without explaining it. Kept as an option, not used.

## D78 — why the model does not learn to hold or go nearer; update fixed (A), distance input rescaled (B)
- Date: 2026-10-05, Status: implemented and tested; no KPI gain yet (1 seed)
- Two analyses (no repo changes):
  - Env to model and model to action are faithful: every observation field
    matched simulator truth to float precision over 1920 decisions; 278 of 278
    relocations and 196 of 199 dispatches executed as chosen (3 lost to a
    same-step timeout); buffer contents matched what was acted on.
  - The network can represent the behaviour: an untrained copy, trained by
    imitation for 1200 steps on "hold, else nearest zone with riders", reached
    100% on stay-versus-move and 92% on the target, on held-out states.
  - The learning signal is the failure: per-decision advantage sd about 1.15
    against a hold-versus-move difference of about 0.1; split-half gradient
    cosine 0.16; 85% of rows forced, giving zero gradient but sitting in the
    batch mean and the advantage normalisation; joint clipping scaled the
    actor's gradient by 0.44; the distance weight could move at most 0.19 in a
    run and moved 0.05.
- A (`agents/ippo.py`, `agents/mappo.py`): advantages normalised over rows with
  more than one legal action; policy loss, entropy and clip fraction averaged
  over those rows; value loss still on all rows; MAPPO clips actor and critic
  separately. Logged entropy and clip_frac now refer to real rows; `real_frac`
  added. `tests/test_update_real_rows.py`.
- B (`networks/encoder.py`): distance input multiplied by a saved scale of 10;
  checkpoints without the key load with scale 1, so they behave as before.
- Run with A and B (split credit, bonus 0.1, beta 0.06, seed 1, 40 iterations):
  validation 66.8% / 462 km / 2.79 min, against 67.1% / 470 km / 2.85 for the
  same settings before A and B. Clip fraction on real rows 0.021, real_frac
  0.138. Effective distance weight -0.17, against 0.006 before.
- Reading: both changes do what they were meant to (larger real steps, a
  distance weight that moves the right way), and neither has moved the KPIs in
  40 iterations at beta 0.06. Also still open from the analyses: relocation
  credit is not tied to the destination (29% of dispatched relocated cars are
  used in the target zone; 36% of relocations are overwritten), and the mask
  ignores cars already inbound.

## D79 — beta 0.12 with A and B, 120 iterations (1 seed)
- Date: 2026-10-05, Status: measured, no default changed
- Split credit, bonus 0.1, distance input scale 10, update over real rows,
  beta 0.12, seed 1, 3 workers. Validation, 12 windows (seeds 9000-9011), argmax:

  | iteration | SR % | empty km | wait min |
  |---|---|---|---|
  | 40 | 68.0 | 461 | 2.77 |
  | 80 | 67.1 | 416 | 2.84 |
  | 120 | 65.8 | 332 | 2.80 |

  Tuned LP (seeds 12000-12011, so not paired): 70.1 / 340 / 2.68.
- Training rollouts (sampled): service steady near 70% from iteration 40 to 120
  while empty km fell 446 -> 370. Effective distance weight -0.80, against
  -0.17 after 40 iterations at beta 0.06.
- Reading: with distance priced and the update fixed, the policy does learn to
  drive less, and it was still improving at iteration 120, so this run is not
  converged. Empty km reached LP's level; service is about 4 points below LP on
  this unpaired comparison, and wait about 0.1 min above.

## D80 — D79 continued to 240 iterations: it keeps trading service for distance
- Date: 2026-10-05, Status: measured (1 seed), no default changed
- D79 resumed (fresh optimiser state) for 120 more iterations, same settings.
  Validation, 12 windows (seeds 9000-9011), argmax:

  | total iterations | SR % | empty km | wait min |
  |---|---|---|---|
  | 40 | 68.0 | 461 | 2.77 |
  | 80 | 67.1 | 416 | 2.84 |
  | 120 | 65.8 | 332 | 2.80 |
  | 160 | 61.2 | 218 | 2.90 |
  | 200 | 60.5 | 234 | 3.02 |
  | 240 | 59.9 | 204 | 2.91 |

  Training rollouts (sampled) at 240: 64.1% / 245 km. Effective distance
  weight -1.61.
- Reading: the policy did not settle near LP. It kept giving up service to save
  distance, as beta 0.12 asks. Its points lie below the LP's own trade-off
  curve from the train-window sweep (LP: 67.6% at 259 km, 58.9% at 151 km;
  RL: 65.8% at 332 km, 59.9% at 204 km; different seeds and split, so
  indicative). So beta sets where on the curve RL sits, but RL's curve is still
  worse than LP's.
- Two-zone video episode (iteration-120 checkpoint): MAPPO 36.9% / 422 km,
  fleet at B 12% -> 25%; tuned LP 44.7% / 488 km, 8% -> 31%. The IPPO video
  uses the pre-fix d64 checkpoint and is not a fair IPPO result.

## D81 — inbound-car observation for the actor (1 seed)
- Date: 2026-10-05, Status: implemented behind `obs_inbound` (default False); no clear gain
- Change: each zone's observation gains, per zone, the cars already heading
  there (in flight plus claimed earlier this turn) as a share of the fleet,
  read from the exact per-driver target record. The destination head takes it
  as a per-target feature. `--obs-inbound`; `tests/test_obs_inbound.py`.
- Run: same settings as D79 (split credit, bonus 0.1, beta 0.12, update over
  real rows, distance scale 10, seed 1). The background run was cut at
  iteration 88 by a session restart and resumed from the iteration-80
  checkpoint with a fresh optimiser state. Validation, 12 windows, argmax:

  | iteration | with inbound | without (D79) |
  |---|---|---|
  | 40 | 67.1% / 433 km / 2.75 | 68.0% / 461 km / 2.77 |
  | 80 | 67.1% / 383 km / 2.76 | 67.1% / 416 km / 2.84 |
  | 120 | 64.4% / 281 km / 2.84 | 65.8% / 332 km / 2.80 |

- Reading: about 30-50 km less at each point, with service equal at 40 and 80
  and 1.4 points lower at 120. That is consistent with moving a little faster
  along the same service-versus-distance curve, not with a better curve: at
  281 km, interpolating D79/D80 gives about 63.7% against 64.4% here. One seed;
  the difference is within noise. Tuned LP: 70.1% / 340 km / 2.68.
- Same episode as the progress video (val window 3, seed 9003, D79 checkpoint):
  greedy 46.2% / 67 km / 2.35; tuned LP 66.2% / 326 km / 2.83;
  MAPPO 62.3% / 247 km / 3.00.

## D82 — relocator payment shared along the relocation chain (env 3.3)
- Date: 2026-10-05, Status: implemented and tested, NOT yet trained on
- Problem (measured in the D78 fidelity audit, 278 relocations): 36% were
  re-relocated by a different zone before any dispatch. The record kept only
  the last mover (D47), so the first zone paid for its leg and earned nothing,
  and a zone could take the whole payment by re-sending a car it had been sent.
- Change: `relocator_chain_share`. The payment for a dispatched relocated car
  is divided equally among the distinct zones that moved it since its last
  dispatch, excluding the dispatching zone. The chain is cleared at dispatch
  and at completion, so it holds consecutive moves of one idle car only. That
  is what D47's chain was not: it was cleared only on completion and so counted
  unrelated moves. Clawback on cancel reverses each share. NYCConfig True,
  EnvConfig False.
- Still sums to the objective: `tests/test_reward_sums_to_objective.py` covers
  split + chain sharing. `tests/test_chain_credit.py`.
- Deliberately not done: paying less when the car is dispatched outside its
  target zone. Mid-flight dispatches carry most of the service (D69), so that
  would penalise moves that work.
