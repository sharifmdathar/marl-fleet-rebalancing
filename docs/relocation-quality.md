# Relocation quality and reward balance

## Why service rate alone is not a result

Service rate can be purchased with deadhead distance. Across every scenario
measured, the MARL policy wins service rate and loses badly on empty
kilometres. Both facts are the same fact.

Relocation precision: when the policy sends a car from zone i to zone j, was j
the best target available at that instant? Chance over 8 neighbours is 12.5%.

| scenario | policy | SR | empty km | reloc/ep | precision | SR per 100km |
|---|---|---|---|---|---|---|
| base | LP | 58.9% | 185 | 90 | **38%** | **31.8** |
| base | RL-argmax | 56.0% | 524 | 225 | 6.6% | 10.7 |
| base | RL-sampled | 68.8% | 483 | 204 | 13.5% | 14.2 |
| local-hub | LP | 5.5% | 44 | 16 | **33%** | **12.5** |
| local-hub | RL-argmax | 16.7% | 648 | 275 | 10.6% | 2.6 |
| local-hub | RL-sampled | 8.4% | 674 | 300 | 13.7% | 1.2 |

RL targets at chance (13.5%) and sends cars to zones with **no unmet demand
87% of the time**. Argmax is *below* chance at 6.6%. LP targets three times
better than chance while moving half as often.

The earlier claim that the RL policy "beats the LP rebalancer by +8.8pp" is
therefore wrong on efficiency grounds: LP delivers 31.8 service-rate points per
100 empty km against RL's 10.7-14.2. Report SR and empty distance together or
the number is meaningless.

## Why: the reward made relocation unprofitable

Per-episode reward decomposition of the trained policy (env 2.7):

| term | value |
|---|---|
| served credit (75 completions × gamma 1.0) | **+75.0** |
| empty distance (506 km × beta 0.15) | **−75.9** |
| relocator credit (50 fires × 0.35) | +17.5 |
| relocation actions | 203 |

The served credit and the distance penalty **cancelled almost exactly**. The
entire signal distinguishing a good move from a wasted one was noise on a net
of zero.

The specific error: a relocation costs `beta × km` = 0.15 × 2.49 = **0.374**
and pays `relocator_share × gamma` = **0.350** when it works. The break-even
share is 0.374, so at 0.35 a relocation was net-negative **even when the move
was perfect**. The zone could only learn that sending cars does not pay.

Against the same weights:

| | served credit | distance cost | net reward | SR | empty km |
|---|---|---|---|---|---|
| LP | +77.2 | −27.6 | **+49.6** | 59.4% | 184 |
| RL | +75.0 | −75.9 | **−0.9** | 68.8% | 483 |

RL scored 50 reward units worse than LP while looking better on service rate.
The policy was doing what the reward asked — buying riders with deadhead — and
the reward was punishing it for succeeding. This also explains the training
plateau: past ~66% service rate, every extra relocation is net-negative, so
there is no gradient to climb further.

## Fix (env 2.8)

- `beta` 0.15 → **0.06**. A 2.49 km move now costs 0.149 instead of 0.374, so
  deadhead stays a real discriminator between good and bad moves without
  cancelling the service term it is meant to modulate.
- `relocator_share` 0.35 → **0.45**, comfortably above the new break-even of
  0.149. A correct move now nets **+0.301**; a wasted one costs −0.149.

Projected per-episode reward with the same behaviour: **−0.9 → +44.6**.

`relocator_share` must be kept above `beta × km_per_relocation / gamma`. That
is the invariant that was violated, and it is the one to assert in any future
reward sweep.

## What this does not fix

The policy still has not learned *where* to send cars. Rebalancing makes a
correct move profitable, which gives the gradient something to climb, but the
observation already contains current per-zone demand (`all_zone_state[:, 2]`)
and an explicit deficit (column 3), so the information is present. Precision
above chance is the acceptance test for the next training run — not validation
service rate, which is what misdirected this investigation.

## Claims from before this analysis that were wrong

Recorded rather than deleted, so the record is not quietly rewritten.

- **"The trained RL policy beats the LP rebalancer, +8.80pp, 12/12 windows."**
  True for service rate only. LP delivers 31.8 service-rate points per 100
  empty km against RL's 14.2, so on efficiency the ordering reverses. The
  +8.80pp figure came from comparing against LP on *default* hyperparameters;
  after re-tuning LP the gap was smaller, and after measuring distance it is
  not the advantage it appeared.
- **"The relocation-claim fix (D34) gave +7pp on argmax."** Measured by
  evaluating a checkpoint trained *without* D34 using the mask *with* D34. The
  corrected reward trains with it from the start and shows no such gain.
- **"Validation service rate 40-52% means the policy is weak."** That range is
  the noise floor of a single-policy mean over 12 hard windows. The policy's own
  per-window spread is 3.1-3.9pp. Reading it as weakness sent this
  investigation down the wrong path for several days.
- **"The do-nothing policy is a safe fallback that scores ≈ −3.2."** It scores
  **0% service rate** and −15.11. The near-flat reward was the actual defect,
  not evidence of safety.