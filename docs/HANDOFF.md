# HANDOFF.md — current state

## Where the study is

Env config **2.3**. Every result produced before 2026-09-25 was measured on a
different environment and has been deleted: all of `results/`, plus the old
run-of-record doc. There are currently **no valid study results**. Anything
quoted from before that date is void.

## Infrastructure (done)

Phases 0-6 are complete and committed: spaces frozen, decode/mask parity fixed
(D29), adjacency + demand history in obs (D30), single arrival pay (D31),
fresh mask per agent step (D32), 23 unit tests passing, attention trunk as an
opt-in alternative to the MLP (D33), intra-turn relocation arrival claim
(D34, uncommitted).

NYC data is built and verified: 62 zones, action space 93, obs 805 dims,
84/12/12 train/val/test windows from real FHVHV April 2026.

## The gate: beat the LP rebalancer first

Degradation curves are not the next step. Nothing gets evaluated for shift
degradation until a frozen MARL policy beats the tuned LP baseline
in-distribution on S0.

Measured on the current env, S0, 5 seeds:

| policy | service rate | empty km | reloc % |
|---|---|---|---|
| Greedy | 51.7% ± 3.2 | 118 | 0.0% |
| LP (default hparams) | 71.7% ± 3.1 | 800 | 7.2% |
| IPPO it30, sampled | 70.0% ± 2.1 | 1734 | ~14% |
| IPPO it30, argmax | 55.4% ± 1.7 | 1883 | 15.1% |

**Gate not met.** Note the LP row is the *default* hyperparameter setting — LP
has never been tuned against env 2.3, and the old (invalid) sweep showed
tuning moved it from 69.5% to 73.0%. The real bar is ≥ 71.7%. Per AGENTS.md
rule 4, re-tune LP on S0 + S3 only before judging MARL.

## Open problem

Same weights, same seeds: 70.0% sampled vs 55.4% argmax. The stochastic
distribution is much better than its own mode. D34 addressed part of it
(104 drivers aimed at one zone per turn before; now demand-capped) for +7pp
on argmax, but the gap is not closed.

Leading hypothesis, untested: the policy relocates ~1150 times per episode with
a 100-car fleet regardless of destination, so the D28 arrival bonus is paying
for the *act* of relocating rather than for service. This cannot be tested by
editing reward weights at eval time — a frozen policy's actions depend only on
observations. It requires retraining with a different bonus.

## Next steps

1. Finish the gate runs: `runs/gate-nyc-{mlp,attention}-s0`, 120 iterations,
   IPPO, S0, val-selected. Tracks argmax val SR every 15 iterations.
2. Re-tune LP on env 2.3, S0 + S3 only, 3 seeds.
3. Judge the gate. If argmax closes toward the sampled number, the deficit was
   undertrained-policy, not structure. If it holds at 120 iterations, attack
   the relocation incentive and retrain.
4. Only after the gate passes: arch lock, 5-seed IPPO vs MAPPO, then S0-S6.

## Hardware note

8 cores, 7.7 GB RAM, GTX 1650. Firefox + VS Code hold ~2.2 GB. Use
`--num-workers 3`; 4 workers gets OOM-killed.
