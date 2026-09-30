# Eval noise — use paired windows, not independent means

## The variance is between windows, not measurement error

12 val windows, per-window service rate (%):

    window   1  2  3  4  5  6  7  8  9 10 11 12
    greedy  50 54 40 53 36 51 38 56 49 41 42 53
    lp      54 62 48 63 38 61 48 53 52 53 47 69

Unpaired means: greedy 46.7%, lp 54.0%, with std 6.7 and 8.2 pp. That
spread is real — some demand windows are simply harder than others, and both
policies are hit by the same windows.

## Pairing removes most of it

Both policies are evaluated on the *same* window per seed, so window
difficulty cancels in the difference:

    unpaired  lp - greedy = +7.1 pp
    paired    std of the gap = 4.9 pp   (vs 8.2 unpaired)
    every one of the 12 windows favours lp

## Rule

Any comparison between two policies on the same scenario set is reported as
the **paired per-window difference** with its std and a sign count, never as
two independent means. This applies to MARL vs LP, IPPO vs MAPPO, and
reward variants.

Unpaired means are only meaningful for a single policy's absolute level.

## Consequences

- Differences below ~5 pp unpaired are not evidence. Paired differences are
  usable at a much smaller margin, which is what 5 seeds x 12 windows gives.
- Checkpoint selection is still a single-policy quantity, so val SR stays a
  plain mean; only the comparisons are paired.
