# Rollout performance notes

Hardware reference: 12-core CPU, 15 GB RAM (~6 GB free), 4 GB laptop GPU (RTX 3050 Ti).
Timings: NYC unless stated, 4 rollout episodes / 1 iteration / IPPO.

## What each phase costs (after D17)

| run | rollout | merge | update | total |
|---|---|---|---|---|
| serial (seed 1) | 31.5 s | — | 5.8 s | 41.7 s |
| 4 workers (seed 2) | 21.9 s | 4.9 s | 5.9 s | 40.1 s |
| 4 workers (seed 1/3) | ~24–27 s | ~4–5 s | ~6–7 s | ~42–46 s |

Pre-D17 serial baseline was 61.6 s (rollout 38.7 + update 9.6) → **1.48x** from
vectorized caches + zero-copy `flatten`. No science change (bit-identical outputs).

## Practical guidance

- **NYC on this laptop: run serial (`--num-workers 1`).** Four workers barely
  beat serial (40 s vs 42 s) because ~0.8 GB shards × workers + merged 3.3 GB
  buffer exceed free RAM — `sys` time ~50 s signals paging. Memory, not cores,
  is the cap. Two workers land in the same place.
- **Bandra: workers are fine** (MB-scale buffers, no pressure). Rule of thumb:
  `workers × shard_GB + merged_GB` must fit comfortably in free RAM.
- **Merge (~5 s at NYC scale)** is unpickling + memcpy of the full buffer.
  Next step if needed: worker results in shared memory (no code change to
  learning semantics).
- **Update (~6 s)** is PCIe-bound minibatch gather + H2D on a 4 GB card that
  cannot hold the batch. Next step if needed: pinned memory + non-blocking
  transfers (identical math).
- **Do NOT reduce zones or horizon for speed** — zone granularity is the
  study's unit of analysis (see `docs/demand-shift.md`); that needs a new
  scenario entry + advisor sign-off, not an infra tweak.

## Projected full training (80 iters × 6 eps, per seed)

- Serial NYC: ~56 s/iter → ~75 min/seed → ~6 h for 5 seeds.
- MAPPO ≈ +15% on the update phase only.
