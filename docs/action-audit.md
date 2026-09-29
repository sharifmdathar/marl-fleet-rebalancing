# Action audit — Phase 1 (inspection only)

## 1. Positional dispatch: fine as a ranking rule
Candidates are re-sorted every turn (`_rebuild_caches`, lexsort distance/wait),
but the obs lists the same order (`pending_orders` = same sorted features).
So action k always means "k-th ranked candidate" with matching obs row.
Stationary. No fix.

## 2. Mask/decode mismatch: fix decode
Mask enforces neighbor-only + surplus-target + claimed-order checks
(`action_mask`). Decode (`_decode_action`) enforces none of them and falls
back to the global pending list, breaking zone ownership. PPO masks with
-inf so on-policy behavior is masked-correct, but any mask bug executes an
illegal hop instead of noop. Fix: decode returns None for non-neighbor
targets, drop the global fallback. Needs decisions.md entry + bump
(behavior change for off-mask actions only).

## 3. Forced noop: accept, mitigate later
~1.6 drivers/zone means most zone-turns have 0 idle → only action 0 legal.
Inherent to one-action-per-zone-turn. Not fixable without multi-driver
actions (advisor call, D13 note). Mitigation is reward/entropy, Phase 3.

## 4. Train/eval mask staleness: real, loop fix
Train batches all masks pre-turn, so later agents miss same-turn claims;
eval steps sequentially with fresh masks. Unify in Phase 4 (fresh mask
per step in `collect_rollout`).
