# Spaces freeze — Phase 0 (config v1.9, no code run)

Locked from code inspection only. Any change below requires a
`docs/decisions.md` entry + config version bump (AGENTS.md).

## Action space — `Discrete(1 + max_pending + num_zones)`

| Preset | Formula | Total | Dispatch idx | Relocate idx |
|---|---|---|---|---|
| NYC (62 zones, max_pending 30) | 1 + 30 + 62 | 93 | 0 = noop, 1..30 = k-th zone candidate | 31..92 → zone `a-31` |
| Bandra (25 zones, max_pending 80) | 1 + 80 + 25 | 106 | 0 = noop, 1..80 = k-th zone candidate | 81..105 → zone `a-81` |

Source: `env/pettingzoo_env.py:134` (`_action_dim`),
`:600-615` (`_decode_action`).

## Observation keys — `env/pettingzoo_env.py:336-350`

| Key | Shape (z=zones, p=max_pending) | Range |
|---|---|---|
| zone_location | (2,) | 0..1 |
| zone_state | (4,) | -1..1 |
| all_zone_state | (z, 4) | -1..1 |
| pending_orders | (p, 6) | 0..1 |
| pending_mask | (p,) | 0..1 |
| pending_passengers | (p,) | 0..1 |
| time_frac | (1,) | 0..1 |

Flattened (`networks/encoder.py:obs_to_vector`): NYC ≈ 495 dims
(2+4+248+180+30+30+1), Bandra ≈ 747 dims.

## Centralized state (MAPPO critic) — `pettingzoo_env.py:962-977`

`zone_fleet_state(z,4)` + `pending_features(p,6)` + `pending_mask(p,)` + clock.
NYC 459 (+2 zone_location in train = 461). Bandra 661.

## Flagged for Phase 1 (not decided here)

- Dispatch index → order_id mapping is positional per turn (list rebuilt in
  `_rebuild_caches`); stationarity unverified.
- Neighbor-only relocation enforced in mask (`:586-595`) but not in decode
  (`:612-615`); global-pending fallback (`:609-610`) breaks zone ownership.
- Majority of zone-turns have 0 idle drivers → only action 0 legal (sparsity audit pending).
