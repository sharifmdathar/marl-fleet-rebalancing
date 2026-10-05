# Spaces freeze

**This file is the 62-zone freeze (v1.9) and no longer matches the build the
study runs on.** It is kept because D65 reduced the environment and the old
freeze is the only record of what changed. Current spaces are in the block
below; the historical entry follows.

---

## Current: 16 zones (D65, `NYCConfig`)

Action: `Discrete(33)` = 1 + 16 + 16
- 0 noop
- 1..16 dispatch to the k-th legal candidate (positional within the turn's mask)
- 17..32 relocate to zone (action - 17)

Observation: length **9**, per zone
- own zone location (2)
- own zone state (4)
- pending riders, pending mask, inbound cars (3) — the inbound term is behind
  `obs_inbound`, default **False** (D81)

MAPPO state: observation + 2 zone-tag terms in train mode.

`max_pending` 16, 26 drivers, action space 33. Verified by construction, not
copied:

```
uv run python -c "
import sys; sys.path.insert(0,'src')
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
c=NYCConfig(); e=RideGymAEC(c); e.reset(seed=1)
print(e._action_dim, len(e.observe(e.agents[0])))"
# -> 33 9
```

**The observation grew a destination-conditioned head, not a bigger flat
vector.** `ZoneAttentionEncoder` (D54) scores each legal relocation target
against a query built from the acting zone, with a real centroid-distance term
(D75) and a scale-10 distance input (D78). So the *action* space is unchanged
at 33 and the *observation* is unchanged at 9; only the network's internals
changed. Checkpoints from before D54 do not load into the attention arch, and
checkpoints from before D78 load with distance scale 1.

Still open, carried from the old freeze and not yet checked:
- dispatch index is positional per turn; stationarity unchecked
- global (non-neighbour) fallback to any zone breaks ownership, and
  `neighbor_relocation_only` is **False** by default (D45 removed the mask)
- most zone-turns have 0 idle cars, so only noop is legal — this is why the PPO
  update now operates on real rows only (D78: 85% of rows were forced, giving
  zero gradient while still sitting in the batch mean)

---

## Historical: 62 zones (v1.9, pre-D65)

Action: Discrete(93) = 1 + 30 + 62
- 0 noop, 1..30 dispatch k-th candidate, 31..92 relocate to a-31

Obs (~495 flat):
zone_location(2), zone_state(4), all_zone_state(62,4),
pending_orders(30,6), pending_mask(30), pending_passengers(30), time_frac(1)

MAPPO state: 459 (+2 zone tag in train = 461)

Open:
- dispatch index is positional per turn, stationarity unchecked
- neighbor check in mask only, not decode; global fallback breaks ownership
- most zone-turns have 0 idle, only noop legal
