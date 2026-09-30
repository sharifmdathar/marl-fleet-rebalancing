# Spaces freeze (v1.9, NYC only)

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
