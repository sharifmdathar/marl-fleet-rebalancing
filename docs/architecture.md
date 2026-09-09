# Architecture

## Current (verified against code)

```
main.py                              # smoke test: --policy random|heuristic, MP4/interactive render
build_zone_partition.py              # one-time script: generates data/zone_partition.json
marl_fleet_rebalancing/
  agents/
    random_policy.py                 # random_actions(env, obs)
    ippo.py                          # IPPO learner: masked PPO-clip + GAE,
                                     # decentralized critic, shared params
    mappo.py                         # MAPPO learner: shared actor + centralized
                                     # critic over global state (D4)
  baselines/
    heuristic_baseline.py            # LP rolling-horizon rebalancer (D3, raw RideGym interface)
  config.py                          # EnvConfig / NYCConfig / BandraConfig (D7)
  demand/
    zones.py                         # ZonePartition + get_default()
  env/
    pettingzoo_env.py                # RideGymAEC wrapper: zone-controller agents
                                     # (D10/D13), joint dispatch+relocate (D9),
                                     # NYC split mode (order_source="nyc")
    test_wrapper.py                  # wrapper smoke test (Bandra; --nyc for NYC)
  experiments/
    train.py                         # IPPO/MAPPO training entry (--algo);
                                     # logs config/seed/git/scenario/KPIs
                                     # to runs/ (git-ignored)
  networks/
    encoder.py                       # obs flattening + ZoneActorCritic MLP
                                     # + CentralizedCritic (MAPPO value head)
  visualization.py                   # custom MP4 renderer + interactive viewer
data/
  bandra_west.gpickle                # OSMnx road network (git-ignored)
  zone_partition.json                # 12-zone 3x4 mapping, verified 1464/1464 nodes
  nyc/                               # NYC pipeline output (git-ignored):
                                     # zone_centroids.csv, manhattan.gpickle
                                     # (4868 nodes), splits/ 84 train / 12 val / 12 test
```

Still empty (only `__init__.py`):
- `shifts/` — planned for demand shift generators

No longer empty: `agents/` (ippo.py, mappo.py), `networks/` (encoder.py),
`experiments/` (train.py) — populated 2026-09-09.

### RideGym interface contract (do not break)

- Env: `RidePoolEnv(num_drivers, driver_capacity, dt, horizon, order_timeout,
  order_generator, road_network)`.
- Per-vehicle action is a dict, keys mutually exclusive:
  - `{"orders": [order_id, ...]}` — bid on pending orders
  - `{"relocate": index_or_coord}` — idle vehicles only; THIS is the
    rebalancing action our agents will learn.
- obs/rewards are dicts keyed by vehicle id; `dones["__all__"]` ends the episode.
- Conflicting bids on the same order raise `ConflictError` — any multi-agent
  dispatch policy MUST coordinate bids (central dispatcher or conflict-free
  bidding rule). This is the main integration risk for MARL.
- Order sources are swappable: the wrapper selects via `EnvConfig.order_source`
  — `synthetic` (OSMnxOrderGenerator), `nyc` (MultiWindowNYCOrderGenerator over
  data/nyc/splits, one window per episode; train random / val+test cyclic),
  `dataframe` (DataFrameOrderGenerator; endpoints must be on-network). Demand
  shift is injected by swapping/parameterizing the generator, never by editing
  env internals.
- The wrapper's region model (one zone per agent) always lives in the network's
  real coordinate space: NYC taxi-zone centroids, or a `relocation_grid` over
  the network bounds for synthetic maps (D13).

### Planned (target layout — create as needed, update this file when you do)

This is the target layout per `AGENTS.md`. Most directories exist on disk
but are empty (only `__init__.py`). Create the listed files to populate:

```
marl_fleet_rebalancing/
  agents/
    random_policy.py                 # already on disk
    heuristic_baseline.py            # rolling-horizon imbalance/LP rebalancer
    ippo.py                          # independent PPO
    mappo.py                         # PPO + centralized critic
  demand/
    scenarios.py                     # shift generators, IDs from demand-shift.md
    zone_partition.json              # node_id -> zone_id mapping (12 zones)
  experiments/
    train.py                         # train on base scenario, freeze
    evaluate.py                      # frozen-policy stress test -> degradation curves
    configs/                         # one config per run family
  metrics.py                         # served %, cancelled %, avg wait, empty VMT,
                                     # rebalancing trips, reward
```

### Zone partition (for demand shift evaluation)

- Bandra West OSMnx graph (1464 nodes, 3281 edges) partitioned into 12 zones
  via 3×4 spatial grid
- Zone mapping persisted as `data/zone_partition.json` (node_id → zone_id)
- Order generators reference zones rather than individual graph nodes for
  demand injection, ensuring shifts operate at the zone level

## KPI definitions (single source of truth)

- **Service rate** = completed / total orders.
- **Cancellation rate** = cancelled / total orders (timeout = 2.0 min currently).
- **Avg wait time** = pickup_time - request_time, over completed orders.
- **Empty VMT** = distance driven without passengers (incl. rebalancing legs).
- **Fleet utilization** = 1 − (idle vehicle-time / total vehicle-time), averaged
  over the episode. (Brief metric; add when metrics module is built.)
- **Rebalancing trips** = count of `relocate` actions executed.
- All KPIs computed per episode, then aggregated mean ± std over seeds.