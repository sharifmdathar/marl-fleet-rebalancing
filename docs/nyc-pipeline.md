# NYC Data Pipeline

How to set up the NYC Manhattan study area for experiments.

Status (2026-09-29): pipeline run locally against `dataset/fhvhv_tripdata_2026-04.parquet`
+ `dataset/taxi_zones/` — Stage 0 (zone_centroids, 263 zones), Stage 1
(manhattan.gpickle, 4869 nodes/10480 edges), Stage 3 (splits: 84 train / 12
val / 12 test windows, 62 in-region zones, manifest saved) all reproduced
the file structure below with no deviations. Stage 2 (single-window
`orders.parquet`) was skipped — `build_splits` calls it internally per
window and nothing else in the codebase reads the single-window file.
`data/nyc/` and `dataset/` stay git-ignored (large/regeneratable); anyone else
on this repo reproduces them by re-running the stages below against their own
copy of the two manual downloads.

**Superseded for the current study (D65).** The 62-zone build described below
is reference material. The study runs on a **16-zone subset** at
`data/nyc16/splits` — the 16 busiest zones by train-window volume, 26 drivers,
`nyc_window_limit=130`, `max_pending` 16, action space 33:

```
uv run python scripts/build_zone_subset.py --num-zones 16
```

Read `data/nyc16/splits/manifest.json` for `zone_ids` and the window lists.
**62-zone artifacts are not comparable to 16-zone results** — do not cite
numbers across the two. `docs/spaces-freeze.md` describes the 62-zone action
space (93) and does not match the current build.

## Prerequisites

Install data extras:
```bash
uv add "ride-gym[data]"
```

## External Data (manual download)

Two files must be downloaded manually:

### 1. NYC TLC Taxi Zone Shapefile
- Source: NYC TLC website or NYC Open Data
  (`https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip` is TLC's S3 mirror)
- Files needed: `taxi_zones.shp`, `taxi_zones.shx`, `taxi_zones.dbf`
- Place in: `./dataset/taxi_zones/`
- Gotcha (2026-09-07): the zip contains a **nested read-only `taxi_zones/` folder**,
  so the shapefile lands at `dataset/taxi_zones/taxi_zones/*.shp` while the
  pipeline expects `dataset/taxi_zones/*.shp`. Unpack with:
  ```bash
  cd dataset/taxi_zones && unzip taxi_zones.zip && chmod -R u+w . \
    && mv taxi_zones/* . && rmdir taxi_zones && cd ../..
  ```

### 2. NYC TLC FHVHV Trip Data
- Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- File: `fhvhv_tripdata_2026-04.parquet` (or any month)
- Place in: `./dataset/fhvhv_tripdata_2026-04.parquet`
- Size: ~1-2 GB (21M rows for April 2026)

## Pipeline Stages

Run from project root (`marl-fleet-rebalancing/`):

### Stage 0 — Build Zone Centroids
```bash
uv run python -m ride_gym.data_tools.nyc.zone_centroids
```
- Reads: `./dataset/taxi_zones/taxi_zones.shp`
- Writes: `./data/nyc/zone_centroids.csv`
- Output: ~60 zones in Region A (midtown/lower Manhattan)

### Stage 1 — Build Manhattan Network
```bash
uv run python -m ride_gym.data_tools.nyc.build_nyc_network
```
- Downloads: OSM drive network for bounding box (-74.02, 40.70, -73.93, 40.80)
- Writes: `./data/nyc/manhattan.gpickle`
- Auto-generates: `manhattan.gpickle.matrices.npz` (distance cache)

### Stage 2 — Preprocess Orders (single window)
```bash
uv run python -m ride_gym.data_tools.nyc.preprocess_orders
```
- Reads: FHVHV parquet + zone centroids
- Writes: `./data/nyc/orders.parquet`
- Optional: `--start "2026-04-01 18:00" --end "2026-04-01 19:00" --sample-rate 0.1`

### Stage 3 — Build Train/Val/Test Splits
```bash
uv run python -m ride_gym.data_tools.nyc.build_splits
```
- Reads: FHVHV parquet + zone centroids
- Writes: `./data/nyc/splits/` (84 train, 12 val, 12 test windows)
- Train days: April 6-12 (Mon-Sun)
- Val days: April 13 (Monday)
- Test days: April 14 (Tuesday)

## Expected File Structure After Pipeline

```
./dataset/
    taxi_zones/
        taxi_zones.shp
        taxi_zones.shx
        taxi_zones.dbf
    fhvhv_tripdata_2026-04.parquet

./data/nyc/
    zone_centroids.csv
    manhattan.gpickle
    manhattan.gpickle.matrices.npz
    orders.parquet
    splits/
        manifest.json
        train/window_0000.parquet ... window_0083.parquet
        val/window_0000.parquet ... window_0011.parquet
        test/window_0000.parquet ... window_0011.parquet
```

## Using NYC Mode in Code

```python
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC

cfg = NYCConfig()
env = RideGymAEC(config=cfg, seed=42)
```

## Notes

- **Zone count is 16 in the current study** (the 16 busiest taxi zones whose
  centroids fall in the Region A bbox; D65). The pipeline still *builds* 62
  in-region zones into `data/nyc/` — that is the source pool the subset is cut
  from, not the environment the study uses.
- **`max_pending` is 16** in the current `NYCConfig`, derived from the 16-zone
  build the way the old 650 was (measured worst-case concurrent 2-min
  arrivals across all 108 windows, plus headroom). The 650 figure below is the
  62-zone measurement and does not apply.
- **Order coordinates are degrees.** `origin_x` / `origin_y` in the window
  parquet files are longitude/latitude. Convert with haversine; dividing by
  1000 as if they were metres silently reports 0.000 km (D57).
- Order **zone ids are not stored** in the window parquet files — only
  `origin_x`, `origin_y`, `dest_x`, `dest_y`, `request_time`,
  `num_passengers`. Zone membership is assigned by nearest centroid at load
  time (`_node_to_zone_map` in `demand/generators.py`). Any per-zone demand
  count must be reproduced the same way; `results/zone_demand_train16.json` was
  produced this way.
- RideGym uses zone centroids as relocation points. KNN adjacency (k=8) defines neighboring zones.
- Held-out splits have 12 windows each. Evaluate **one episode per window**
  (D70): a seeded reset used to rewind the window cursor, so single-seed
  evaluation silently covered window 0 only.
