# NYC Data Pipeline

How to set up the NYC Manhattan study area for experiments.

## Prerequisites

Install data extras:
```bash
uv add "ride-gym[data]"
```

## External Data (manual download)

Two files must be downloaded manually:

### 1. NYC TLC Taxi Zone Shapefile
- Source: NYC TLC website or NYC Open Data
- Files needed: `taxi_zones.shp`, `taxi_zones.shx`, `taxi_zones.dbf`
- Place in: `./dataset/taxi_zones/`

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

- Zone count is ~60 (taxi zones whose centroids fall in Region A bbox)
- `max_pending` in NYCConfig is currently 80 (carried from Bandra). MUST be updated after measuring actual peak pending from NYC data.
- RideGym uses zone centroids as relocation points. KNN adjacency (k=8) defines neighboring zones.
