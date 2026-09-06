"""Partition Bandra West OSMnx nodes into 12 zones (3x4 spatial grid). Run once: uv run python build_zone_partition.py"""

import json
import pickle
import pathlib

import numpy as np


DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
GRAPH_PATH = DATA_DIR / "bandra_west.gpickle"
OUTPUT_PATH = DATA_DIR / "zone_partition.json"

N_ROWS = 3
N_COLS = 4


def build_zone_partition() -> None:
    """Load the OSMnx graph, partition nodes into a 3x4 spatial grid, and write zone_partition.json."""
    # Load graph
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    nodes = list(G.nodes(data=True))
    node_ids = [n for n, _ in nodes]
    lats = np.array([data["y"] for _, data in nodes], dtype=float)
    lons = np.array([data["x"] for _, data in nodes], dtype=float)

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    lat_step = (lat_max - lat_min) / N_ROWS
    lon_step = (lon_max - lon_min) / N_COLS

    node_to_zone: dict[str, int] = {}
    zone_node_lists: dict[int, list[int]] = {z: [] for z in range(N_ROWS * N_COLS)}

    for node_id, lat, lon in zip(node_ids, lats, lons):
        row = int((lat - lat_min) / lat_step)
        col = int((lon - lon_min) / lon_step)
        # Clamp to valid range (handles nodes exactly on the max boundary)
        row = min(row, N_ROWS - 1)
        col = min(col, N_COLS - 1)
        zone_id = row * N_COLS + col
        node_to_zone[str(node_id)] = zone_id
        zone_node_lists[zone_id].append(node_id)

    zone_meta: dict[str, dict] = {}
    for zone_id in range(N_ROWS * N_COLS):
        row = zone_id // N_COLS
        col = zone_id % N_COLS
        zone_meta[str(zone_id)] = {
            "row": row,
            "col": col,
            "lat_min": lat_min + row * lat_step,
            "lat_max": lat_min + (row + 1) * lat_step,
            "lon_min": lon_min + col * lon_step,
            "lon_max": lon_min + (col + 1) * lon_step,
            "node_count": len(zone_node_lists[zone_id]),
        }

    output = {
        "node_to_zone": node_to_zone,
        "zone_meta": zone_meta,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    total_nodes = len(node_ids)
    counts = np.array([len(zone_node_lists[z]) for z in range(N_ROWS * N_COLS)], dtype=float)
    print(f"Total nodes: {total_nodes}")
    print(f"Nodes per zone: mean={counts.mean():.1f} ± std={counts.std():.1f}")
    print(f"Zone counts: {[int(c) for c in counts]}")
    print(f"Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_zone_partition()
