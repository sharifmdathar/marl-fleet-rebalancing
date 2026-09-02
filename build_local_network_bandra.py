"""Build an OSMnx graph for Bandra West, Mumbai.

Run once (online):
    uv run python build_local_network_bandra.py
"""
import os
import pickle

import networkx as nx
import osmnx as ox

LAT, LON = 19.0540, 72.8310
RADIUS_M = 2000
OUT_PATH = "data/bandra_west.gpickle"

os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

print(f"downloading drive network: center=({LAT}, {LON}) radius={RADIUS_M} m ...")
g = ox.graph_from_point((LAT, LON), dist=RADIUS_M, network_type="drive")

g = ox.add_edge_speeds(g, fallback=20)
g = ox.add_edge_travel_times(g)

if not nx.is_strongly_connected(g):
    largest = max(nx.strongly_connected_components(g), key=len)
    before = g.number_of_nodes()
    g = g.subgraph(largest).copy()
    print(f"  pruned to largest strongly-connected component: {before} -> {g.number_of_nodes()} nodes")

with open(OUT_PATH, "wb") as f:
    pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"saved: {OUT_PATH}\n  nodes={g.number_of_nodes()} edges={g.number_of_edges()}")