#!/usr/bin/env python3
"""Create a synthetic network for quick testing when OSM data isn't available.

This generates a simple grid network that can be used with the BandraConfig
when the real OSM network data (data/bandra_west.gpickle) isn't available.
"""

import pickle
import networkx as nx
import osmnx as ox
from pathlib import Path
import os

# Create a simple grid network (5x5 zones = 25 nodes)
# This approximates Bandra West's grid-like structure
print("Creating synthetic 5x5 grid network...")

G = nx.grid_2d_graph(5, 5, create_using=nx.DiGraph)

# Add node attributes
for i, (x, y) in enumerate(G.nodes()):
    G.nodes[(x, y)]["x"] = 72.8310 + x * 0.01  # Longitude
    G.nodes[(x, y)]["y"] = 19.0540 + y * 0.01  # Latitude
    G.nodes[(x, y)]["osmid"] = i

# Add edge attributes (speeds and travel times)
for u, v in G.edges():
    # Approximate distance in km (assuming ~1.1 km between grid points)
    distance_km = 1.1
    speed_kmh = 30.0
    travel_time_min = (distance_km / speed_kmh) * 60
    
    G[u][v]["length"] = distance_km * 1000  # meters
    G[u][v]["speed_kmh"] = speed_kmh
    G[u][v]["travel_time"] = travel_time_min

print(f"Created network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save the network
output_path = "data/synthetic_grid_5x5.gpickle"
os.makedirs("data", exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to: {output_path}")

# Update the BandraConfig to use this synthetic network
config_update = f'''
# Update BandraConfig.network_path in src/marl_fleet_rebalancing/config.py
# from: network_path: str = "data/bandra_west.gpickle"
# to:   network_path: str = "data/synthetic_grid_5x5.gpickle"
'''

print("\nTo use this synthetic network, update BandraConfig.network_path:")
print(config_update)
