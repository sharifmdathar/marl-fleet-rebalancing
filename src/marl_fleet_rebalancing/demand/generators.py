"""Parameterized demand generators for spatial distribution shifts (D15)."""

from __future__ import annotations

from typing import Any, List
import numpy as np
from ride_gym import Order
from ride_gym.osmnx_network import OSMnxNetwork


class ParameterizedZoneOrderGenerator:
    """Wraps a base OrderGenerator to force a chosen percentage of orders to originate
    inside a specific target zone, while maintaining normal request timestamps and
    network-wide destinations.

    Implements the RideGym OrderGenerator interface (generate() and reseed(seed)).
    """

    def __init__(
        self,
        base_generator: Any,
        network: OSMnxNetwork,
        relocation_centroids: list[tuple[float, float]],
        target_zone: int,
        target_zone_pct: float,
        seed: int | None = None,
    ) -> None:
        self.base_generator = base_generator
        self.network = network
        self.relocation_centroids = np.asarray(relocation_centroids, dtype=float)
        self.target_zone = int(target_zone)
        # Normalize pct to 0..100 (accepts both 0.08 and 8.0 for 8%)
        pct = float(target_zone_pct)
        if 0.0 < pct <= 1.0:
            pct = pct * 100.0
        self.target_zone_pct = float(max(0.0, min(100.0, pct)))
        self._rng = np.random.default_rng(seed)
        self._last_orders: List[Order] = []

        # Precompute graph nodes belonging to target_zone (by nearest centroid)
        node_coords = np.asarray(network.node_coords, dtype=float)
        dists_sq = np.sum(
            (node_coords[:, None, :] - self.relocation_centroids[None, :, :]) ** 2,
            axis=2,
        )
        node_to_zone = np.argmin(dists_sq, axis=1)
        self.target_nodes = np.where(node_to_zone == self.target_zone)[0]
        if len(self.target_nodes) == 0:
            # Fallback to the nearest node to target centroid if none strictly assigned
            target_cent = self.relocation_centroids[self.target_zone]
            nearest_idx = int(np.argmin(np.sum((node_coords - target_cent) ** 2, axis=1)))
            self.target_nodes = np.array([nearest_idx], dtype=int)

    @property
    def orders(self) -> List[Order]:
        if self._last_orders:
            return self._last_orders
        return getattr(self.base_generator, "orders", [])

    def reseed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        if hasattr(self.base_generator, "reseed"):
            self.base_generator.reseed(seed)

    def generate(self) -> List[Order]:
        orders = self.base_generator.generate()
        if not orders or self.target_zone_pct <= 0.0 or len(self.target_nodes) == 0:
            self._last_orders = orders
            return orders

        total = len(orders)
        num_target = int(round(total * (self.target_zone_pct / 100.0)))
        num_target = max(1, min(total, num_target))

        # Select random order indices to originate from the target zone
        chosen_indices = self._rng.choice(total, size=num_target, replace=False)
        for idx in chosen_indices:
            node_idx = int(self._rng.choice(self.target_nodes))
            coord = self.network.node_coord(node_idx)
            orders[idx].origin = (float(coord[0]), float(coord[1]))

        self._last_orders = orders
        return orders
