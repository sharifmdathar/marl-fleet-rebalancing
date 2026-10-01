"""Two-hotspot demand: concentrate orders into exactly two chosen zones.

Complements the drifting-hotspot scenario, which moves demand between two hubs
over time. Here both hotspots are fixed for the whole episode, so the question
is narrow: can the policy serve demand that is concentrated in two places?

Concentration is set as a combined share, split evenly between the two zones
unless a split ratio is given. Order count and timestamps are preserved, so
episode length and the served/cancelled denominator stay comparable with the
base scenario.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np


class TwoZoneHotspotOrderGenerator:
    """Forces a share of orders to originate from two specified zones.

    Parameters
    ----------
    hotspot_zones:
        The two zone indices to concentrate demand into.
    hotspot_pct:
        Combined share of orders (0-100) redirected into the two zones.
    split:
        Fraction of ``hotspot_pct`` going to ``hotspot_zones[0]``. The
        remainder goes to the second zone. Defaults to an even split.
    """

    def __init__(
        self,
        base_generator: Any,
        network: Any,
        relocation_centroids: list[tuple[float, float]],
        hotspot_zones: tuple[int, int],
        hotspot_pct: float,
        split: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if len(set(hotspot_zones)) != 2:
            raise ValueError(f"need two distinct zones, got {hotspot_zones!r}")
        self.base_generator = base_generator
        self.network = network
        self.relocation_centroids = np.asarray(relocation_centroids, dtype=float)
        self.hotspot_zones = (int(hotspot_zones[0]), int(hotspot_zones[1]))
        self.hotspot_pct = float(max(0.0, min(100.0, hotspot_pct)))
        self.split = float(max(0.0, min(1.0, split)))
        self._rng = np.random.default_rng(seed)
        self._last_orders: List[Any] = []

        node_coords = np.asarray(network.node_coords, dtype=float)
        d = ((node_coords[:, None, :] - self.relocation_centroids[None, :, :]) ** 2).sum(-1)
        node_to_zone = np.argmin(d, axis=1)
        self.hotspot_nodes = {
            z: np.where(node_to_zone == z)[0] for z in self.hotspot_zones
        }
        for z, nodes in self.hotspot_nodes.items():
            if len(nodes) == 0:
                cent = self.relocation_centroids[z]
                idx = int(np.argmin(np.sum((node_coords - cent) ** 2, axis=1)))
                self.hotspot_nodes[z] = np.array([idx], dtype=int)

    @property
    def orders(self) -> List[Any]:
        if self._last_orders:
            return self._last_orders
        return getattr(self.base_generator, "orders", [])

    def reseed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        if hasattr(self.base_generator, "reseed"):
            self.base_generator.reseed(seed)

    def generate(self) -> List[Any]:
        orders = self.base_generator.generate()
        total = len(orders)
        if not orders or self.hotspot_pct <= 0.0:
            self._last_orders = orders
            return orders

        n_hot = int(round(total * self.hotspot_pct / 100.0))
        n_hot = max(2, min(total - 1, n_hot))
        n_first = int(round(n_hot * self.split))
        n_first = max(1, min(n_hot - 1, n_first))
        n_second = n_hot - n_first

        chosen = self._rng.choice(total, size=n_hot, replace=False)
        order = np.argsort(chosen)
        first_idx = order[:n_first]
        second_idx = order[n_first:]

        for idx, zone in (
            [(int(i), self.hotspot_zones[0]) for i in first_idx]
            + [(int(i), self.hotspot_zones[1]) for i in second_idx]
        ):
            node = int(self._rng.choice(self.hotspot_nodes[zone]))
            coord = self.network.node_coord(node)
            orders[idx].origin = (float(coord[0]), float(coord[1]))

        self._last_orders = orders
        return orders