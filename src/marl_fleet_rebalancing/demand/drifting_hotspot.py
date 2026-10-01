"""Drifting-hotspot demand: a two-hub relocation scenario.

Demand starts concentrated in a source hub A whose nearby zones B and C also
carry riders, then over a fixed interval it migrates to a far zone D that is
not adjacent to A. Cars positioned for A are therefore useless once the
demand moves, and the only way to serve D is to relocate across the map.

The point is to make relocation decisions legible: a policy that holds cars
where demand used to be, and never revisits the decision, is visibly wrong.
All timestamps and order counts are preserved, so episode length, order
count and the served/cancelled denominator stay comparable with the base
scenario.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np


class DriftingHotspotOrderGenerator:
    """Wraps a base generator and reassigns origins between two hub zones.

    Parameters
    ----------
    source_zone, dest_zone:
        Zone indices for the early and late demand hubs. Should be far apart
        for the scenario to mean anything.
    ramp_start_min, ramp_end_min:
        Interval over which demand transfers from source to destination.
        Outside it, demand sits entirely at one hub or the other.
    source_pct:
        Share of orders originating at ``source_zone`` before the ramp.
    """

    def __init__(
        self,
        base_generator: Any,
        network: Any,
        relocation_centroids: list[tuple[float, float]],
        source_zone: int,
        dest_zone: int,
        ramp_start_min: float,
        ramp_end_min: float,
        source_pct: float,
        seed: int | None = None,
    ) -> None:
        self.base_generator = base_generator
        self.network = network
        self.relocation_centroids = np.asarray(relocation_centroids, dtype=float)
        self.source_zone = int(source_zone)
        self.dest_zone = int(dest_zone)
        self.ramp_start_min = float(ramp_start_min)
        self.ramp_end_min = float(ramp_end_min)
        self.source_pct = float(source_pct)
        self._rng = np.random.default_rng(seed)
        self._last_orders: List[Any] = []

        node_coords = np.asarray(network.node_coords, dtype=float)
        dists_sq = np.sum(
            (node_coords[:, None, :] - self.relocation_centroids[None, :, :]) ** 2,
            axis=2,
        )
        self.node_to_zone = np.argmin(dists_sq, axis=1)
        self.source_nodes = np.where(self.node_to_zone == self.source_zone)[0]
        self.dest_nodes = np.where(self.node_to_zone == self.dest_zone)[0]
        if len(self.source_nodes) == 0 or len(self.dest_nodes) == 0:
            raise ValueError(
                f"zones {self.source_zone}/{self.dest_zone} have no network nodes",
            )

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
        from ride_gym import Order

        orders = self.base_generator.generate()
        if not orders:
            self._last_orders = orders
            return orders

        node_coords = np.asarray(self.network.node_coords, dtype=float)
        order_ids = [
            o.order_id if getattr(o, "order_id", None) is not None else i
            for i, o in enumerate(orders)
        ]

        span = max(self.ramp_end_min - self.ramp_start_min, 1e-9)
        out = []
        n_moved = 0
        for o, oid in zip(orders, order_ids):
            t = float(getattr(o, "request_time", 0.0))
            if t < self.ramp_start_min:
                frac = self.source_pct
            elif t > self.ramp_end_min:
                frac = 0.0
            else:
                frac = self.source_pct * (1.0 - (t - self.ramp_start_min) / span)

            origin = np.asarray(o.origin, dtype=float)
            zone = int(self.node_to_zone[int(np.argmin(np.sum(
                (node_coords - origin) ** 2, axis=1)))])
            if zone == self.source_zone and self._rng.random() > frac:
                node = int(self._rng.choice(self.dest_nodes))
                pt = node_coords[node]
                out.append(Order(
                    order_id=oid,
                    origin=(float(pt[0]), float(pt[1])),
                    destination=o.destination,
                    request_time=o.request_time,
                    num_passengers=o.num_passengers,
                ))
                n_moved += 1
            else:
                out.append(o)

        self._last_orders = out
        self._n_moved = n_moved
        return out