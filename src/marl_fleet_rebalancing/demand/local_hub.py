"""Local-hub scenario: every order originates at one of exactly two zones.

Tighter than the surge scenarios, which left a fraction of orders spread across
the other zones. Here:

* Orders before the switch originate ONLY at ``stage1_zone`` (default zone 10),
  and their destinations are nodes in zones adjacent to it, so every trip is a
  short local hop.
* Orders after the switch originate ONLY at ``stage2_zone`` (default zone 12),
  with destinations adjacent to that zone instead.

No order ever originates anywhere else. That makes the scenario a direct test
of whether a policy redeploys cars from the first hub to the second, since all
of the demand is at one place at a time and nothing else competes for service.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np


class LocalHubOrderGenerator:
    """All demand at stage1_zone, then all of it at stage2_zone.

    Parameters
    ----------
    stage1_zone, stage2_zone:
        The only two zones orders may originate from.
    switch_min:
        Orders before this time are stage 1, at or after it are stage 2.
    """

    def __init__(
        self,
        base_generator: Any,
        network: Any,
        relocation_centroids: list[tuple[float, float]],
        stage1_zone: int,
        stage2_zone: int,
        switch_min: float,
        region_neighbours: Any = None,
        seed: int | None = None,
    ) -> None:
        if stage1_zone == stage2_zone:
            raise ValueError("stage1 and stage2 zones must differ")
        self.base_generator = base_generator
        self.network = network
        self.relocation_centroids = np.asarray(relocation_centroids, dtype=float)
        self.stage1_zone = int(stage1_zone)
        self.stage2_zone = int(stage2_zone)
        self.switch_min = float(switch_min)
        self._rng = np.random.default_rng(seed)
        self._last_orders: List[Any] = []

        node_coords = np.asarray(network.node_coords, dtype=float)
        d = ((node_coords[:, None, :] - self.relocation_centroids[None, :, :]) ** 2).sum(-1)
        self.node_to_zone = np.argmin(d, axis=1)
        self._origin_nodes = {
            z: self._nodes_in(z, node_coords) for z in (stage1_zone, stage2_zone)
        }
        # Destinations stay local: nodes belonging to a zone adjacent to the
        # active hub, so trips are short hops rather than cross-map.
        # region_neighbours lives on the RideGym env, NOT on the raw network,
        # so it must be passed in; without it destinations spread map-wide.
        if region_neighbours is None:
            raise ValueError(
                "region_neighbours is required so destinations stay adjacent "
                "to the active hub"
            )
        self._dest_nodes = {}
        for z in (stage1_zone, stage2_zone):
            nbrs = [n for n in region_neighbours[z] if n != z]
            pool = [self._nodes_in(n, node_coords) for n in nbrs]
            self._dest_nodes[z] = (
                np.concatenate(pool) if pool else self._nodes_in(z, node_coords)
            )

    def _nodes_in(self, zone, node_coords):
        nodes = np.where(self.node_to_zone == int(zone))[0]
        if len(nodes) == 0:
            cent = self.relocation_centroids[int(zone)]
            nodes = np.array([int(np.argmin(np.sum(
                (node_coords - cent) ** 2, axis=1)))])
        return nodes

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
        """Re-point every order at the active hub with a local destination."""
        out = self.base_generator.generate()
        for o in out:
            zone = self.stage1_zone if float(o.request_time) < self.switch_min else self.stage2_zone
            oi = int(self._rng.choice(self._origin_nodes[zone]))
            di = int(self._rng.choice(self._dest_nodes[zone]))
            o.origin = _coord(self.network, oi)
            o.destination = _coord(self.network, di)
        self._last_orders = out
        return out


def _coord(network, node_idx):
    c = network.node_coord(node_idx)
    return (float(c[0]), float(c[1]))