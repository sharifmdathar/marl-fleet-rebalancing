"""Sequential two-stage surge: demand piles into A, then later into D.

Distinct from the drifting-hotspot scenario, which only moves orders that were
already at the source. Here demand is CONCENTRATED into A for the first stage,
concentrated into D for the second, and transitions between them. So the policy
has to pull cars toward A, then pull them across the map to D, with the second
surge strictly later in time.

Order count and timestamps are preserved, so episode length and the
served/cancelled denominator stay comparable with the base scenario.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np


class SequentialSurgeOrderGenerator:
    """Concentrates demand into A early and into D late.

    Parameters
    ----------
    stage1_zone, stage2_zone:
        Surge hubs. Any distance apart; far apart is the harder case.
    surge_pct:
        Combined share of orders (0-100) redirected into the active hub.
    stage1_end_min, stage2_start_min:
        Demand sits entirely at stage 1's hub before ``stage1_end_min``,
        entirely at stage 2's hub after ``stage2_start_min``, and hands over
        linearly in between.
    """

    def __init__(
        self,
        base_generator: Any,
        network: Any,
        relocation_centroids: list[tuple[float, float]],
        stage1_zone: int,
        stage2_zone: int,
        surge_pct: float,
        stage1_end_min: float,
        stage2_start_min: float,
        seed: int | None = None,
    ) -> None:
        if stage1_zone == stage2_zone:
            raise ValueError("stage1 and stage2 zones must differ")
        if stage2_start_min <= stage1_end_min:
            raise ValueError("stage2 must start after stage1 ends")
        self.base_generator = base_generator
        self.network = network
        self.relocation_centroids = np.asarray(relocation_centroids, dtype=float)
        self.stage1_zone = int(stage1_zone)
        self.stage2_zone = int(stage2_zone)
        self.surge_pct = float(max(0.0, min(100.0, surge_pct)))
        self.stage1_end_min = float(stage1_end_min)
        self.stage2_start_min = float(stage2_start_min)
        self._rng = np.random.default_rng(seed)
        self._last_orders: List[Any] = []

        node_coords = np.asarray(network.node_coords, dtype=float)
        d = ((node_coords[:, None, :] - self.relocation_centroids[None, :, :]) ** 2).sum(-1)
        node_to_zone = np.argmin(d, axis=1)
        self._nodes = {}
        for z in (self.stage1_zone, self.stage2_zone):
            nodes = np.where(node_to_zone == z)[0]
            if len(nodes) == 0:
                cent = self.relocation_centroids[z]
                nodes = np.array([int(np.argmin(np.sum(
                    (node_coords - cent) ** 2, axis=1)))])
            self._nodes[z] = nodes

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
        if not orders or self.surge_pct <= 0.0:
            self._last_orders = orders
            return orders

        span = max(self.stage2_start_min - self.stage1_end_min, 1e-9)
        weights = np.zeros(total, dtype=float)
        for i, o in enumerate(orders):
            t = float(getattr(o, "request_time", 0.0))
            if t < self.stage1_end_min:
                weights[i] = 1.0
            elif t > self.stage2_start_min:
                weights[i] = -1.0
            else:
                frac = (t - self.stage1_end_min) / span
                weights[i] = 1.0 - 2.0 * frac

        n_hit = int(round(total * self.surge_pct / 100.0))
        n_hit = max(1, min(total, n_hit))
        idx = self._rng.choice(total, size=n_hit, replace=False)

        moved = 0
        for i in idx:
            w = weights[i]
            if w >= 0.0 and self._rng.random() <= w:
                zone = self.stage1_zone
            elif w < 0.0 and self._rng.random() <= -w:
                zone = self.stage2_zone
            else:
                continue
            node = int(self._rng.choice(self._nodes[zone]))
            coord = self.network.node_coord(node)
            orders[int(i)].origin = (float(coord[0]), float(coord[1]))
            moved += 1

        self._last_orders = orders
        self._n_moved = moved
        return orders