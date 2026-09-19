"""Parameterized demand generators for spatial distribution shifts (D15)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from ride_gym import Order
from ride_gym.order_generator import MultiWindowNYCOrderGenerator, NYCOrderGenerator
from ride_gym.osmnx_network import OSMnxNetwork


class UniformSampledNYCOrderGenerator(MultiWindowNYCOrderGenerator):
    """Uniformly samples orders across the full time horizon for NYC windows.

    Prevents the chronological truncation artifact where limit=N takes only the
    first N arrivals (minutes 0 to 4), leaving the rest of the episode dead.

    Samples uniformly from the DataFrame BEFORE snapping to the road network,
    so only the sampled subset pays the snapping cost.
    """

    def generate(self) -> List[Order]:
        import pandas as pd

        idx = self._pick_window_index()
        path = self.order_paths[idx]
        self.last_window_index = idx
        self.last_window_path = path

        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        df = df[df["request_time"] < self.horizon]
        df = df.sort_values("request_time").reset_index(drop=True)

        # Subsample uniformly across the full time horizon BEFORE snapping
        if self.limit is not None and len(df) > self.limit:
            chosen = np.sort(self._rng.choice(len(df), size=self.limit, replace=False))
            df = df.iloc[chosen].reset_index(drop=True)

        suffix = ".parquet" if path.endswith(".parquet") else ".csv"
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=Path(path).parent
        ) as tmp:
            if suffix == ".parquet":
                df.to_parquet(tmp.name, index=False)
            else:
                df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            window_gen = NYCOrderGenerator(
                network=self.network,
                order_path=tmp_path,
                horizon=self.horizon,
                limit=None,
                random_party_size=self.random_party_size,
                max_party_size=self.max_party_size,
                perturb_od_radius_km=self.perturb_od_radius_km,
                rng=self._rng,
            )
            orders = window_gen.generate()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Re-index order_ids consecutively
        for new_id, o in enumerate(orders):
            o.order_id = new_id

        return orders


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


def _node_to_zone_map(network: OSMnxNetwork,
                      centroids: np.ndarray) -> np.ndarray:
    """Map every graph node index to its nearest-centroid zone index."""
    node_coords = np.asarray(network.node_coords, dtype=float)
    dists_sq = np.sum(
        (node_coords[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(dists_sq, axis=1)


def _nodes_in_zone(node_to_zone: np.ndarray, zone: int,
                   centroids: np.ndarray,
                   network: OSMnxNetwork) -> np.ndarray:
    """Node indices of a zone, falling back to the nearest node."""
    nodes = np.where(node_to_zone == int(zone))[0]
    if len(nodes) == 0:
        node_coords = np.asarray(network.node_coords, dtype=float)
        target_cent = np.asarray(centroids, dtype=float)[int(zone)]
        nodes = np.array(
            [int(np.argmin(np.sum((node_coords - target_cent) ** 2, axis=1)))],
            dtype=int,
        )
    return nodes


class BurstOrderGenerator:
    """Wraps a base generator with a time-localized demand burst (S5).

    A share of the orders whose ``request_time`` falls inside
    [burst_start, burst_end) is reassigned to originate in ``burst_zone``
    (timestamps preserved): same total volume, concentrated in time.
    Outside the window, demand is untouched base demand.

    Implements the RideGym OrderGenerator interface (generate/reseed).
    """

    def __init__(
        self,
        base_generator: Any,
        network: OSMnxNetwork,
        relocation_centroids: list[tuple[float, float]],
        burst_zone: int,
        burst_start: float,
        burst_end: float,
        burst_share: float,
        seed: int | None = None,
    ) -> None:
        self.base_generator = base_generator
        self.network = network
        self.centroids = np.asarray(relocation_centroids, dtype=float)
        self.burst_zone = int(burst_zone)
        self.burst_start = float(burst_start)
        self.burst_end = float(burst_end)
        share = float(burst_share)
        if 0.0 < share <= 1.0:
            share = share * 100.0
        self.burst_share = float(max(0.0, min(100.0, share)))
        self._rng = np.random.default_rng(seed)
        self._last_orders: List[Order] = []
        node_to_zone = _node_to_zone_map(network, self.centroids)
        self.burst_nodes = _nodes_in_zone(
            node_to_zone, self.burst_zone, self.centroids, network,
        )

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
        if (
            not orders
            or self.burst_share <= 0.0
            or len(self.burst_nodes) == 0
        ):
            self._last_orders = orders
            return orders

        in_window = [
            i for i, o in enumerate(orders)
            if self.burst_start <= float(o.request_time) < self.burst_end
        ]
        num_burst = int(round(len(in_window) * (self.burst_share / 100.0)))
        num_burst = max(0, min(len(in_window), num_burst))
        if num_burst:
            chosen = self._rng.choice(
                np.asarray(in_window), size=num_burst, replace=False,
            )
            for idx in chosen:
                node_idx = int(self._rng.choice(self.burst_nodes))
                coord = self.network.node_coord(node_idx)
                orders[int(idx)].origin = (float(coord[0]), float(coord[1]))

        self._last_orders = orders
        return orders


class PermutedZoneOrderGenerator:
    """Wraps a base generator with spatial OD permutation (S1).

    With probability ``strength`` per order, origin AND destination zones
    are remapped through a fixed ``permutation`` (hotspots move instead of
    intensifying); endpoints become random nodes of the mapped zone.
    Same order count and timestamps.

    Implements the RideGym OrderGenerator interface (generate/reseed).
    """

    def __init__(
        self,
        base_generator: Any,
        network: OSMnxNetwork,
        relocation_centroids: list[tuple[float, float]],
        permutation: Sequence[int],
        strength: float,
        seed: int | None = None,
    ) -> None:
        self.base_generator = base_generator
        self.network = network
        self.centroids = np.asarray(relocation_centroids, dtype=float)
        self.permutation = np.asarray(list(permutation), dtype=int)
        s = float(strength)
        if s > 1.0:
            s = s / 100.0
        self.strength = float(max(0.0, min(1.0, s)))
        self._rng = np.random.default_rng(seed)
        self._last_orders: List[Order] = []
        self._node_coords = np.asarray(network.node_coords, dtype=float)
        self._node_to_zone = _node_to_zone_map(network, self.centroids)
        self._zone_nodes = {
            zi: _nodes_in_zone(self._node_to_zone, zi, self.centroids, network)
            for zi in range(len(self.centroids))
        }

    @property
    def orders(self) -> List[Order]:
        if self._last_orders:
            return self._last_orders
        return getattr(self.base_generator, "orders", [])

    def reseed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        if hasattr(self.base_generator, "reseed"):
            self.base_generator.reseed(seed)

    def _remap_point(self, point: tuple[float, float]) -> tuple[float, float]:
        dists = np.sum(
            (self._node_coords - np.asarray(point, dtype=float)) ** 2, axis=1,
        )
        zone = int(self._node_to_zone[int(np.argmin(dists))])
        mapped = int(self.permutation[zone] % len(self.centroids))
        node_idx = int(self._rng.choice(self._zone_nodes[mapped]))
        coord = self.network.node_coord(node_idx)
        return (float(coord[0]), float(coord[1]))

    def generate(self) -> List[Order]:
        orders = self.base_generator.generate()
        if not orders or self.strength <= 0.0:
            self._last_orders = orders
            return orders

        draw = self._rng.random(len(orders))
        for i, o in enumerate(orders):
            if draw[i] < self.strength:
                o.origin = self._remap_point(tuple(o.origin))
                o.destination = self._remap_point(tuple(o.destination))

        self._last_orders = orders
        return orders
