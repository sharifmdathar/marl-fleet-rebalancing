"""Zone lookup helpers — load and query the 3×4 spatial zone partition."""

import json
from pathlib import Path


# From zones.py: demand/ -> marl_fleet_rebalancing/ -> src/ -> project root.
DEFAULT_PARTITION_PATH = Path(__file__).resolve().parents[3] / "data" / "zone_partition.json"

_NUM_ZONES = 12


class ZonePartition:
    """Load and query a 3×4 spatial zone partition of OSMnx graph nodes."""

    def __init__(self, path: "Path | str | None" = None) -> None:
        """Load the zone partition JSON.

        Args:
            path: Path to zone_partition.json. If None, uses DEFAULT_PARTITION_PATH.
        """
        self._path = Path(path) if path is not None else DEFAULT_PARTITION_PATH
        with open(self._path) as f:
            data = json.load(f)

        # node_id (int) -> zone_id (int)
        self._node_to_zone: dict[int, int] = {
            int(node_id): int(zone_id) for node_id, zone_id in data["node_to_zone"].items()
        }
        # zone_id (int) -> metadata dict
        self._zone_meta: dict[int, dict] = {
            int(zone_id): meta for zone_id, meta in data["zone_meta"].items()
        }
        # zone_id (int) -> list of node_ids
        self._zone_to_nodes: dict[int, list[int]] = {z: [] for z in self._zone_meta}
        for node_id, zone_id in self._node_to_zone.items():
            self._zone_to_nodes.setdefault(zone_id, []).append(node_id)

    def node_to_zone(self, node_id: int) -> int:
        """Return the zone_id for a node.

        Args:
            node_id: The OSMnx node id.

        Returns:
            The zone_id (0–11) containing the node.

        Raises:
            KeyError: If the node is not present in the partition.
        """
        try:
            return self._node_to_zone[int(node_id)]
        except KeyError:
            raise KeyError(
                f"node_id {node_id!r} not found in zone partition "
                f"(loaded from {self._path})"
            ) from None

    def zone_to_nodes(self, zone_id: int) -> "list[int]":
        """Return all node_ids in a zone.

        Args:
            zone_id: The zone id (0–11).

        Returns:
            A list of node_ids assigned to the zone.
        """
        return self._zone_to_nodes.get(int(zone_id), [])

    def num_zones(self) -> int:
        """Return the total number of zones (12)."""
        return _NUM_ZONES

    def zone_meta(self, zone_id: int) -> dict:
        """Return the metadata dict for a zone.

        Args:
            zone_id: The zone id (0–11).

        Returns:
            The metadata dict (row, col, lat/lon bounds, node_count).
        """
        return self._zone_meta[int(zone_id)]

    def all_zones(self) -> "list[int]":
        """Return the sorted list of zone ids (0–11)."""
        return sorted(self._zone_meta)

    def __repr__(self) -> str:
        """Return a concise representation with num zones and total nodes."""
        return (
            f"{type(self).__name__}(num_zones={self.num_zones()}, "
            f"total_nodes={len(self._node_to_zone)})"
        )


_default: "ZonePartition | None" = None


def get_default() -> ZonePartition:
    """Lazily load and cache the default zone partition.

    Returns:
        The cached ZonePartition loaded from DEFAULT_PARTITION_PATH.
    """
    global _default
    if _default is None:
        _default = ZonePartition()
    return _default
