"""
RoutingGraph: thin wrapper around a weighted adjacency dict.

Adapts the dict-based adjacency built by main.py into an object with the
same interface as Part A's Graph class, so search algorithms can work with
either representation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class RoutingGraph:
    """
    Directed weighted graph backed by an adjacency dict.

    Parameters
    ----------
    adjacency : Dict[str, List[Tuple[str, float]]]
        Mapping node_id → [(neighbour_id, weight_seconds), ...]
    coords : Dict[str, Tuple[float, float]], optional
        Mapping node_id → (latitude, longitude) for heuristic use
    """

    def __init__(
        self,
        adjacency: Dict[str, List[Tuple[str, float]]],
        coords: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self._adj = adjacency
        self._coords: Dict[str, Tuple[float, float]] = coords or {}

    # ------------------------------------------------------------------
    # Part A–compatible interface
    # ------------------------------------------------------------------

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """Return [(neighbour_id, cost), ...] sorted by neighbour id."""
        return sorted(self._adj.get(node_id, []), key=lambda x: x[0])

    def get_coord(self, node_id: str) -> Optional[Tuple[float, float]]:
        """Return (lat, lon) for the node, or None if unknown."""
        return self._coords.get(node_id)

    @property
    def nodes(self) -> List[str]:
        return sorted(self._adj.keys())

    # ------------------------------------------------------------------
    # Haversine heuristic (replaces Euclidean from Part A)
    # ------------------------------------------------------------------

    def haversine_heuristic(self, node_id: str, goal_id: str, speed_kmh: float = 60.0) -> float:
        """
        Admissible heuristic: straight-line travel time in seconds
        at `speed_kmh` between node and goal (using their lat/lon).
        Returns 0 if coordinates are unavailable.
        """
        import math
        src = self._coords.get(node_id)
        dst = self._coords.get(goal_id)
        if src is None or dst is None:
            return 0.0

        lat1, lon1 = math.radians(src[0]), math.radians(src[1])
        lat2, lon2 = math.radians(dst[0]), math.radians(dst[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        dist_km = 2 * 6371.0 * math.asin(math.sqrt(a))
        return (dist_km / speed_kmh) * 3600.0  # seconds
