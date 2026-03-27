"""
Routing search algorithms for the TBRGS route planner.

Implements:
  - Dijkstra shortest path (used internally by Yen's)
  - Yen's k-shortest loopless paths algorithm
  - find_top_k_routes: the public API called by main.py
"""
from __future__ import annotations

import heapq
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Dijkstra (single shortest path)
# ---------------------------------------------------------------------------

def _dijkstra(
    adjacency: Dict[str, List[Tuple[str, float]]],
    source: str,
    target: str,
    forbidden_nodes: Optional[Set[str]] = None,
    forbidden_edges: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[Optional[List[str]], float]:
    """
    Standard Dijkstra from source to target with optional node/edge exclusions.

    Returns (path, cost) or (None, inf) if unreachable.
    """
    forbidden_nodes = forbidden_nodes or set()
    forbidden_edges = forbidden_edges or set()

    if source in forbidden_nodes or target in forbidden_nodes:
        return None, float("inf")

    # dist[node] = (cost, path)
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}

    heap: List[Tuple[float, str]] = [(0.0, source)]

    while heap:
        cost, node = heapq.heappop(heap)

        if node == target:
            # Reconstruct path
            path: List[str] = []
            cur: Optional[str] = target
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path, cost

        if cost > dist.get(node, float("inf")):
            continue

        for neighbour, edge_cost in adjacency.get(node, []):
            if neighbour in forbidden_nodes:
                continue
            if (node, neighbour) in forbidden_edges:
                continue

            new_cost = cost + edge_cost
            if new_cost < dist.get(neighbour, float("inf")):
                dist[neighbour] = new_cost
                prev[neighbour] = node
                heapq.heappush(heap, (new_cost, neighbour))

    return None, float("inf")


# ---------------------------------------------------------------------------
# Yen's k-shortest loopless paths
# ---------------------------------------------------------------------------

def yens_k_shortest_paths(
    adjacency: Dict[str, List[Tuple[str, float]]],
    source: str,
    target: str,
    k: int = 5,
) -> List[Tuple[List[str], float]]:
    """
    Yen's algorithm for k shortest loopless paths.

    Parameters
    ----------
    adjacency : weighted adjacency dict {node: [(neighbour, cost), ...]}
    source : origin node id
    target : destination node id
    k : number of shortest paths to return

    Returns
    -------
    List of (path, total_cost) sorted by cost ascending, length <= k.
    """
    # Candidate set A — confirmed k-shortest paths
    A: List[Tuple[float, List[str]]] = []
    # Candidate set B — potential next paths (heap)
    B: List[Tuple[float, List[str]]] = []

    # Step 0: find the 1st shortest path
    first_path, first_cost = _dijkstra(adjacency, source, target)
    if first_path is None:
        return []

    A.append((first_cost, first_path))
    seen_paths: Set[Tuple[str, ...]] = {tuple(first_path)}

    for _ in range(k - 1):
        if not A:
            break
        prev_cost, prev_path = A[-1]

        # Iterate over each spur node in the last found path
        for spur_idx in range(len(prev_path) - 1):
            spur_node = prev_path[spur_idx]
            root_path = prev_path[: spur_idx + 1]
            root_cost = 0.0

            # Accumulate root cost
            for i in range(len(root_path) - 1):
                u, v = root_path[i], root_path[i + 1]
                for nb, w in adjacency.get(u, []):
                    if nb == v:
                        root_cost += w
                        break

            # Build forbidden sets
            forbidden_edges: Set[Tuple[str, str]] = set()
            forbidden_nodes: Set[str] = set(root_path[:-1])  # exclude root path nodes except spur

            # For all known shortest paths sharing the same root, forbid the next edge
            for _, path in A:
                if len(path) > spur_idx and path[: spur_idx + 1] == root_path:
                    if spur_idx + 1 < len(path):
                        forbidden_edges.add((path[spur_idx], path[spur_idx + 1]))

            # Find spur path
            spur_path, spur_cost = _dijkstra(
                adjacency,
                spur_node,
                target,
                forbidden_nodes=forbidden_nodes,
                forbidden_edges=forbidden_edges,
            )

            if spur_path is None:
                continue

            total_path = root_path[:-1] + spur_path
            total_cost = root_cost + spur_cost

            path_key = tuple(total_path)
            if path_key not in seen_paths:
                heapq.heappush(B, (total_cost, total_path))
                seen_paths.add(path_key)

        if not B:
            break

        # Pop the best candidate from B and add to A
        next_cost, next_path = heapq.heappop(B)
        A.append((next_cost, next_path))

    return [(path, cost) for cost, path in A]


# ---------------------------------------------------------------------------
# Public API — called by main.py via _try_external_search
# ---------------------------------------------------------------------------

def find_top_k_routes(
    adjacency: Dict[str, List[Tuple[str, float]]],
    origin: str,
    destination: str,
    k: int = 5,
) -> List[Tuple[List[str], float]]:
    """
    Find the top-k shortest (minimum travel time) loopless routes.

    This is the function dynamically imported by main.py's _try_external_search.

    Parameters
    ----------
    adjacency  : {node_id: [(neighbour_id, travel_time_sec), ...]}
    origin     : source SCATS site id
    destination: destination SCATS site id
    k          : number of routes to return (default 5)

    Returns
    -------
    List of (path: List[str], total_time_sec: float), sorted ascending.
    """
    return yens_k_shortest_paths(adjacency, origin, destination, k=k)
