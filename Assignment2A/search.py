"""Main CLI entry for Assignment2A search."""

import sys

from algorithms.astar import astar
from algorithms.gbfs import gbfs
from algorithms.cus2 import cus2
from algorithms.cus1 import cus1
from algorithms.dfs import dfs
from algorithms.bfs import bfs
from graph import Graph
from parser import parse_input


def _format_path(path_list):
    return ", ".join(str(node_id) for node_id in path_list)

def compute_path_cost(graph, path):
    """
    Compute total cost of a path based on the graph's adjacency list.
    Assumes graph.adj[u] is iterable of (v, cost) pairs.
    """
    if path is None or len(path) < 2:
        return 0.0

    total = 0.0

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        # Look up the cost of the directed edge u -> v
        found = False
        for (nbr, cost) in graph.adj.get(u, []):
            if nbr == v:
                total += float(cost)
                found = True
                break

        if not found:
            raise ValueError(f"No edge found in graph for path segment {u} -> {v}")

    return total

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python search.py <filename> <method>")

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    nodes, edges, origin, destinations = parse_input(filename)
    graph = Graph(nodes, edges)

    if method == "AS":
        goal_node, nodes_created, path = astar(graph, origin, destinations)
    elif method == "GBFS":
        goal_node, nodes_created, path = gbfs(graph, origin, destinations)
    elif method == "CUS2":
        goal_node, nodes_created, path = cus2(graph, origin, destinations)
    elif method == "CUS1":
        goal_node, nodes_created, path = cus1(graph, origin, destinations)
    elif method == "DFS":
        goal_node, nodes_created, path = dfs(graph, origin, destinations)
    elif method == "BFS":
        goal_node, nodes_created, path = bfs(graph, origin, destinations)
    else:
        raise SystemExit(f"Unsupported method: {method}")

    print(f"{filename} {method}")
    if goal_node is None:
        print(f"None {nodes_created}")
        print("None")
    else:
        print(f"{goal_node['id']} {nodes_created}")
        print(_format_path(path))

    if path is not None:
        print(f"Total cost: {compute_path_cost(graph, path)}")
    else:
        print("Total cost: None")


if __name__ == "__main__":
    main()
