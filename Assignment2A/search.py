"""Main CLI entry for Assignment2A search."""

import sys

from algorithms.astar import astar
from algorithms.gbfs import gbfs
from algorithms.cus2 import cus2
from algorithms.cus1 import cus1
from graph import Graph
from parser import parse_input


def _format_path(path_list):
    return " ".join(str(node_id) for node_id in path_list)


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
    else:
        raise SystemExit(f"Unsupported method: {method}")

    print(f"{filename} {method}")
    if goal_node is None:
        print(f"None {nodes_created}")
        print("None")
    else:
        print(f"{goal_node['id']} {nodes_created}")
        print(_format_path(path))


if __name__ == "__main__":
    main()
