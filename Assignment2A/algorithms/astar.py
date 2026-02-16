"""A* search (cost-optimal)."""

from utils.heuristic import min_distance_to_goal
from utils.priority_queue import PriorityQueue


def _is_ancestor(node, target_id):
    current = node
    while current is not None:
        if current["id"] == target_id:
            return True
        current = current["parent"]
    return False


def _build_path(goal_node):
    path = []
    current = goal_node
    while current is not None:
        path.append(current["id"])
        current = current["parent"]
    path.reverse()
    return path


def astar(graph, origin, destinations):
    destination_set = set(destinations)
    root = {"id": origin, "parent": None, "g": 0.0, "depth": 0}
    nodes_created = 0

    if origin in destination_set:
        return root, nodes_created, [origin]

    frontier = PriorityQueue()
    insertion_order = 0
    root_h = min_distance_to_goal(origin, graph, destinations)
    frontier.push(root["g"] + root_h, origin, insertion_order, root)
    insertion_order += 1

    while not frontier.is_empty():
        current = frontier.pop()

        if current["id"] in destination_set:
            return current, nodes_created, _build_path(current)

        for neighbor, cost in graph.get_neighbors(current["id"]):
            if _is_ancestor(current, neighbor):
                continue

            child = {
                "id": neighbor,
                "parent": current,
                "g": current["g"] + cost,
                "depth": current["depth"] + 1,
            }
            h = min_distance_to_goal(neighbor, graph, destinations)
            f = child["g"] + h
            frontier.push(f, neighbor, insertion_order, child)
            insertion_order += 1
            nodes_created += 1

    return None, nodes_created, None
