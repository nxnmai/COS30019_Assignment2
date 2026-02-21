"""Step-optimal A* search."""

from utils.heuristic import compute_lmax, step_heuristic
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


def cus2(graph, origin, destinations):
    destination_set = set(destinations)
    root = {"id": origin, "parent": None, "depth": 0}
    # Count unique graph nodes created, including the root.
    nodes_created = 1
    created_ids = {origin}

    if origin in destination_set:
        return root, nodes_created, [origin]

    lmax = compute_lmax(graph)
    frontier = PriorityQueue()
    insertion_order = 0
    root_h = step_heuristic(origin, graph, destinations, lmax)
    frontier.push(root["depth"] + root_h, origin, insertion_order, root)
    insertion_order += 1

    while not frontier.is_empty():
        current = frontier.pop()

        if current["id"] in destination_set:
            return current, nodes_created, _build_path(current)

        for neighbor, _cost in graph.get_neighbors(current["id"]):
            if _is_ancestor(current, neighbor):
                continue

            child = {
                "id": neighbor,
                "parent": current,
                "depth": current["depth"] + 1,
            }
            h_steps = step_heuristic(neighbor, graph, destinations, lmax)
            f = child["depth"] + h_steps
            frontier.push(f, neighbor, insertion_order, child)
            insertion_order += 1
            if neighbor not in created_ids:
                created_ids.add(neighbor)
                nodes_created += 1

    return None, nodes_created, None
