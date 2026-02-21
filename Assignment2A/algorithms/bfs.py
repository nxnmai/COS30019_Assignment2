"""BFS: Breadth-first search (tree-based)."""

from collections import deque

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


def bfs(graph, origin, destinations):
    destination_set = set(destinations)

    root = {"id": origin, "parent": None, "depth": 0}
    nodes_created = 1
    created_ids = {origin}

    # Trivial goal
    if origin in destination_set:
        return root, nodes_created, [origin]

    frontier = deque([root])

    while frontier:
        current = frontier.popleft()

        # Goal test on expansion (consistent with A*)
        if current["id"] in destination_set:
            return current, nodes_created, _build_path(current)

        neighbors = list(graph.get_neighbors(current["id"]))
        neighbors.sort(key=lambda x: x[0])  # ascending neighbor id

        for neighbor, _cost in neighbors:
            if _is_ancestor(current, neighbor):
                continue

            child = {
                "id": neighbor,
                "parent": current,
                "depth": current["depth"] + 1,
            }
            frontier.append(child)

            if neighbor not in created_ids:
                created_ids.add(neighbor)
                nodes_created += 1

    return None, nodes_created, None
