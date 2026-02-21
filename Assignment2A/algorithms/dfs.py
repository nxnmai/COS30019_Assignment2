"""DFS: Depth-first search (tree-based)."""

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


def dfs(graph, origin, destinations):
    destination_set = set(destinations)

    root = {"id": origin, "parent": None, "depth": 0}
    nodes_created = 1
    created_ids = {origin}

    # Trivial goal
    if origin in destination_set:
        return root, nodes_created, [origin]

    # Stack holds nodes (LIFO)
    stack = [root]

    while stack:
        current = stack.pop()

        # Goal test on expansion (same place as your A*)
        if current["id"] in destination_set:
            return current, nodes_created, _build_path(current)

        # Get neighbors and enforce ascending expansion order
        neighbors = list(graph.get_neighbors(current["id"]))  # [(nid, cost), ...]
        neighbors.sort(key=lambda x: x[0])  # sort by neighbor id asc

        # Push in reverse so smaller id is popped first
        for neighbor, _cost in reversed(neighbors):
            if _is_ancestor(current, neighbor):
                continue

            child = {
                "id": neighbor,
                "parent": current,
                "depth": current["depth"] + 1,
            }
            stack.append(child)

            if neighbor not in created_ids:
                created_ids.add(neighbor)
                nodes_created += 1

    return None, nodes_created, None
