

"""Greedy Best First Search (GBFS) algorithm"""
import math
from utils.priority_queue import PriorityQueue

def _build_path(goal_node):
    path = []
    current = goal_node
    while current is not None:
        path.append(current["id"])
        current = current["parent"]
    path.reverse()
    return path

def _is_ancestor(node, target_id):
    current = node
    while current is not None:
        if current["id"] == target_id:
            return True
        current = current["parent"]
    return False

def gbfs(graph, origin, destinations):
    destination_set = set(destinations)
    def calculate_h(node_id):
        n_c = graph.coords[node_id]
        return min(
            math.sqrt((n_c[0] - graph.coords[g_id][0])**2 + 
                      (n_c[1] - graph.coords[g_id][1])**2)
            for g_id in destination_set
        )

    
    root_h = calculate_h(origin)
    root = {"id": origin, "parent": None, "h": root_h}
    nodes_created = 1

    if origin in destination_set:
        return root, nodes_created, [origin]

    frontier = PriorityQueue()
    insertion_order = 0
    frontier.push(root["h"], origin, insertion_order, root)
    insertion_order += 1

    explored = set()

    while not frontier.is_empty():
        current = frontier.pop()

        if current["id"] in destination_set:
            return current, nodes_created, _build_path(current)

        if current["id"] in explored:
            continue
        explored.add(current["id"])

        for neighbor, _ in graph.get_neighbors(current["id"]):
            if _is_ancestor(current, neighbor):
                continue
        
            if neighbor not in explored:
                h_neighbor = calculate_h(neighbor)
                child = {
                    "id": neighbor,
                    "parent": current,
                    "h": h_neighbor
                }
                # Tie-breaking: h(n) -> neighbor_id -> insertion_order
                frontier.push(child["h"], neighbor, insertion_order, child)
                insertion_order += 1
                nodes_created += 1

    return None, nodes_created, None