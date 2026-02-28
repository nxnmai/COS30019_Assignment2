"""CUS1: Uniformed-cost search (UCS)"""
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
def cus1(graph, origin, destinations):
    destination_set = set(destinations)
    root = {"id": origin, "parent": None, "g": 0}
<<<<<<< HEAD
    # track unique graph nodes created (root counts as one)
    nodes_created = 1
    created_ids = {origin}
=======
    nodes_created = 1
>>>>>>> e8d1ef09a5eb4f30a440cb13b8b3dbdf05cba291
    
    if origin in destination_set:
        return root, nodes_created, [origin]
    
    frontier = PriorityQueue()
    insertion_order = 0

    frontier.push(root["g"], origin, insertion_order, root)
    insertion_order += 1
    explored = {}
    while not frontier.is_empty():
        current = frontier.pop()
        
        if current["id"] in destination_set:
            return current, nodes_created, _build_path(current)
        
        if current["id"] in explored and explored[current["id"]] <= current["g"]:
            continue
        explored[current["id"]] = current["g"]
        
        for neighbor, g in graph.get_neighbors(current["id"]):
            g = current["g"] + g
            if _is_ancestor(current, neighbor):
                continue
            
            child = {"id": neighbor, "parent": current, "g": g}
            # increment counter only when we see a new graph node id
            if neighbor not in created_ids:
                created_ids.add(neighbor)
                nodes_created += 1
            frontier.push(child["g"], neighbor, insertion_order, child)
            insertion_order += 1
    return None, nodes_created, None
