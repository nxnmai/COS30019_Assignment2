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
    nodes_created = 0
    
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
            frontier.push(child["g"], neighbor, insertion_order, child)
            insertion_order += 1
            nodes_created += 1
    return None, nodes_created, None
