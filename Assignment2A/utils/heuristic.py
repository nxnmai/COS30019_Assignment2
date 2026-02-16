"""Heuristic helpers for informed search."""

import math


def euclidean(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def min_distance_to_goal(node_id, graph, destinations):
    node_coord = graph.get_coord(node_id)
    if node_coord is None:
        return math.inf

    best = math.inf
    for goal_id in destinations:
        goal_coord = graph.get_coord(goal_id)
        if goal_coord is None:
            continue
        dist = euclidean(node_coord, goal_coord)
        if dist < best:
            best = dist
    return best


def compute_lmax(graph):
    lmax = 0.0
    for u, neighbors in graph.adj.items():
        u_coord = graph.get_coord(u)
        if u_coord is None:
            continue
        for v, _cost in neighbors:
            v_coord = graph.get_coord(v)
            if v_coord is None:
                continue
            length = euclidean(u_coord, v_coord)
            if length > lmax:
                lmax = length

    if lmax == 0.0:
        return 1
    return lmax


def step_heuristic(node_id, graph, destinations, lmax):
    if lmax <= 0:
        lmax = 1.0

    min_dist = min_distance_to_goal(node_id, graph, destinations)
    if math.isinf(min_dist):
        return math.inf
    return int(math.ceil(min_dist / lmax))
