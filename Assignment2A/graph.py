"""Graph representation used by search algorithms."""


class Graph:
    """Directed graph with deterministic neighbor ordering."""

    def __init__(self, coords, edges):
        self.coords = dict(coords)
        self.adj = {node_id: [] for node_id in self.coords}

        for u, v, cost in edges:
            if u not in self.adj:
                self.adj[u] = []
            if v not in self.adj:
                self.adj[v] = []
            self.adj[u].append((v, cost))

        for node_id in self.adj:
            self.adj[node_id].sort(key=lambda item: item[0])

    def get_neighbors(self, node_id):
        return self.adj.get(node_id, [])

    def get_coord(self, node_id):
        return self.coords.get(node_id)
