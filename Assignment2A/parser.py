"""Input parser for Assignment2A search testcases."""


def _parse_coord(coord_text):
    text = coord_text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"Invalid coordinate format: {coord_text}")
    body = text[1:-1]
    parts = body.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format: {coord_text}")
    x = float(parts[0].strip())
    y = float(parts[1].strip())
    return (x, y)


def _parse_edge(edge_text, cost_text):
    text = edge_text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"Invalid edge format: {edge_text}")
    body = text[1:-1]
    parts = body.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid edge format: {edge_text}")
    u = int(parts[0].strip())
    v = int(parts[1].strip())
    cost = float(cost_text.strip())
    return (u, v, cost)


def parse_input(filename):
    """Parse an input file into nodes, edges, origin, and destinations."""
    nodes = {}
    edges = []
    origin = None
    destinations = []
    section = None

    with open(filename, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line == "Nodes:":
                section = "nodes"
                continue
            if line == "Edges:":
                section = "edges"
                continue
            if line == "Origin:":
                section = "origin"
                continue
            if line == "Destinations:":
                section = "destinations"
                continue

            if section == "nodes":
                node_part, coord_part = line.split(":", 1)
                node_id = int(node_part.strip())
                nodes[node_id] = _parse_coord(coord_part)
            elif section == "edges":
                edge_part, cost_part = line.split(":", 1)
                edges.append(_parse_edge(edge_part, cost_part))
            elif section == "origin":
                origin = int(line)
            elif section == "destinations":
                for value in line.split(";"):
                    token = value.strip()
                    if token:
                        destinations.append(int(token))

    return nodes, edges, origin, destinations
