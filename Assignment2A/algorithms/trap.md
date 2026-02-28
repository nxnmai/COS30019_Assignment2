ALGORITHM GreedyBestFirstSearch(graph, origin, destinations)
    // 1. Initial Checks
    IF origin IN destinations THEN 
        RETURN (origin, 1, [origin])

    // 2. Initialize
    frontier = PriorityQueue() // Priority = h(n)
    explored = Set()
    nodes_created = 1

    
    h_root = MIN(EuclideanDist(origin, d) FOR d IN destinations)
    root = {"id": origin, "parent": None, "h": h_root}
    frontier.PUSH(root, root.h)

    // 3. Search Loop
    WHILE frontier IS NOT EMPTY DO
        current = frontier.POP()

        IF current.id IN destinations THEN
            RETURN (current, nodes_created, ReconstructPath(current))

        IF current.id IN explored THEN CONTINUE
        ADD current.id TO explored

        FOR EACH neighbor IN graph.get_neighbors(current.id) DO
            // Skip immediate cycles and previously visited nodes
            IF neighbor IS current.parent OR neighbor IN explored THEN CONTINUE

            h_neighbor = MIN(EuclideanDist(neighbor, d) FOR d IN destinations)
            child = {"id": neighbor, "parent": current, "h": h_neighbor}
            
            frontier.PUSH(child, child.h)
            nodes_created += 1

    RETURN (None, nodes_created, None)






ALGORITHM UniformCostSearch(graph, origin, destinations)
    // Initialize root and tracking
    root = {"id": origin, "parent": None, "g": 0}
    nodes_created = 1
    
    IF origin IN destinations THEN RETURN root, 0, [origin]

    frontier = PriorityQueue() // Priority = cumulative cost 'g'
    explored = Dictionary()    // Stores {node_id: min_cost}
    
    frontier.PUSH(root.g, origin, root)

    WHILE frontier IS NOT EMPTY DO
        current = frontier.POP()
        
        // Goal reached
        IF current.id IN destinations THEN
            RETURN current, nodes_created, ReconstructPath(current)
        
        // Skip if we already found a cheaper path to this node
        IF current.id IN explored AND explored[current.id] <= current.g THEN
            CONTINUE
        
        explored[current.id] = current.g
        
        // Expand neighbors
        FOR EACH (neighbor, edge_cost) IN graph.get_neighbors(current.id) DO
            // Cycle check: avoid immediate parent or ancestors
            IF IsAncestor(current, neighbor) THEN CONTINUE
            
            new_g = current.g + edge_cost
            child = {"id": neighbor, "parent": current, "g": new_g}
            
            frontier.PUSH(child.g, neighbor, child)
            nodes_created += 1

    RETURN None, nodes_created, None







    ALGORITHM BreadthFirstSearch(graph, origin, destinations)
    // 1. Initialize root node and tracking
    root = {"id": origin, "parent": None, "depth": 0}
    nodes_created = 1
    created_ids = {origin}

    // Trivial goal check
    IF origin IN destinations THEN
        RETURN root, nodes_created, [origin]

    // 2. Initialize frontier as a Queue (FIFO)
    frontier = Queue([root])

    WHILE frontier IS NOT EMPTY DO
        current = frontier.POP_LEFT()

        // Goal test upon expansion
        IF current.id IN destinations THEN
            RETURN current, nodes_created, ReconstructPath(current)

        // Sort neighbors by ID for deterministic expansion
        neighbors = SORT(graph.get_neighbors(current.id))

        FOR EACH (neighbor, cost) IN neighbors DO
            // Cycle prevention: avoid moving back to an ancestor
            IF IsAncestor(current, neighbor) THEN CONTINUE

            // Create child node
            child = {"id": neighbor, "parent": current, "depth": current.depth + 1}
            frontier.PUSH_BACK(child)

            // Track unique nodes created
            IF neighbor NOT IN created_ids THEN
                ADD neighbor TO created_ids
                nodes_created += 1

    RETURN None, nodes_created, None





    ALGORITHM DepthFirstSearch(graph, origin, destinations)
    // 1. Initialize root and tracking
    root = {"id": origin, "parent": None, "depth": 0}
    nodes_created = 1
    created_ids = {origin}

    IF origin IN destinations THEN 
        RETURN root, nodes_created, [origin]

    // 2. Initialize Stack (LIFO)
    stack = [root]

    WHILE stack IS NOT EMPTY DO
        current = stack.POP() // Takes the last item added

        // Goal test upon expansion
        IF current.id IN destinations THEN
            RETURN current, nodes_created, ReconstructPath(current)

        // Get neighbors and sort them by ID
        neighbors = SORT(graph.get_neighbors(current.id))

        // Push in reverse order so smaller IDs are at the top of the stack
        FOR EACH (neighbor, cost) IN REVERSED(neighbors) DO
            // Cycle prevention: avoid moving back to an ancestor
            IF IsAncestor(current, neighbor) THEN CONTINUE

            child = {"id": neighbor, "parent": current, "depth": current.depth + 1}
            stack.PUSH(child)

            // Track unique nodes discovered
            IF neighbor NOT IN created_ids THEN
                ADD neighbor TO created_ids
                nodes_created += 1

    RETURN None, nodes_created, None





ALGORITHM AStarSearch(graph, origin, destinations)
    // 1. Initialize root and tracking
    root = {"id": origin, "parent": None, "g": 0.0, "depth": 0}
    nodes_created = 1
    created_ids = {origin}

    IF origin IN destinations THEN 
        RETURN root, nodes_created, [origin]

    // 2. Initialize Priority Queue (Priority = f = g + h)
    frontier = PriorityQueue()
    h_root = MinDistanceToGoal(origin, destinations)
    frontier.PUSH(root.g + h_root, origin, root)

    WHILE frontier IS NOT EMPTY DO
        current = frontier.POP()

        // Goal test upon expansion
        IF current.id IN destinations THEN
            RETURN current, nodes_created, ReconstructPath(current)

        // Expand neighbors
        FOR EACH (neighbor, cost) IN graph.get_neighbors(current.id) DO
            // Cycle prevention
            IF IsAncestor(current, neighbor) THEN CONTINUE

            // Calculate costs
            new_g = current.g + cost
            h_neighbor = MinDistanceToGoal(neighbor, destinations)
            f_neighbor = new_g + h_neighbor

            child = {
                "id": neighbor, 
                "parent": current, 
                "g": new_g, 
                "depth": current.depth + 1
            }

            frontier.PUSH(f_neighbor, neighbor, child)

            // Track unique nodes discovered
            IF neighbor NOT IN created_ids THEN
                ADD neighbor TO created_ids
                nodes_created += 1

    RETURN None, nodes_created, None




ALGORITHM AStarStepSearch(graph, origin, destinations)
    // 1. Initial Checks
    IF origin IN destinations THEN RETURN (origin, 1, [origin])

    // 2. Setup
    lmax = ComputeLmax(graph)
    root = {"id": origin, "parent": None, "depth": 0}
    
    frontier = PriorityQueue() // Priority = depth + step_heuristic
    h_root = StepHeuristic(origin, destinations, lmax)
    frontier.PUSH(root, root.depth + h_root)

    created_ids = {origin}
    nodes_created = 1

    // 3. Search Loop
    WHILE frontier IS NOT EMPTY DO
        current = frontier.POP()

        IF current.id IN destinations THEN
            RETURN (current, nodes_created, ReconstructPath(current))

        FOR EACH neighbor IN graph.get_neighbors(current.id) DO
            // Cycle prevention
            IF IsAncestor(current, neighbor) THEN CONTINUE

            // Costs based on steps (depth)
            child = {"id": neighbor, "parent": current, "depth": current.depth + 1}
            h_neighbor = StepHeuristic(neighbor, destinations, lmax)
            f_score = child.depth + h_neighbor

            frontier.PUSH(child, f_score)

            // Count unique nodes created
            IF neighbor NOT IN created_ids THEN
                ADD neighbor TO created_ids
                nodes_created += 1

    RETURN (None, nodes_created, None)