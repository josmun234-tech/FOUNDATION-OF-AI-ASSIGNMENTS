


import heapq

def a_star(graph, start, goal, heuristics):
    # Priority queue: (total_estimated_cost, current_cost, node, path)
    frontier = [(heuristics[start], 0, start, [start])]
    visited = set()

    while frontier:
        f, cost, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        
        if current == goal:
            return path, cost

        visited.add(current)

        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in visited:
                new_cost = cost + weight
                # f(n) = g(n) + h(n)
                f_n = new_cost + heuristics.get(neighbor, 0)
                heapq.heappush(frontier, (f_n, new_cost, neighbor, path + [neighbor]))

    return None, float('inf')

# Example Graph
adj_list = {
    'A': {'B': 1, 'C': 3},
    'B': {'D': 1, 'E': 4},
    'C': {'F': 2},
    'D': {'G': 5},
    'E': {'G': 1},
    'F': {'G': 1}
}

h_values = {'A': 6, 'B': 4, 'C': 4, 'D': 3, 'E': 1, 'F': 1, 'G': 0}

best_path, total_cost = a_star(adj_list, 'A', 'G', h_values)
print(f"Optimal Path: {best_path} with cost {total_cost}")