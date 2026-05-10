#  Depth First Search (DFS)
#  Explores as far as possible along each branch before
#  backtracking. Uses a STACK (LIFO) — or recursion.

GRAPH = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': ['F']
}

#Iterative DFS using an explicit stack


def dfs_iterative(graph, start, goal):
    """
    Iterative DFS using an explicit stack (list as LIFO).

    Parameters
    ----------
    graph : dict  — adjacency list
    start : str   — initial node
    goal  : str   — target node

    Returns
    -------
    path          : list | None — nodes from start → goal
    visited_order : list        — all nodes in pop order
    """

    # ── Initialise ───────────────────────────────────────────
    # Stack stores (current_node, path_so_far) pairs.
    # stack.pop() always takes the TOP (most recently added).
    stack = [(start, [start])]

    visited       = set([start])
    visited_order = []

    print(f"\n{'='*55}")
    print(f"  DFS (iterative)  |  Start: {start}  →  Goal: {goal}")
    print(f"{'='*55}")

    # ── Main DFS loop ─────────────────────────────────────────
    while stack:
        # Pop the TOP of the stack (LIFO — deepest branch first)
        current_node, path = stack.pop()
        visited_order.append(current_node)

        print(f"\n  Pop          : {current_node}")
        print(f"  Path so far  : {' → '.join(path)}")

        # ── Goal check ───────────────────────────────────────
        if current_node == goal:
            print(f"\n  ✓ Goal '{goal}' FOUND!")
            return path, visited_order

        # ── Expand neighbours ─────────────────────────────────
        # Reverse so the FIRST neighbour ends on TOP of stack
        neighbours = graph.get(current_node, [])
        print(f"  Neighbours   : {neighbours}")

        for neighbour in reversed(neighbours):
            if neighbour not in visited:
                visited.add(neighbour)
                new_path = path + [neighbour]
                stack.append((neighbour, new_path))
                print(f"    Push    : {neighbour}  "
                      f"(path: {' → '.join(new_path)})")

        stack_display = [item[0] for item in stack]
        print(f"  Stack now    : {stack_display}")

    print(f"\n  ✗ No path from '{start}' to '{goal}' exists.")
    return None, visited_order


# ════════════════════════════════════════════════════════════
#  VERSION 2 — Recursive DFS (Python's call stack = DFS stack)
# ════════════════════════════════════════════════════════════

def dfs_recursive(graph, current, goal, visited=None, path=None):
    """
    Recursive DFS. Python's own call stack acts as the DFS stack.

    Parameters
    ----------
    graph   : dict  — adjacency list
    current : str   — node currently being explored
    goal    : str   — target node
    visited : set   — nodes already explored (shared across calls)
    path    : list  — path accumulated to current node

    Returns
    -------
    list | None  — complete path if found, else None
    """

    # ── First-call initialisation ─────────────────────────────
    if visited is None:
        visited = set()
    if path is None:
        path = []

    # Mark current node visited and append to working path
    visited.add(current)
    path = path + [current]

    print(f"  Visiting : {current}  |  Path: {' → '.join(path)}")

    # ── Base case: goal reached ───────────────────────────────
    if current == goal:
        print(f"  ✓ Goal '{goal}' FOUND via recursion!")
        return path

    # ── Recursive case: explore each unvisited neighbour ─────
    for neighbour in graph.get(current, []):
        if neighbour not in visited:
            print(f"    → Recurse into: {neighbour}")
            result = dfs_recursive(graph, neighbour, goal,
                                   visited, path)
            if result is not None:
                # Found — propagate the path back up the call stack
                return result

    # ── Backtrack ─────────────────────────────────────────────
    # All neighbours explored with no success — step back up
    print(f"  ✗ Backtrack from: {current}")
    return None


# ── Run both DFS versions ────────────────────────────────────
if __name__ == "__main__":
    START_NODE = 'A'
    GOAL_NODE  = 'G'

    # --- Iterative ---
    dfs_path, dfs_order = dfs_iterative(GRAPH, START_NODE, GOAL_NODE)

    print(f"\n{'='*55}")
    print("  DFS ITERATIVE RESULTS")
    print(f"{'='*55}")
    if dfs_path:
        print(f"  Visited order : {' → '.join(dfs_order)}")
        print(f"  Path found    : {' → '.join(dfs_path)}")
        print(f"  Path length   : {len(dfs_path) - 1} edges")
    print(f"{'='*55}\n")

    # --- Recursive ---
    print(f"\n{'='*55}")
    print(f"  DFS (recursive)  |  Start: {START_NODE}  →  Goal: {GOAL_NODE}")
    print(f"{'='*55}")
    rec_path = dfs_recursive(GRAPH, START_NODE, GOAL_NODE)

    print(f"\n{'='*55}")
    print("  DFS RECURSIVE RESULTS")
    print(f"{'='*55}")
    if rec_path:
        print(f"  Path found  : {' → '.join(rec_path)}")
        print(f"  Path length : {len(rec_path) - 1} edges")
    print(f"{'='*55}\n")