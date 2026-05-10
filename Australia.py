"""
Task 2 — Australia Map Coloring
Constraint Satisfaction Problem (CSP)

BUG FIX: 'from constraint import Problem' fails because 'python-constraint'
is not a built-in module. This solution uses Python's standard library only,
implementing backtracking search with constraint propagation.

Colors  : Red, Green, Blue
Regions : WA, NT, SA, Q, NSW, V, T
Rule    : No two adjacent regions may share the same color.
"""

# ── Define the Map ────────────────────────────────────────────────────────────

states = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
colors = ['Red', 'Green', 'Blue']

# Adjacent regions that cannot share the same color
adjacencies = [
    ('WA', 'NT'), ('WA', 'SA'),
    ('NT', 'SA'), ('NT', 'Q'),
    ('SA', 'Q'),  ('SA', 'NSW'), ('SA', 'V'),
    ('Q',  'NSW'),
    ('NSW', 'V')
    # T (Tasmania) has no land adjacency → any color is valid
]

# Build adjacency lookup for fast constraint checking
neighbours = {s: set() for s in states}
for s1, s2 in adjacencies:
    neighbours[s1].add(s2)
    neighbours[s2].add(s1)


# ── Constraint Check ──────────────────────────────────────────────────────────

def is_consistent(state, color, assignment):
    """Return True if assigning 'color' to 'state' violates no constraint."""
    for neighbour in neighbours[state]:
        if neighbour in assignment and assignment[neighbour] == color:
            return False
    return True


# ── Backtracking Solver ───────────────────────────────────────────────────────

def backtrack(assignment):
    """Recursively assign colors; backtrack on conflict."""
    if len(assignment) == len(states):
        return assignment                          # All states assigned ✅

    # Pick the next unassigned state (in order)
    unassigned = [s for s in states if s not in assignment]
    state = unassigned[0]

    for color in colors:
        if is_consistent(state, color, assignment):
            assignment[state] = color
            result = backtrack(assignment)
            if result is not None:
                return result
            del assignment[state]                  # Backtrack

    return None                                    # No valid color found


# ── Solve ─────────────────────────────────────────────────────────────────────

def solve_australia_map():
    print("=" * 50)
    print("  Australia Map Coloring — CSP Solver")
    print("=" * 50)

    solution = backtrack({})

    if solution:
        print("\n✅  Solution Found!\n")
        print(f"  {'Region':<8}  Color")
        print(f"  {'-'*8}  {'-'*5}")
        for state in states:
            print(f"  {state:<8}  {solution[state]}")

        print("\n📋  Constraint Verification:")
        all_ok = True
        for s1, s2 in adjacencies:
            ok = solution[s1] != solution[s2]
            status = "✅" if ok else "❌"
            print(f"  {status}  {s1} ({solution[s1]:<5}) — {s2} ({solution[s2]})")
            if not ok:
                all_ok = False

        print(f"\n{'✅ All constraints satisfied!' if all_ok else '❌ Conflict detected!'}")
    else:
        print("\n❌  No solution exists.")

    return solution


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    solve_australia_map()
