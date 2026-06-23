"""
NSGA-II utilities: Non-dominated sorting, crowding distance, elite selection
(Ref: Deb et al., 2000)
"""

from typing import List, Tuple, Callable
from model import Solution


def dominates(sol_a: Tuple[float, float], sol_b: Tuple[float, float]) -> bool:
    """
    Check if sol_a dominates sol_b.
    f1 (cost) is minimized, f2 (satisfaction) is maximized.
    """
    f1_a, f2_a = sol_a
    f1_b, f2_b = sol_b
    # a dominates b if a is no worse in all objectives and strictly better in at least one
    # f1: lower is better, f2: higher is better
    return (f1_a <= f1_b and f2_a >= f2_b) and (f1_a < f1_b or f2_a > f2_b)


def non_dominated_sort(solutions: List[Solution],
                       objectives: List[Tuple[float, float]]) -> List[List[int]]:
    """
    Fast non-dominated sort (NSGA-II).

    Args:
        solutions: List of solutions
        objectives: List of (f1_cost, f2_satisfaction) tuples, same order as solutions

    Returns:
        Fronts: List of lists, each containing indices of solutions in that front.
        Fronts[0] = Pareto front (rank 0)
    """
    n = len(solutions)
    if n == 0:
        return []

    # For each solution, count how many dominate it and which it dominates
    domination_count = [0] * n
    dominated_sets = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated_sets[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

    # Front 0 = solutions with no dominators
    fronts = [[]]
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    # Build subsequent fronts
    front_idx = 0
    while fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        front_idx += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def crowding_distance(solutions: List[Solution],
                      objectives: List[Tuple[float, float]],
                      front: List[int]) -> List[float]:
    """
    Calculate crowding distance for solutions in a front (NSGA-II).

    Larger distance = more diversity. Used to break ties in selection.
    """
    n = len(front)
    if n <= 2:
        return [float('inf')] * n

    distance = [0.0] * n
    n_obj = 2  # two objectives

    for obj_idx in range(n_obj):
        # Sort front by this objective
        front_sorted = sorted(range(n), key=lambda i: (
            objectives[front[i]][obj_idx]
        ))

        # Extreme points get infinite distance
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')

        # Normalize and accumulate distance
        obj_max = objectives[front[front_sorted[-1]]][obj_idx]
        obj_min = objectives[front[front_sorted[0]]][obj_idx]
        obj_range = max(obj_max - obj_min, 1e-10)

        for i in range(1, n - 1):
            prev_obj = objectives[front[front_sorted[i - 1]]][obj_idx]
            next_obj = objectives[front[front_sorted[i + 1]]][obj_idx]
            distance[front_sorted[i]] += (next_obj - prev_obj) / obj_range

    return distance


def tournament_selection(objectives: List[Tuple[float, float]],
                         fronts: List[List[int]],
                         crowding: List[float],
                         pop_indices: List[int]) -> int:
    """
    Binary tournament selection based on rank and crowding distance.
    Returns index of selected solution.
    """
    import random

    # Map solution index to its front rank
    rank_map = {}
    for rank_idx, front in enumerate(fronts):
        for sol_idx in front:
            rank_map[sol_idx] = rank_idx

    i = random.choice(pop_indices)
    j = random.choice(pop_indices)

    rank_i = rank_map.get(i, float('inf'))
    rank_j = rank_map.get(j, float('inf'))

    if rank_i < rank_j:
        return i
    elif rank_j < rank_i:
        return j
    else:
        # Same rank: prefer larger crowding distance
        dist_i = crowding[i] if i < len(crowding) else 0
        dist_j = crowding[j] if j < len(crowding) else 0
        return i if dist_i >= dist_j else j


def select_next_population(objectives: List[Tuple[float, float]],
                           pop_size: int) -> Tuple[List[int], List[List[int]]]:
    """
    Select the next population using NSGA-II elite strategy.

    Args:
        objectives: List of (f1, f2) for each solution
        pop_size: Desired population size

    Returns:
        (selected_indices, fronts)
    """
    n = len(objectives)
    # Create dummy solutions list for the function signatures
    solutions = [None] * n

    # Non-dominated sort
    fronts = non_dominated_sort(solutions, objectives)

    # Select solutions front by front
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            # Need to select partially from this front using crowding distance
            distances = crowding_distance(solutions, objectives, front)
            # Sort front by crowding distance (descending)
            sorted_front = sorted(
                range(len(front)),
                key=lambda i: distances[i],
                reverse=True
            )
            remaining = pop_size - len(selected)
            for i in range(remaining):
                selected.append(front[sorted_front[i]])
            break

    return selected, fronts
