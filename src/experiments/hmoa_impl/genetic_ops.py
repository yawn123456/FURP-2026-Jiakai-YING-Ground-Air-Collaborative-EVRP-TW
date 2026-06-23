"""
Specialized Genetic Operations (Section IV-E)

1. One-point crossover
2. Multi-mode mutation (N1, N2, N3)
"""

import random
from typing import Tuple, List
from model import Solution, ProblemInstance
from neighborhood import n1_truck_to_drone, n2_drone_to_truck, n3_swap
from nsga2_utils import tournament_selection
from repair import repair


def crossover(inst: ProblemInstance,
              parent_a: Solution, parent_b: Solution) -> Tuple[Solution, Solution]:
    """
    One-point crossover (Section IV-E1)

    Exchange a customer node between two parent chromosomes at a random position.
    """
    child_a = parent_a.copy()
    child_b = parent_b.copy()

    # Choose random crossover position from Part 1 or Part 2 of the chromosome
    # The paper: "a random and common crossover position will be selected
    # from Part 1 or Part 2 in two chromosomes"
    route_len_a = len(child_a.truck_route) - 2  # exclude depots
    route_len_b = len(child_b.truck_route) - 2

    if route_len_a < 1 or route_len_b < 1:
        return child_a, child_b

    pos_a = random.randrange(1, route_len_a + 1)  # position in route
    pos_b = random.randrange(1, route_len_b + 1)  # position in route

    node_a = child_a.truck_route[pos_a]
    node_b = child_b.truck_route[pos_b]

    # Cannot swap depot nodes
    if node_a in (inst.depot_start, inst.depot_end) or node_b in (inst.depot_start, inst.depot_end):
        return child_a, child_b

    # Exchange the nodes between the two chromosomes
    child_a.truck_route[pos_a] = node_b
    child_b.truck_route[pos_b] = node_a

    # Check for duplicates and repair
    infeasible_a = _find_duplicates_in_route(child_a, node_b)
    infeasible_b = _find_duplicates_in_route(child_b, node_a)

    if infeasible_a:
        child_a = repair(inst, child_a, infeasible_a)
    if infeasible_b:
        child_b = repair(inst, child_b, infeasible_b)

    return child_a, child_b


def _find_duplicates_in_route(sol: Solution, new_node: int) -> List[int]:
    """Find nodes that appear more than once in the solution"""
    route_nodes = [n for n in sol.truck_route if n not in (0, sol.truck_route[-1])]
    drone_nodes = sol.drone_customers

    counts = {}
    for n in route_nodes:
        counts[n] = counts.get(n, 0) + 1
    for n in drone_nodes:
        counts[n] = counts.get(n, 0) + 1

    duplicates = [n for n, c in counts.items() if c > 1]
    return duplicates


def multi_mode_mutation(inst: ProblemInstance, sol: Solution,
                        ltl: int) -> Solution:
    """
    Multi-mode mutation (Section IV-E2)

    Randomly applies one of N1 (Truck-to-Drone), N2 (Drone-to-Truck), or N3 (Swap).

    Rules:
    - If truck route length < LTL: use N2 or N3 (can't move more to drone)
    - If drone assignments < num_drones: use N1 or N3 (can't move more to truck)
    - Otherwise: any of N1, N2, N3
    """
    n_drones = inst.num_drones
    n_flights = len(sol.drone_customers)

    candidates = []

    truck_len = len([n for n in sol.truck_route
                     if n not in (inst.depot_start, inst.depot_end)])

    if truck_len <= ltl:
        candidates = [n2_drone_to_truck, n3_swap]
    elif n_flights < n_drones:
        candidates = [n1_truck_to_drone, n3_swap]
    else:
        candidates = [n1_truck_to_drone, n2_drone_to_truck, n3_swap]

    operator = random.choice(candidates)
    result = operator(inst, sol)

    return result
