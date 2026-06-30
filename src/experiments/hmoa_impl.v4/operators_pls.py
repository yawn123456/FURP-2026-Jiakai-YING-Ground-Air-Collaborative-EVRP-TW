"""
Neighborhood operators N4, N5, N6 for PLS.
Strictly following the paper Section IV-D. Each operator does ONE operation.
If infeasible, returns the input solution unchanged.
"""
import numpy as np
from problem import Solution, ProblemInstance, DroneDelivery
from typing import Set


def operator_n4_2opt(instance: ProblemInstance, solution: Solution,
                     random_state: np.random.RandomState) -> Solution:
    """
    N4: 2-Opt Operator (Section IV-D.4).

    Performs 2-opt on Part 1 (truck route) by exchanging two edges.
    Customer nodes are randomly selected. Part 2 and Part 4 values
    are changed accordingly. Repair is used for infeasibility.
    If no feasible solution, returns input unchanged.
    """
    if len(solution.truck_route) < 4:
        return solution.copy()

    sol = solution.copy()
    n = len(sol.truck_route)

    # Randomly select two positions
    pos1 = random_state.randint(0, n - 2)
    pos2 = random_state.randint(pos1 + 2, n)

    # 2-opt: reverse segment between pos1+1 and pos2
    sol.truck_route[pos1 + 1:pos2 + 1] = list(reversed(sol.truck_route[pos1 + 1:pos2 + 1]))
    sol.clear_cache()

    # Repair potential infeasibility
    from operators import repair
    sol = repair(instance, sol, random_state)

    # Evaluate
    instance.evaluate(sol)

    if sol._feasible:
        return sol

    return solution.copy()  # Return input unchanged


def operator_n5_greedy_deletion_reinsertion(instance: ProblemInstance, solution: Solution,
                                            random_state: np.random.RandomState) -> Solution:
    """
    N5: Greedy-Deletion-Reinsertion (Section IV-D.5).

    Deletes the most expensive drone assignment (by flight distance),
    then uses Repair to reassign the deleted node to a drone.
    If no feasible solution, returns input unchanged.
    """
    if not solution.drone_deliveries:
        return solution.copy()

    sol = solution.copy()

    # Find the most expensive drone delivery
    max_dist = -1
    worst_idx = -1
    worst_node = -1

    for idx, dd in enumerate(sol.drone_deliveries):
        flight_dist = (instance.drone_distance(dd.launch_node, dd.customer) +
                      instance.drone_distance(dd.customer, dd.retrieve_node))
        if flight_dist > max_dist:
            max_dist = flight_dist
            worst_idx = idx
            worst_node = dd.customer

    if worst_idx < 0:
        return sol

    # Delete the expensive delivery
    sol.drone_deliveries.pop(worst_idx)
    sol.clear_cache()

    # Use Repair to reassign
    from operators import repair
    sol = repair(instance, sol, random_state)

    instance.evaluate(sol)

    if sol._feasible:
        return sol

    return solution.copy()  # Return input unchanged


def operator_n6_random_deletion_reinsertion(instance: ProblemInstance, solution: Solution,
                                            random_state: np.random.RandomState) -> Solution:
    """
    N6: Random-Deletion-Reinsertion (Section IV-D.6).

    Randomly deletes a drone assignment, then reassigns a random
    feasible assignment via Repair.
    If no feasible solution, returns input unchanged.
    """
    if not solution.drone_deliveries:
        return solution.copy()

    sol = solution.copy()

    # Randomly select a drone delivery to delete
    del_idx = random_state.randint(0, len(sol.drone_deliveries))
    removed_node = sol.drone_deliveries[del_idx].customer
    sol.drone_deliveries.pop(del_idx)
    sol.clear_cache()

    # Use Repair to reassign
    from operators import repair
    sol = repair(instance, sol, random_state)

    instance.evaluate(sol)

    if sol._feasible:
        return sol

    return solution.copy()  # Return input unchanged
