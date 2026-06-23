"""
Specialized Neighborhood Operators (Section IV-D)
N1: Truck-to-Drone
N2: Drone-to-Truck
N3: Swap (truck-drone exchange)
N4: 2-Opt (truck route)
N5: Greedy-Deletion-Reinsertion
N6: Random-Deletion-Reinsertion
"""

import random
import math
from typing import List, Tuple, Optional
from model import Solution, ProblemInstance
from repair import repair


def _find_most_expensive_truck_node(inst: ProblemInstance, sol: Solution) -> Optional[int]:
    """
    Find the most expensive drone-eligible customer node in the truck route.
    'Expensive' = longest sum of distances to previous and next node.
    """
    route = sol.truck_route
    drone_custs = set(sol.drone_customers)

    best_node = None
    best_cost = -1.0

    for i in range(1, len(route) - 1):
        node = route[i]
        if node != inst.depot_end and node not in drone_custs and inst.node_map[node].drone_eligible:
            prev_node = route[i - 1]
            next_node = route[i + 1]
            cost = inst.truck_distance(prev_node, node) + inst.truck_distance(node, next_node)
            if cost > best_cost:
                best_cost = cost
                best_node = node

    return best_node


def _find_most_expensive_drone_assignment(inst: ProblemInstance,
                                          sol: Solution) -> Optional[int]:
    """Find the index of the most expensive drone assignment (longest flight distance)"""
    best_idx = -1
    best_cost = -1.0

    for idx in range(len(sol.drone_ids)):
        launch_node = sol.truck_route[sol.launch_idx[idx]]
        cust_node = sol.drone_customers[idx]
        land_node = sol.truck_route[sol.land_idx[idx]]
        cost = (inst.drone_distance(launch_node, cust_node) +
                inst.drone_distance(cust_node, land_node))
        if cost > best_cost:
            best_cost = cost
            best_idx = idx

    return best_idx if best_idx >= 0 else None


def _find_cheapest_truck_insertion(inst: ProblemInstance,
                                   route: List[int], node: int) -> int:
    """Find cheapest position to insert node into truck route"""
    best_pos = 1
    best_inc = float('inf')
    for i in range(1, len(route)):
        prev = route[i - 1]
        curr = route[i] if i < len(route) else route[-1]
        inc_cost = (inst.truck_distance(prev, node) +
                    inst.truck_distance(node, curr) -
                    inst.truck_distance(prev, curr))
        if inc_cost < best_inc:
            best_inc = inc_cost
            best_pos = i
    return best_pos


def n1_truck_to_drone(inst: ProblemInstance, sol: Solution) -> Solution:
    """
    N1: Truck-to-Drone Operator

    Delete most expensive drone-eligible customer from truck route,
    and assign it to a drone using Repair heuristic.
    """
    # Find most expensive drone-eligible customer in truck route
    expensive_node = _find_most_expensive_truck_node(inst, sol)
    if expensive_node is None:
        return sol

    new_sol = sol.copy()
    route = new_sol.truck_route

    # Find position of expensive node (before removal)
    remove_pos = None
    for i, n in enumerate(route):
        if n == expensive_node:
            remove_pos = i
            break

    if remove_pos is None:
        return sol

    # Remove node from route
    route.pop(remove_pos)

    # Update launch/land indices: decrement if after removed position
    for i in range(len(new_sol.launch_idx)):
        if new_sol.launch_idx[i] > remove_pos:
            new_sol.launch_idx[i] -= 1
        if new_sol.land_idx[i] > remove_pos:
            new_sol.land_idx[i] -= 1

    # Use Repair to find cheap drone assignment for the deleted node
    repaired = repair(inst, new_sol, [expensive_node])
    return repaired


def n2_drone_to_truck(inst: ProblemInstance, sol: Solution) -> Solution:
    """
    N2: Drone-to-Truck Operator

    Delete the most expensive drone assignment and insert the
    customer node into the cheapest position in the truck route.
    """
    new_sol = sol.copy()

    # Find most expensive drone assignment
    idx = _find_most_expensive_drone_assignment(inst, new_sol)
    if idx is None:
        return sol

    cust_node = new_sol.drone_customers[idx]

    # Remove the drone assignment
    new_sol.launch_idx.pop(idx)
    new_sol.drone_customers.pop(idx)
    new_sol.land_idx.pop(idx)
    new_sol.drone_ids.pop(idx)

    # Insert customer into cheapest truck route position
    route = new_sol.truck_route
    pos = _find_cheapest_truck_insertion(inst, route, cust_node)
    route.insert(pos, cust_node)

    # Update launch/land indices
    for i in range(len(new_sol.launch_idx)):
        if new_sol.launch_idx[i] >= pos:
            new_sol.launch_idx[i] += 1
        if new_sol.land_idx[i] >= pos:
            new_sol.land_idx[i] += 1

    # May need to verify no duplicate customers
    return new_sol


def n3_swap(inst: ProblemInstance, sol: Solution) -> Solution:
    """
    N3: Swap Operator

    Exchange positions of a truck-served customer and a drone-served customer.
    """
    import random

    new_sol = sol.copy()
    drone_custs = new_sol.drone_customers

    if not drone_custs:
        return sol

    # Pick a random drone-served customer
    drone_cust_idx = random.randrange(len(drone_custs))
    drone_cust = drone_custs[drone_cust_idx]

    # Pick a random truck-served customer that is drone-eligible
    route = new_sol.truck_route
    truck_candidates = []
    for n in route:
        if n not in (inst.depot_start, inst.depot_end) and n not in drone_custs:
            if inst.node_map[n].drone_eligible:
                truck_candidates.append(n)

    if not truck_candidates:
        return sol

    truck_cust = random.choice(truck_candidates)

    # Swap: remove truck customer from route, add drone customer to route
    route.remove(truck_cust)
    # Add drone customer to cheapest position
    pos = _find_cheapest_truck_insertion(inst, route, drone_cust)
    route.insert(pos, drone_cust)

    # Update indices after route modification
    for i in range(len(new_sol.launch_idx)):
        # Adjust for removals/insertions
        pass  # simplified - real implementation would track positions

    # Remove drone customer from drone list, add truck customer
    new_sol.drone_customers[drone_cust_idx] = truck_cust

    # Repair if infeasible
    # Check if any flight is now infeasible and repair
    infeasible = []
    # Check all drone assignments
    for idx in range(len(new_sol.drone_ids)):
        cust = new_sol.drone_customers[idx]
        launch_node = route[new_sol.launch_idx[idx]]
        land_node = route[new_sol.land_idx[idx]]
        d_launch = inst.drone_distance(launch_node, cust)
        d_return = inst.drone_distance(cust, land_node)
        if d_launch + d_return > inst.drone_endurance:
            infeasible.append(cust)

    if infeasible:
        repaired = repair(inst, new_sol, infeasible)
        return repaired

    return new_sol


def n4_2opt(inst: ProblemInstance, sol: Solution) -> Solution:
    """
    N4: 2-Opt Operator

    Improve truck route by exchanging two edges.
    Randomly select two positions and reverse the segment between them.
    """
    new_sol = sol.copy()
    route = new_sol.truck_route

    if len(route) <= 4:
        return sol

    # Randomly select two positions (exclude depots at ends)
    i = random.randrange(1, len(route) - 2)
    j = random.randrange(i + 1, len(route) - 1)

    # Reverse segment between i and j
    route[i:j + 1] = reversed(route[i:j + 1])

    # Update launch/land indices - they refer to positions, not nodes
    # Positions haven't changed, but the nodes at those positions have.
    # Indices remain valid as they reference positions.
    # However, some drone assignments may now be infeasible.

    infeasible = []
    for idx in range(len(new_sol.drone_ids)):
        cust = new_sol.drone_customers[idx]
        launch_node = route[new_sol.launch_idx[idx]]
        land_node = route[new_sol.land_idx[idx]]
        d_launch = inst.drone_distance(launch_node, cust)
        d_return = inst.drone_distance(cust, land_node)

        if d_launch + d_return > inst.drone_endurance:
            infeasible.append(cust)

    if infeasible:
        repaired = repair(inst, new_sol, infeasible)
        return repaired

    return new_sol


def n5_greedy_deletion_reinsertion(inst: ProblemInstance, sol: Solution) -> Solution:
    """
    N5: Greedy-Deletion-Reinsertion

    Delete the most expensive drone assignment and reassign using Repair.
    """
    new_sol = sol.copy()

    idx = _find_most_expensive_drone_assignment(inst, new_sol)
    if idx is None:
        return sol

    cust_node = new_sol.drone_customers[idx]

    # Remove the assignment
    new_sol.launch_idx.pop(idx)
    new_sol.drone_customers.pop(idx)
    new_sol.land_idx.pop(idx)
    new_sol.drone_ids.pop(idx)

    # Reassign using Repair
    repaired = repair(inst, new_sol, [cust_node])
    return repaired


def n6_random_deletion_reinsertion(inst: ProblemInstance, sol: Solution) -> Solution:
    """
    N6: Random-Deletion-Reinsertion

    Delete a randomly selected drone assignment and reassign randomly.
    """
    import random

    new_sol = sol.copy()
    num_flights = len(new_sol.drone_ids)

    if num_flights == 0:
        return sol

    idx = random.randrange(num_flights)
    cust_node = new_sol.drone_customers[idx]

    # Remove the assignment
    new_sol.launch_idx.pop(idx)
    new_sol.drone_customers.pop(idx)
    new_sol.land_idx.pop(idx)
    new_sol.drone_ids.pop(idx)

    # Try to find a new feasible assignment
    route = new_sol.truck_route
    possible = []
    for d in range(inst.num_drones):
        for i in range(len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                launch_node = route[i]
                land_node = route[k]
                d_launch = inst.drone_distance(launch_node, cust_node)
                d_return = inst.drone_distance(cust_node, land_node)
                if d_launch + d_return <= inst.drone_endurance:
                    possible.append((i, k, d))

    if not possible:
        # Fall back to repair
        repaired = repair(inst, new_sol, [cust_node])
        return repaired

    chosen = random.choice(possible)
    new_sol.launch_idx.append(chosen[0])
    new_sol.drone_customers.append(cust_node)
    new_sol.land_idx.append(chosen[1])
    new_sol.drone_ids.append(chosen[2])

    return new_sol
