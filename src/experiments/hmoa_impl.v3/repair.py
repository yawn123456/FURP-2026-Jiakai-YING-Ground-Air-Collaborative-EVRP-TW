"""
Repair heuristic (Algorithm 4) - Reassign infeasible customer nodes

Used by neighborhood operators and crossover to fix infeasible solutions.
"""

from typing import List, Set, Tuple, Optional
from model import Solution, ProblemInstance


def repair(inst: ProblemInstance, sol: Solution,
           infeasible_nodes: List[int]) -> Solution:
    """
    Algorithm 4: Repair — OPTIMIZED

    Reassign infeasible customer nodes to feasible drone flights.
    Uses pre-computed nearest-neighbor lookups to avoid exhaustive search.

    Args:
        inst: Problem instance
        sol: Solution with infeasible nodes
        infeasible_nodes: List of customer node IDs that are infeasible

    Returns:
        Repaired solution (or original if cannot repair)
    """
    # Step 1: Remove assignments associated with infeasible nodes
    new_sol = sol.copy()
    _remove_assignments(new_sol, infeasible_nodes)

    route = new_sol.truck_route
    route_len = len(route)
    # Set of route nodes for fast lookup
    route_nodes = set(route)

    # Track existing drone assignments to check conflicts
    existing_flights = {}
    for idx in range(len(new_sol.drone_ids)):
        d = new_sol.drone_ids[idx]
        if d not in existing_flights:
            existing_flights[d] = set()
        existing_flights[d].add(new_sol.launch_idx[idx])
        existing_flights[d].add(new_sol.land_idx[idx])

    # Step 2: Build candidate assignments — OPTIMIZED with O(1) lookup
    P = []
    # Pre-build position lookup for O(1) access
    pos_lookup = {node: idx for idx, node in enumerate(route)}

    for cust_j in infeasible_nodes:
        # Only check nearest route nodes — use simple list + sort of top-K
        # Avoid full sort by using heapq for top-K
        route_distances = []
        for i, node in enumerate(route):
            if i == 0 or i >= route_len - 1:
                continue
            dist = inst.drone_distance(node, cust_j)
            route_distances.append((dist, i))
        # Sort by distance and take top 15
        route_distances.sort(key=lambda x: x[0])
        top_k_positions = route_distances[:15]

        # Pre-compute launch distances for this customer
        for _, launch_i in top_k_positions:
            launch_node = route[launch_i]
            d_launch = inst.drone_distance(launch_node, cust_j)

            if d_launch > inst.drone_endurance:
                continue

            for _, land_k in top_k_positions:
                if land_k <= launch_i:
                    continue

                land_node = route[land_k]
                d_return = inst.drone_distance(cust_j, land_node)

                if d_launch + d_return > inst.drone_endurance:
                    continue

                # Find first available drone
                for d in range(inst.num_drones):
                    used = existing_flights.get(d, set())
                    if launch_i not in used and land_k not in used:
                        P.append((launch_i, cust_j, land_k, d))
                        break

    # Step 3: Check if every infeasible node has at least one assignment
    infeasible_set = set(infeasible_nodes)
    nodes_with_assignments = set(a[1] for a in P)

    if not infeasible_set.issubset(nodes_with_assignments):
        return sol  # Cannot repair, return original

    # Step 4: Greedy assignment
    Pin = []
    unassigned = set(infeasible_nodes)

    while P and unassigned:
        # Find customer with LEAST potential assignments
        candidate_counts = {}
        for a in P:
            cid = a[1]
            if cid in unassigned:
                candidate_counts[cid] = candidate_counts.get(cid, 0) + 1

        if not candidate_counts:
            break

        least_choices_cust = min(candidate_counts, key=candidate_counts.get)

        # Find cheapest assignment for this customer
        candidates_for_j = [a for a in P if a[1] == least_choices_cust]

        def flight_cost(a):
            launch_node = route[a[0]]
            land_node = route[a[2]]
            return (inst.drone_distance(launch_node, a[1]) +
                    inst.drone_distance(a[1], land_node))

        candidates_for_j.sort(key=flight_cost)
        best = candidates_for_j[0]

        Pin.append(best)
        unassigned.remove(least_choices_cust)

        # Update P: remove conflicting assignments
        new_P = []
        for a in P:
            conflict = False
            # Remove assignments for the same customer
            if a[1] == best[1]:
                conflict = True
            # Remove assignments with same drone that conflict on positions
            if a[3] == best[3]:
                if (a[0] == best[0] or a[2] == best[0] or
                    a[0] == best[2] or a[2] == best[2]):
                    conflict = True
            if not conflict:
                new_P.append(a)
        P = new_P

    # Step 5: Build final solution
    if not unassigned:
        for a in Pin:
            new_sol.launch_idx.append(a[0])
            new_sol.drone_customers.append(a[1])
            new_sol.land_idx.append(a[2])
            new_sol.drone_ids.append(a[3])
        return new_sol
    else:
        return sol


def _remove_assignments(sol: Solution, node_ids: List[int]):
    """Remove drone assignments involving the given customer nodes"""
    node_set = set(node_ids)
    keep = []
    for idx in range(len(sol.drone_customers)):
        if sol.drone_customers[idx] not in node_set:
            keep.append(idx)

    sol.launch_idx = [sol.launch_idx[i] for i in keep]
    sol.drone_customers = [sol.drone_customers[i] for i in keep]
    sol.land_idx = [sol.land_idx[i] for i in keep]
    sol.drone_ids = [sol.drone_ids[i] for i in keep]
