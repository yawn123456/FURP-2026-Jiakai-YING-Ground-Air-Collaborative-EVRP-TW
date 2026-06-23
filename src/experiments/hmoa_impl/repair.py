"""
Repair heuristic (Algorithm 4) - Reassign infeasible customer nodes

Used by neighborhood operators and crossover to fix infeasible solutions.
"""

from typing import List, Set, Tuple, Optional
from model import Solution, ProblemInstance


def repair(inst: ProblemInstance, sol: Solution,
           infeasible_nodes: List[int]) -> Solution:
    """
    Algorithm 4: Repair

    Reassign infeasible customer nodes to feasible drone flights.

    Args:
        inst: Problem instance
        sol: Solution with infeasible nodes
        infeasible_nodes: List of customer node IDs that are infeasible

    Returns:
        Repaired solution (or original if cannot repair)
    """
    # Step 1: Remove assignments associated with infeasible nodes
    # Make a copy of the solution minus the infeasible assignments
    new_sol = sol.copy()
    _remove_assignments(new_sol, infeasible_nodes)

    # Step 2: Build set of all possible feasible assignments
    # For each infeasible node, find all feasible <i, j, k, d>
    Ct = [n for n in new_sol.truck_route
          if n not in (inst.depot_start, inst.depot_end)]

    # Track existing drone assignments to check conflicts
    existing_flights = []
    for idx in range(len(new_sol.drone_ids)):
        existing_flights.append({
            'drone_id': new_sol.drone_ids[idx],
            'launch_idx': new_sol.launch_idx[idx],
            'land_idx': new_sol.land_idx[idx],
            'cust_id': new_sol.drone_customers[idx]
        })

    # P = all possible feasible assignments (line 8 of Algorithm 4)
    P = []  # each entry: (launch_idx, cust_id, land_idx, drone_id)
    for d in range(inst.num_drones):
        for i in range(len(new_sol.truck_route) - 2):  # launch positions
            for k in range(i + 1, len(new_sol.truck_route) - 1):  # land positions
                for cust_j in infeasible_nodes:
                    launch_node = new_sol.truck_route[i]
                    land_node = new_sol.truck_route[k]
                    cust = inst.node_map[cust_j]

                    d_launch = inst.drone_distance(launch_node, cust_j)
                    d_return = inst.drone_distance(cust_j, land_node)
                    total = d_launch + d_return

                    # Feasibility check: endurance constraint
                    if total > inst.drone_endurance:
                        continue

                    # Conflict check with existing flights
                    conflict = False
                    for ef in existing_flights:
                        if ef['drone_id'] == d:
                            if (ef['launch_idx'] == i or ef['land_idx'] == i or
                                ef['launch_idx'] == k or ef['land_idx'] == k):
                                conflict = True
                                break

                    if not conflict:
                        P.append((i, cust_j, k, d))

    # Step 3: Check if every infeasible node has at least one assignment in P
    infeasible_set = set(infeasible_nodes)
    nodes_with_assignments = set(a[1] for a in P)

    if not infeasible_set.issubset(nodes_with_assignments):
        return sol  # Cannot repair, return original

    # Step 4: Greedy assignment (lines 10-16)
    Pin = []  # Selected assignments
    unassigned = set(infeasible_nodes)

    while P and unassigned:
        # Find customer with LEAST potential assignments (line 11)
        candidate_counts = {}
        for a in P:
            cid = a[1]
            if cid in unassigned:
                candidate_counts[cid] = candidate_counts.get(cid, 0) + 1

        if not candidate_counts:
            break

        # Customer with fewest options
        least_choices_cust = min(candidate_counts, key=candidate_counts.get)

        # Find cheapest assignment for this customer (line 12)
        candidates_for_j = [a for a in P if a[1] == least_choices_cust]
        candidates_for_j.sort(key=lambda a: (
            inst.drone_distance(inst.node_map[a[1]].id, inst.node_map[a[1]].id) +  # placeholder
            inst.drone_distance(new_sol.truck_route[a[0]], a[1]) +
            inst.drone_distance(a[1], new_sol.truck_route[a[2]])
        ))

        # Actually compute cost for sorting
        def flight_cost(a):
            launch_node = new_sol.truck_route[a[0]]
            land_node = new_sol.truck_route[a[2]]
            return (inst.drone_distance(launch_node, a[1]) +
                    inst.drone_distance(a[1], land_node))

        candidates_for_j.sort(key=flight_cost)
        best = candidates_for_j[0]

        # Add to Pin (line 13)
        Pin.append(best)
        unassigned.remove(least_choices_cust)

        # Update P: remove conflicting assignments (line 14)
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
    if not unassigned:  # All nodes assigned
        for a in Pin:
            new_sol.launch_idx.append(a[0])
            new_sol.drone_customers.append(a[1])
            new_sol.land_idx.append(a[2])
            new_sol.drone_ids.append(a[3])
        return new_sol
    else:
        return sol  # Cannot fully repair


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
