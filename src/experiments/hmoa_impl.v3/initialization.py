"""
AssignNodes - Greedy initial solution generation (Algorithm 2)

Also includes the NearestNeighborTW heuristic for truck route generation.
"""

import math
import random
from typing import List, Tuple, Optional
from model import Solution, ProblemInstance, Customer


def nearest_neighbor_tw(inst: ProblemInstance, nodes: List[int]) -> List[int]:
    """
    Nearest Neighbor heuristic with time window consideration.
    Generate a permutation of customer nodes starting/ending at depot.
    """
    if not nodes:
        return [inst.depot_start, inst.depot_end]

    unvisited = set(nodes)
    route = [inst.depot_start]
    current = inst.depot_start
    current_time = 0.0

    while unvisited:
        # Find nearest feasible unvisited node
        best_node = None
        best_dist = float('inf')

        for node in unvisited:
            dist = inst.truck_distance(current, node)
            cust = inst.node_map[node]
            # Consider time window: prefer nodes we can reach
            arrival = current_time + dist
            if arrival <= cust.l:  # feasible within flexible window
                if dist < best_dist:
                    best_dist = dist
                    best_node = node

        # If no feasible node found in flexible window, take nearest anyway
        if best_node is None:
            for node in unvisited:
                dist = inst.truck_distance(current, node)
                if dist < best_dist:
                    best_dist = dist
                    best_node = node

        # Travel to best node
        current_time += best_dist
        if best_node not in (inst.depot_start, inst.depot_end):
            cust = inst.node_map[best_node]
            # Wait if early
            if current_time < cust.e:
                current_time = cust.e
            # Add service time
            current_time += cust.service_time

        route.append(best_node)
        unvisited.remove(best_node)
        current = best_node

    route.append(inst.depot_end)
    return route


def assign_nodes(inst: ProblemInstance, num_solutions: int) -> List[Solution]:
    """
    Algorithm 2: AssignNodes - Create initial population

    Optimized version with pre-computed nearest-neighbor lookups to avoid
    exhaustive O(n^4) search on large instances.

    Args:
        inst: Problem instance
        num_solutions: Population size n

    Returns:
        List of initial solutions
    """
    all_cust_ids = [c.id for c in inst.customers]
    n_cust = len(all_cust_ids)
    m = inst.num_drones

    # LTL: Lower Truck Limit (Eq. 34)
    ltl = math.ceil((n_cust - m) / (m + 1))
    ltl = max(1, min(ltl, n_cust - 1))

    # ---- Pre-compute nearest truck-stop nodes for each customer ----
    # For each customer j, pre-sort all possible truck-stop nodes by drone distance.
    # A "truck-stop" is any node in the truck route (including depot) that can be
    # a launch or land point.
    all_possible_stops = list(range(1, n_cust + 1)) + [0, inst.depot_end]

    # nearest_stops[j] = list of truck-stop nodes sorted by drone distance from j
    nearest_stops = {}
    for j in all_cust_ids:
        stops_with_dist = []
        for stop in all_possible_stops:
            if stop == j:
                continue
            dist = inst.drone_distance(stop, j)
            stops_with_dist.append((dist, stop))
        stops_with_dist.sort(key=lambda x: x[0])
        nearest_stops[j] = [s for _, s in stops_with_dist]

    population = []

    while len(population) < num_solutions:
        # Randomly divide nodes into truck group (Ct) and drone group (Cd)
        shuffled = all_cust_ids.copy()
        random.shuffle(shuffled)

        ct_size = max(ltl, random.randint(ltl, min(ltl + 3, n_cust)))
        ct_size = min(ct_size, n_cust - 1)

        ct = shuffled[:ct_size]
        cd = shuffled[ct_size:]

        # Move non-drone-eligible nodes from cd to ct
        cd_eligible = []
        for n in cd:
            if inst.node_map[n].drone_eligible:
                cd_eligible.append(n)
            else:
                ct.append(n)
        cd = cd_eligible
        if not cd:
            continue

        # Generate truck route with NearestNeighborTW
        ct_route = nearest_neighbor_tw(inst, ct)

        # Greedily assign drone nodes (OPTIMIZED)
        assignments = []  # list of (launch_idx, cust_id, land_idx, drone_id)
        unassigned = set(cd)
        used_positions = {d: set() for d in range(m)}
        route_len = len(ct_route)

        # Build position lookup dict for O(1) position queries
        node_to_pos = {node: idx for idx, node in enumerate(ct_route)}
        route_node_set = set(ct_route)

        while unassigned:
            candidates = []

            for cust_j in list(unassigned):
                best_for_node = None
                best_cost_for_node = float('inf')

                # Only check top-K nearest stops as launch candidates
                nearest_for_j = nearest_stops.get(cust_j, [])
                launch_candidates = [s for s in nearest_for_j
                                     if s in route_node_set and s != inst.depot_end]
                launch_candidates = launch_candidates[:15]

                # Pre-compute launch distances for this customer
                launch_dists = {}
                for launch_node in launch_candidates:
                    launch_dists[launch_node] = inst.drone_distance(launch_node, cust_j)

                for launch_node in launch_candidates:
                    launch_i = node_to_pos.get(launch_node)
                    if launch_i is None or launch_i >= route_len - 1:
                        continue

                    d_launch = launch_dists[launch_node]

                    # Check land candidates AFTER launch position
                    for land_node in launch_candidates:
                        if land_node == launch_node:
                            continue
                        land_k = node_to_pos.get(land_node)
                        if land_k is None or land_k <= launch_i:
                            continue

                        d_return = inst.drone_distance(cust_j, land_node)

                        if d_launch + d_return > inst.drone_endurance:
                            continue

                        cost = (d_launch + d_return) * inst.drone_cost_per_km

                        # Find first available drone
                        for d in range(m):
                            used = used_positions[d]
                            if launch_i not in used and land_k not in used:
                                if cost < best_cost_for_node:
                                    best_cost_for_node = cost
                                    best_for_node = (launch_i, cust_j, land_k, d)
                                break

                if best_for_node is not None:
                    candidates.append(best_for_node)

            if not candidates:
                # Cannot assign remaining nodes, move to truck
                for node in list(unassigned):
                    best_pos = route_len - 1
                    best_inc = float('inf')
                    for pos in range(1, route_len - 1):
                        prev = ct_route[pos - 1]
                        curr = ct_route[pos]
                        inc = (inst.truck_distance(prev, node) +
                               inst.truck_distance(node, curr) -
                               inst.truck_distance(prev, curr))
                        if inc < best_inc:
                            best_inc = inc
                            best_pos = pos
                    ct_route.insert(best_pos, node)
                    route_len += 1
                break

            # Pick the cheapest overall flight
            candidates.sort(key=lambda x: (
                inst.drone_distance(ct_route[x[0]], x[1]) +
                inst.drone_distance(x[1], ct_route[x[2]])
            ))
            best = candidates[0]
            launch_i, cust_j, land_k, d = best
            assignments.append(best)
            unassigned.remove(cust_j)
            used_positions[d].add(launch_i)
            used_positions[d].add(land_k)

        # Build solution from assignments
        sol = Solution()
        sol.truck_route = ct_route
        for a in assignments:
            sol.launch_idx.append(a[0])
            sol.drone_customers.append(a[1])
            sol.land_idx.append(a[2])
            sol.drone_ids.append(a[3])

        # Ensure all customers are assigned
        all_assigned = set()
        for node in ct_route:
            if node not in (inst.depot_start, inst.depot_end):
                all_assigned.add(node)
        all_assigned.update(sol.drone_customers)

        if len(all_assigned) == n_cust:
            population.append(sol)

    return population
