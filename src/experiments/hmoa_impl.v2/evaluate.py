"""
Solution evaluation - Objective functions f1 (cost) and f2 (customer satisfaction)
"""

from typing import List, Tuple
from model import Solution, ProblemInstance, DroneFlight


def evaluate(inst: ProblemInstance, sol: Solution,
             compute_times: bool = True) -> Tuple[float, float]:
    """
    Evaluate solution: (f1_total_cost, f2_total_satisfaction)

    f1 = truck_cost * sum(truck_edges) + sum(drone_cost * drone_distances)  (Eq. 1)
    f2 = sum of customer satisfaction φ_j(arrival_time)                       (Eq. 2, 3)
    """
    route = sol.truck_route

    # --- Truck cost ---
    truck_dist = 0.0
    for i in range(len(route) - 1):
        truck_dist += inst.truck_distance(route[i], route[i + 1])
    truck_cost = truck_dist * inst.truck_cost_per_km

    # --- Drone cost ---
    drone_cost = 0.0
    for i in range(len(sol.drone_ids)):
        launch_node = route[sol.launch_idx[i]]
        cust_node = sol.drone_customers[i]
        land_node = route[sol.land_idx[i]]
        d_launch = inst.drone_distance(launch_node, cust_node)
        d_return = inst.drone_distance(cust_node, land_node)
        drone_cost += (d_launch + d_return) * inst.drone_cost_per_km

    # --- Customer satisfaction ---
    total_satisfaction = 0.0

    if compute_times:
        # Simulate timeline
        arrival_times = _compute_arrival_times(inst, sol)
        for node_id, arr_time in arrival_times.items():
            if node_id in inst.node_map:
                total_satisfaction += inst.node_map[node_id].satisfaction(arr_time)

        # Drone customer arrivals
        for idx in range(len(sol.drone_ids)):
            cust_id = sol.drone_customers[idx]
            launch_node = route[sol.launch_idx[idx]]
            d_launch = inst.drone_distance(launch_node, cust_id)

            drone_arrival = _estimate_node_time(inst, route, sol.launch_idx[idx])
            drone_arrival += d_launch  # drone travel time

            if cust_id in inst.node_map:
                total_satisfaction += inst.node_map[cust_id].satisfaction(drone_arrival)

    f1 = truck_cost + drone_cost
    f2 = total_satisfaction
    return f1, f2


def _compute_arrival_times(inst: ProblemInstance, sol: Solution) -> dict:
    """Simulate truck route and return {node_id: arrival_time} for truck-served nodes"""
    route = sol.truck_route
    arrival_times = {}
    drone_custs = set(sol.drone_customers)
    current_time = 0.0

    for i in range(len(route) - 1):
        curr, nxt = route[i], route[i + 1]
        travel = inst.truck_distance(curr, nxt)
        current_time += travel

        if nxt not in (inst.depot_start, inst.depot_end) and nxt not in drone_custs:
            arrival_times[nxt] = current_time
            if nxt in inst.node_map:
                current_time += inst.node_map[nxt].service_time

    return arrival_times


def _estimate_node_time(inst: ProblemInstance, route: List[int],
                        route_idx: int) -> float:
    """Estimate arrival time at a given truck-route position (index)"""
    drone_custs = set()
    time = 0.0
    for i in range(min(route_idx, len(route) - 1)):
        time += inst.truck_distance(route[i], route[i + 1])
        node = route[i + 1]
        if node not in (inst.depot_start, inst.depot_end) and node not in drone_custs:
            if node in inst.node_map:
                time += inst.node_map[node].service_time
    return time
