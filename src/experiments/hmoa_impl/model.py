"""
Mo-CRPTW-mD Problem Model
Paper: Hybrid Multi-Objective Optimization Approach With Pareto Local Search
       for Collaborative Truck-Drone Routing Problems Considering Flexible Time Windows
IEEE TITS, 2022
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Customer:
    """Customer node with flexible time window"""
    id: int
    x: float
    y: float
    # Hard time window [a, b]
    a: float  # earliest
    b: float  # latest
    # Flexible time window [e, l] (extended from hard window)
    e: float  # earliest with tolerance
    l: float  # latest with tolerance
    service_time: float = 0.0  # service time (truck only)
    drone_eligible: bool = True

    def satisfaction(self, arrival_time: float) -> float:
        """Customer satisfaction φ_j(t) (Eq. 2)"""
        if self.a <= arrival_time <= self.b:
            return 1.0
        elif self.e <= arrival_time < self.a:
            return (arrival_time - self.e) / (self.a - self.e)
        elif self.b < arrival_time <= self.l:
            return (self.l - arrival_time) / (self.l - self.b)
        else:
            return 0.0


@dataclass
class DroneSpec:
    """Drone specifications"""
    id: int
    endurance: float  # max flight distance per sortie
    cost_per_km: float = 1.0


@dataclass
class DroneFlight:
    """A single drone delivery <d, i, j, k>"""
    drone_id: int
    launch_node_idx: int      # index in truck route (Part 2)
    customer_id: int          # customer served (Part 3)
    land_node_idx: int        # index in truck route (Part 4)
    # Computed fields
    launch_distance: float = 0.0
    return_distance: float = 0.0
    total_distance: float = 0.0


@dataclass
class Solution:
    """
    Solution representation with 5-part chromosome.

    Part 1 (truck_route): list of node IDs visited by truck [0, ..., N+1]
        where 0 = start depot, N+1 = end depot
    Part 2 (launch_idx): list of truck-route indices where drones are launched
    Part 3 (drone_customers): list of customer IDs served by drones
    Part 4 (land_idx): list of truck-route indices where drones are retrieved
    Part 5 (drone_ids): list of drone IDs for each flight

    Each column <launch_idx[i], drone_customers[i], land_idx[i], drone_ids[i]>
    forms one drone flight.
    """
    truck_route: List[int] = field(default_factory=list)
    launch_idx: List[int] = field(default_factory=list)
    drone_customers: List[int] = field(default_factory=list)
    land_idx: List[int] = field(default_factory=list)
    drone_ids: List[int] = field(default_factory=list)

    # Computed values (cached)
    cost: float = float('inf')
    satisfaction: float = 0.0
    feasible: bool = True

    def copy(self) -> 'Solution':
        """Deep copy of solution"""
        return Solution(
            truck_route=self.truck_route.copy(),
            launch_idx=self.launch_idx.copy(),
            drone_customers=self.drone_customers.copy(),
            land_idx=self.land_idx.copy(),
            drone_ids=self.drone_ids.copy(),
            cost=self.cost,
            satisfaction=self.satisfaction,
            feasible=self.feasible
        )

    def get_truck_node(self, idx: int) -> int:
        """Get node ID at given position in truck route"""
        return self.truck_route[idx]

    def get_flights(self) -> List[DroneFlight]:
        """Get all drone flights as DroneFlight objects"""
        flights = []
        for i in range(len(self.drone_ids)):
            flight = DroneFlight(
                drone_id=self.drone_ids[i],
                launch_node_idx=self.launch_idx[i],
                customer_id=self.drone_customers[i],
                land_node_idx=self.land_idx[i]
            )
            flights.append(flight)
        return flights

    def get_served_by_truck(self) -> set:
        """Get set of customer IDs served by truck"""
        served = set()
        for node in self.truck_route:
            if node > 0:  # not depot
                # Check if node appears in drone_customers
                if node not in self.drone_customers:
                    served.add(node)
        return served

    def get_served_by_drone(self) -> set:
        """Get set of customer IDs served by drones"""
        return set(self.drone_customers)


class ProblemInstance:
    """Mo-CRPTW-mD problem instance"""

    def __init__(self,
                 customers: List[Customer],
                 num_drones: int,
                 truck_cost_per_km: float = 1.0,
                 drone_endurance: float = 50.0,
                 drone_cost_per_km: float = 1.0,
                 wbli: float = 0.2,
                 wbui: float = 0.2):
        self.customers = customers
        self.num_drones = num_drones
        self.truck_cost_per_km = truck_cost_per_km
        self.drone_endurance = drone_endurance
        self.drone_cost_per_km = drone_cost_per_km
        self.wbli = wbli
        self.wbui = wbui

        # Build node lookup
        self.node_map = {c.id: c for c in customers}

        # Precompute distance matrix
        all_nodes = [0] + [c.id for c in customers] + [len(customers) + 1]
        n = len(all_nodes)
        self.truck_dist = [[0.0] * n for _ in range(n)]
        self.drone_dist = [[0.0] * n for _ in range(n)]

        coords = {}
        coords[0] = (0.0, 0.0)  # depot start
        for c in customers:
            coords[c.id] = (c.x, c.y)
        end_id = len(customers) + 1
        coords[end_id] = (0.0, 0.0)  # depot end

        for i in range(n):
            for j in range(n):
                ni = all_nodes[i]
                nj = all_nodes[j]
                # Manhattan distance for truck
                dx = abs(coords[ni][0] - coords[nj][0])
                dy = abs(coords[ni][1] - coords[nj][1])
                self.truck_dist[ni][nj] = dx + dy
                # Euclidean distance for drone
                self.drone_dist[ni][nj] = math.sqrt(dx * dx + dy * dy)

    @property
    def num_customers(self) -> int:
        return len(self.customers)

    @property
    def depot_start(self) -> int:
        return 0

    @property
    def depot_end(self) -> int:
        return self.num_customers + 1

    def generate_flexible_time_windows(self):
        """Generate flexible time windows [e, l] from hard windows [a, b]"""
        for c in self.customers:
            width = c.b - c.a
            c.e = c.a - self.wbli * width
            c.l = c.b + self.wbui * width

    def truck_distance(self, node_i: int, node_j: int) -> float:
        return self.truck_dist[node_i][node_j]

    def drone_distance(self, node_i: int, node_j: int) -> float:
        return self.drone_dist[node_i][node_j]

    def evaluate(self, sol: Solution) -> Tuple[float, float]:
        """
        Evaluate solution: (total_cost, total_satisfaction)
        f1 = truck_cost + drone_costs (Eq. 1)
        f2 = sum of customer satisfaction (Eq. 2, 3)
        """
        # Calculate truck cost
        truck_cost = 0.0
        route = sol.truck_route
        for i in range(len(route) - 1):
            truck_cost += self.truck_distance(route[i], route[i + 1])
        truck_cost *= self.truck_cost_per_km

        # Calculate drone cost and satisfaction
        drone_cost = 0.0
        total_satisfaction = 0.0

        # Track arrival times for customers
        arrival_times = {}

        # Simulate truck route to get arrival times
        current_time = 0.0
        for i in range(len(route) - 1):
            curr_node = route[i]
            next_node = route[i + 1]
            travel_time = self.truck_distance(curr_node, next_node)

            if next_node in [self.depot_start, self.depot_end]:
                current_time += travel_time
            else:
                current_time += travel_time
                # Check if this customer is served by truck (not in drone_customers)
                if next_node not in sol.drone_customers:
                    arrival_times[next_node] = current_time
                    # Service time at node
                    customer = self.node_map[next_node]
                    current_time += customer.service_time

        # Calculate drone flights
        flights = sol.get_flights()
        for flight in flights:
            launch_node = route[flight.launch_node_idx]
            cust_node = flight.customer_id
            land_node = route[flight.land_node_idx]

            d_launch = self.drone_distance(launch_node, cust_node)
            d_return = self.drone_distance(cust_node, land_node)
            flight.launch_distance = d_launch
            flight.return_distance = d_return
            flight.total_distance = d_launch + d_return

            drone_cost += (d_launch + d_return) * self.drone_cost_per_km

        total_cost = truck_cost + drone_cost

        # Calculate customer satisfaction
        # For customers served by truck
        for node_id, arr_time in arrival_times.items():
            if node_id in self.node_map:
                total_satisfaction += self.node_map[node_id].satisfaction(arr_time)

        # For customers served by drones - estimate arrival time at customer
        # For simplicity, estimate based on truck's position when drone is launched
        for flight in flights:
            cust_node = flight.customer_id
            if cust_node in self.node_map:
                # Estimate drone arrival time
                launch_node = route[flight.launch_node_idx]
                d_launch = self.drone_distance(launch_node, cust_node)
                # Approximate launch time from truck schedule
                launch_time = self._estimate_truck_time_at_idx(route, flight.launch_node_idx)
                drone_arrival = launch_time + d_launch  # assume same speed
                total_satisfaction += self.node_map[cust_node].satisfaction(drone_arrival)

        return total_cost, total_satisfaction

    def _estimate_truck_time_at_idx(self, route: List[int], idx: int) -> float:
        """Estimate truck arrival time at a given route position"""
        time = 0.0
        for i in range(min(idx, len(route) - 1)):
            time += self.truck_distance(route[i], route[i + 1])
            node = route[i + 1]
            if node not in [self.depot_start, self.depot_end] and node not in self.drone_customers_in_route(route):
                time += self.node_map[node].service_time if node in self.node_map else 0
        return time

    def drone_customers_in_route(self, route):
        """Helper - just skip service time for all nodes in route estimation"""
        return set()
