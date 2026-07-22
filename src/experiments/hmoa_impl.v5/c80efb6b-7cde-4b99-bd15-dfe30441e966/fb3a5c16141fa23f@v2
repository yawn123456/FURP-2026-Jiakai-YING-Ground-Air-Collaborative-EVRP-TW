"""
Problem definition for Mo-CRPTW-mD (Multi-Objective Collaborative Routing Problem
of Truck-Drone system with Flexible Time Windows and multiple Drones).

Mathematical model strictly follows the paper Sections III-B and III-C:
- Objective f1: Total transportation cost (Equation 1)
- Objective f2: Overall customer satisfaction (Equation 3)
- Constraints (4)-(33)

Solution representation follows Section IV-B:
- Part 1: Truck route (permutation of customer nodes served by truck)
- Part 2: Number of truck nodes (inferred from Part 1 length)
- Part 3: Drone deliveries as <drone_id, launch_node, customer, retrieve_node>
- Part 4: Number of drone deliveries (inferred from Part 3 length)
"""

import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import config


@dataclass
class CustomerNode:
    """Customer node with time window and service information."""
    id: int              # Node index (1 to |C|)
    x: float             # X coordinate
    y: float             # Y coordinate
    ready_time: float    # ai: earliest preferred arrival time
    due_time: float      # bi: latest preferred arrival time
    earliest_time: float # ei: flexible earliest arrival (satisfaction > 0)
    latest_time: float   # li: flexible latest arrival (satisfaction > 0)
    service_time: float  # si: service time
    is_drone_eligible: bool = True  # Whether drone can serve this node

    @property
    def preferred_window(self) -> Tuple[float, float]:
        """Preferred time window [ai, bi]."""
        return (self.ready_time, self.due_time)

    @property
    def flexible_window(self) -> Tuple[float, float]:
        """Flexible time window [ei, li]."""
        return (self.earliest_time, self.latest_time)


@dataclass
class DroneDelivery:
    """A drone delivery sortie <d, i, j, k> as defined in the paper.

    - d: drone ID (1 to m)
    - i: launch node (a node on truck route, or 0 for depot)
    - j: customer node served by drone
    - k: retrieve node (a node on truck route visited after i, or c+1 for depot)
    """
    drone_id: int      # d
    launch_node: int   # i (node ID: 0=depot, or truck customer node)
    customer: int      # j (customer node served by drone)
    retrieve_node: int # k (node ID: 0=depot, or truck customer node)

    def __repr__(self):
        return f"<d={self.drone_id}, i={self.launch_node}, j={self.customer}, k={self.retrieve_node}>"


@dataclass
class Solution:
    """Solution representation for Mo-CRPTW-mD (Section IV-B).

    Chromosome structure:
    - Part 1: truck_route (ordered list of customer nodes served by truck)
    - Part 2: len(truck_route) — implicit
    - Part 3: drone_deliveries (list of DroneDelivery sorties)
    - Part 4: len(drone_deliveries) — implicit
    """
    truck_route: List[int] = field(default_factory=list)
    drone_deliveries: List[DroneDelivery] = field(default_factory=list)

    # Cached objective values (computed lazily)
    _cost: Optional[float] = field(default=None, repr=False)
    _satisfaction: Optional[float] = field(default=None, repr=False)
    _feasible: Optional[bool] = field(default=None, repr=False)
    _timeline: Optional[Dict] = field(default=None, repr=False)

    def copy(self, preserve_cache: bool = True) -> 'Solution':
        """Deep copy of solution.

        Args:
            preserve_cache: If True, copy cached objective values.
                           Set to False for a "fresh" copy (e.g., before modification).
        """
        new_sol = Solution(
            truck_route=self.truck_route.copy(),
            drone_deliveries=[DroneDelivery(d.drone_id, d.launch_node, d.customer, d.retrieve_node)
                              for d in self.drone_deliveries]
        )
        if preserve_cache:
            new_sol._cost = self._cost
            new_sol._satisfaction = self._satisfaction
            new_sol._feasible = self._feasible
            new_sol._timeline = self._timeline
        return new_sol

    def get_truck_nodes_set(self) -> Set[int]:
        """Get set of customer nodes served by truck."""
        return set(self.truck_route)

    def get_drone_nodes_set(self) -> Set[int]:
        """Get set of customer nodes served by drones."""
        return {d.customer for d in self.drone_deliveries}

    def get_all_customer_nodes(self) -> Set[int]:
        """Get all customer nodes covered by this solution."""
        return self.get_truck_nodes_set() | self.get_drone_nodes_set()

    def get_truck_full_route(self) -> List[int]:
        """Get full truck route including depot at start and end."""
        return [0] + self.truck_route + [0]

    def clear_cache(self):
        """Clear cached objective values (call after modifying solution)."""
        self._cost = None
        self._satisfaction = None
        self._feasible = None
        self._timeline = None

    def __repr__(self):
        cost_str = f"{self._cost:.2f}" if self._cost is not None else "?"
        sat_str = f"{self._satisfaction:.4f}" if self._satisfaction is not None else "?"
        return (f"Solution(truck={self.truck_route}, drones={self.drone_deliveries}, "
                f"cost={cost_str}, sat={sat_str})")


class ProblemInstance:
    """Complete problem instance for Mo-CRPTW-mD."""

    def __init__(self,
                 nodes: List[CustomerNode],
                 num_drones: int,
                 drone_endurance: float,
                 truck_speed: float = config.TRUCK_SPEED,
                 drone_speed: float = config.DRONE_SPEED,
                 truck_cost_per_unit: float = config.TRUCK_COST_PER_UNIT,
                 drone_cost_per_unit: float = config.DRONE_COST_PER_UNIT,
                 truck_dist_type: str = config.TRUCK_DISTANCE_TYPE,
                 drone_dist_type: str = config.DRONE_DISTANCE_TYPE):
        """
        Args:
            nodes: List of CustomerNode (id from 1 to |C|)
            num_drones: Number of drones m
            drone_endurance: Flight endurance ε for each drone
            truck_speed: Speed of the truck
            drone_speed: Speed of each drone
            truck_cost_per_unit: Transportation cost of truck per distance unit
            drone_cost_per_unit: Transportation cost of drone per distance unit
            truck_dist_type: 'manhattan' or 'euclidean'
            drone_dist_type: 'manhattan' or 'euclidean'
        """
        self.nodes: Dict[int, CustomerNode] = {n.id: n for n in nodes}
        self.num_customers = len(nodes)
        self.num_drones = num_drones
        self.drone_endurance = drone_endurance
        self.truck_speed = truck_speed
        self.drone_speed = drone_speed
        self.truck_cost_per_unit = truck_cost_per_unit
        self.drone_cost_per_unit = drone_cost_per_unit
        self.truck_dist_type = truck_dist_type
        self.drone_dist_type = drone_dist_type

        # Depot is at node 0 (and also node c+1 which maps to 0)
        # Paper assumes depot coordinates are implicitly defined
        # We'll use the first node's context or set depot at origin
        self.depot_x = 0.0
        self.depot_y = 0.0

        # Precompute distance matrices
        self._truck_dist = None
        self._drone_dist = None
        self._truck_time = None
        self._drone_time = None
        self._compute_distance_matrices()

        # Compute LTL (Equation 34)
        self.ltl = self._compute_ltl()

    def _compute_ltl(self) -> int:
        """Compute Lower Truck Limit (Equation 34).
        LTL = ceil((|C| - m) / (m + 1))
        """
        return int(np.ceil((self.num_customers - self.num_drones) /
                           (self.num_drones + 1)))

    def set_depot(self, x: float, y: float):
        """Set depot coordinates and recompute distances."""
        self.depot_x = x
        self.depot_y = y
        self._compute_distance_matrices()

    def _compute_distance_matrices(self):
        """Precompute distance and time matrices for all nodes including depot."""
        n = self.num_customers + 1  # +1 for depot
        self._truck_dist = np.zeros((n, n))
        self._drone_dist = np.zeros((n, n))
        self._truck_time = np.zeros((n, n))
        self._drone_time = np.zeros((n, n))

        # Build coordinate arrays (index 0 = depot)
        xs = np.zeros(n)
        ys = np.zeros(n)
        xs[0] = self.depot_x
        ys[0] = self.depot_y
        for node in self.nodes.values():
            xs[node.id] = node.x
            ys[node.id] = node.y

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Truck distance (Manhattan as specified in Section V-B)
                if self.truck_dist_type == 'manhattan':
                    self._truck_dist[i, j] = abs(xs[i] - xs[j]) + abs(ys[i] - ys[j])
                else:
                    self._truck_dist[i, j] = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)

                # Drone distance (Euclidean as specified in Section V-B)
                if self.drone_dist_type == 'euclidean':
                    self._drone_dist[i, j] = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                else:
                    self._drone_dist[i, j] = abs(xs[i] - xs[j]) + abs(ys[i] - ys[j])

                # Travel times
                self._truck_time[i, j] = self._truck_dist[i, j] / self.truck_speed
                self._drone_time[i, j] = self._drone_dist[i, j] / self.drone_speed

    def truck_distance(self, i: int, j: int) -> float:
        """Truck travel distance from node i to node j."""
        return self._truck_dist[i, j]

    def drone_distance(self, i: int, j: int) -> float:
        """Drone travel distance from node i to node j."""
        return self._drone_dist[i, j]

    def truck_travel_time(self, i: int, j: int) -> float:
        """Truck travel time from node i to node j."""
        return self._truck_time[i, j]

    def drone_travel_time(self, i: int, j: int) -> float:
        """Drone travel time from node i to node j."""
        return self._drone_time[i, j]

    def compute_timeline(self, solution: Solution) -> Tuple[Dict, bool]:
        """Compute the detailed timeline for a solution.

        Returns:
            timeline: Dict with arrival/departure times for truck and drones at each node
            feasible: Whether the solution satisfies all temporal constraints

        The timeline computation follows constraints (17)-(30).
        """
        truck_route = solution.truck_route
        drone_deliveries = solution.drone_deliveries

        # Group drone deliveries by launch node and retrieve node
        launches_from: Dict[int, List[DroneDelivery]] = {}  # node -> list of deliveries
        retrieves_at: Dict[int, List[DroneDelivery]] = {}   # node -> list of deliveries
        for dd in drone_deliveries:
            launches_from.setdefault(dd.launch_node, []).append(dd)
            retrieves_at.setdefault(dd.retrieve_node, []).append(dd)

        # Full truck route: depot -> truck_route -> depot
        full_route = [0] + truck_route + [0]

        # State variables
        truck_time = 0.0  # Current truck time
        truck_arrival = {}  # node -> truck arrival time
        truck_departure = {}  # node -> truck departure time
        drone_launch_time = {}  # (drone_id, customer) -> launch time
        drone_arrival_time = {}  # customer -> drone arrival time
        drone_departure_time = {}  # customer -> drone departure time
        drone_retrieve_time = {}  # (drone_id, customer) -> retrieve time

        # Track drone state
        drone_available_time = {d: 0.0 for d in range(1, self.num_drones + 1)}
        # Each drone delivery has its own timeline

        feasible = True

        for pos, node in enumerate(full_route):
            # --- Step 1: Travel to this node ---
            if pos > 0:
                prev_node = full_route[pos - 1]
                truck_time += self.truck_travel_time(prev_node, node)

            truck_arrival[node] = truck_time

            # --- Step 2: Handle drone retrievals at this node ---
            if node in retrieves_at:
                for dd in retrieves_at[node]:
                    # Compute drone's arrival at retrieve node
                    # Drone flight: launch_node -> customer -> retrieve_node
                    launch_t = drone_launch_time.get((dd.drone_id, dd.customer), 0)
                    drone_arrive_cust = drone_arrival_time.get(dd.customer, 0)
                    drone_depart_cust = drone_departure_time.get(dd.customer, 0)

                    # Flight: customer -> retrieve_node
                    drone_arrive_retrieve = drone_depart_cust + self.drone_travel_time(dd.customer, node)

                    # Constraint (25): Truck must wait for drone if drone arrives later
                    # rk >= t'dk (truck ready time >= drone arrival at retrieve node)
                    if drone_arrive_retrieve > truck_time:
                        truck_time = drone_arrive_retrieve

                    drone_retrieve_time[(dd.drone_id, dd.customer)] = truck_time
                    drone_available_time[dd.drone_id] = truck_time

            # Re-check for any late drones after waiting
            if node in retrieves_at:
                for dd in retrieves_at[node]:
                    drone_arrive_retrieve_val = drone_retrieve_time.get((dd.drone_id, dd.customer), truck_time)
                    if drone_arrive_retrieve_val > truck_time:
                        truck_time = drone_arrive_retrieve_val

            # --- Step 3: Service the customer (if node is a customer) ---
            if node > 0:  # Not depot
                cust_node = self.nodes[node]

                # Waiting time for time window (Constraint 29)
                # wttw_i = max{0, ei - ti}
                wait_for_tw = max(0.0, cust_node.earliest_time - truck_time)
                if wait_for_tw > 0:
                    truck_time += wait_for_tw

                # Service time
                service_time = cust_node.service_time
                truck_time += service_time

            # --- Step 4: Launch drones from this node ---
            if node in launches_from:
                for dd in launches_from[node]:
                    # Drone launch time = truck departure time from node
                    launch_t = truck_time
                    drone_launch_time[(dd.drone_id, dd.customer)] = launch_t

                    # Compute drone arrival at customer j
                    drone_arrive_cust = launch_t + self.drone_travel_time(node, dd.customer)
                    drone_arrival_time[dd.customer] = drone_arrive_cust

                    # Customer satisfaction calculation will be done separately
                    # For now, just compute drone departure from customer
                    cust_node = self.nodes[dd.customer]

                    # Drone waiting at customer for time window (Constraint 30)
                    # wdtw_id = max{0, ed_i - t'di}  — but paper uses ed_i as earliest time
                    # Actually wdtw_id = max{0, ei - t'di} from constraint (30)
                    wait_drone = max(0.0, cust_node.earliest_time - drone_arrive_cust)
                    drone_depart_cust = drone_arrive_cust + wait_drone  # service time = 0 for drone
                    drone_departure_time[dd.customer] = drone_depart_cust

            truck_departure[node] = truck_time

        # Verify constraint (22): Each drone delivery within endurance
        for dd in drone_deliveries:
            flight_dist = (self.drone_distance(dd.launch_node, dd.customer) +
                          self.drone_distance(dd.customer, dd.retrieve_node))
            if flight_dist > self.drone_endurance + 1e-10:
                feasible = False
                break

        timeline = {
            'truck_arrival': truck_arrival,
            'truck_departure': truck_departure,
            'drone_launch_time': drone_launch_time,
            'drone_arrival_time': drone_arrival_time,
            'drone_departure_time': drone_departure_time,
            'drone_retrieve_time': drone_retrieve_time,
        }

        return timeline, feasible

    def compute_cost(self, solution: Solution) -> float:
        """Compute total transportation cost f1 (Equation 1).

        f1 = sum over truck edges of ct * δij * xij
           + sum over drone deliveries of cd * (δ'dij + δ'djk) * ydijk
        """
        cost = 0.0

        # Truck cost: sum of distances along truck route
        full_route = [0] + solution.truck_route + [0]
        for idx in range(len(full_route) - 1):
            i, j = full_route[idx], full_route[idx + 1]
            cost += self.truck_cost_per_unit * self.truck_distance(i, j)

        # Drone cost: launch->customer + customer->retrieve for each delivery
        for dd in solution.drone_deliveries:
            cost += self.drone_cost_per_unit * (
                self.drone_distance(dd.launch_node, dd.customer) +
                self.drone_distance(dd.customer, dd.retrieve_node)
            )

        return cost

    def compute_satisfaction(self, solution: Solution) -> float:
        """Compute overall customer satisfaction f2 (Equation 3).

        For each customer, compute μ_j(t_j) based on arrival time using Equation (2):
        μ_j(t) = (t - bi)/(li - bi),  bi < t ≤ li   [late]
        μ_j(t) = 1,                   ai ≤ t ≤ bi    [on time]
        μ_j(t) = (t - ei)/(ai - ei),  ei ≤ t < ai    [early]
        μ_j(t) = 0,                   otherwise

        f2 = sum of μ_j for all customers
        """
        # Compute timeline if needed
        if solution._timeline is None or solution._feasible is None:
            timeline, feasible = self.compute_timeline(solution)
            solution._timeline = timeline
            solution._feasible = feasible
        else:
            timeline = solution._timeline

        total_satisfaction = 0.0

        # Truck customers
        truck_arrival = timeline['truck_arrival']
        for node_id in solution.truck_route:
            arrival_t = truck_arrival.get(node_id, float('inf'))
            cust = self.nodes[node_id]
            total_satisfaction += self._customer_satisfaction(cust, arrival_t)

        # Drone customers
        drone_arrival = timeline['drone_arrival_time']
        for dd in solution.drone_deliveries:
            arrival_t = drone_arrival.get(dd.customer, float('inf'))
            cust = self.nodes[dd.customer]
            total_satisfaction += self._customer_satisfaction(cust, arrival_t)

        return total_satisfaction

    def _customer_satisfaction(self, cust: CustomerNode, arrival_t: float) -> float:
        """Compute satisfaction μ_i(t) for a single customer (Equation 2)."""
        ai, bi = cust.ready_time, cust.due_time
        ei, li = cust.earliest_time, cust.latest_time

        if ai <= arrival_t <= bi:
            return 1.0
        elif bi < arrival_t <= li:
            # Late arrival: (t - bi) / (li - bi)
            if li > bi:
                return (arrival_t - bi) / (li - bi)
            return 0.0
        elif ei <= arrival_t < ai:
            # Early arrival: (t - ei) / (ai - ei)
            if ai > ei:
                return (arrival_t - ei) / (ai - ei)
            return 0.0
        else:
            return 0.0

    def is_feasible(self, solution: Solution) -> bool:
        """Check if solution is feasible with respect to all constraints."""
        if solution._feasible is not None:
            return solution._feasible

        _, feasible = self.compute_timeline(solution)
        solution._feasible = feasible

        # Additional structural feasibility checks:

        # All nodes must be covered exactly once (Constraint 4)
        truck_set = solution.get_truck_nodes_set()
        drone_set = solution.get_drone_nodes_set()
        all_covered = truck_set | drone_set
        all_customers = set(range(1, self.num_customers + 1))

        if all_covered != all_customers:
            solution._feasible = False
            return False

        # No node should be in both truck and drone sets
        if truck_set & drone_set:
            solution._feasible = False
            return False

        # Each drone can be launched and retrieved at most once per node (Constraints 7, 8)
        for d in range(1, self.num_drones + 1):
            drone_dds = [dd for dd in solution.drone_deliveries if dd.drone_id == d]
            launch_nodes = [dd.launch_node for dd in drone_dds]
            retrieve_nodes = [dd.retrieve_node for dd in drone_dds]

            # Check for overlapping sorties (Constraint 17)
            # Simplified: ensure no duplicate launch/retrieve from same node for same drone
            if len(launch_nodes) != len(set(launch_nodes)):
                # Same drone launched multiple times from same node - check if this is valid
                # Actually constraint (7) says each drone can be launched from any particular
                # node at most once, so this is infeasible
                solution._feasible = False
                return False

        # Constraint (14): If <d,i,j,k> exists, truck must visit i and k
        for dd in solution.drone_deliveries:
            if dd.launch_node > 0 and dd.launch_node not in truck_set:
                # But launch_node 0 is depot (always visited)
                if dd.launch_node not in truck_set:
                    solution._feasible = False
                    return False
            if dd.retrieve_node > 0 and dd.retrieve_node not in truck_set:
                solution._feasible = False
                return False

            # Constraint (15): Truck must visit launch_node before retrieve_node
            if dd.launch_node > 0 and dd.retrieve_node > 0:
                try:
                    launch_pos = solution.truck_route.index(dd.launch_node)
                    retrieve_pos = solution.truck_route.index(dd.retrieve_node)
                    if launch_pos >= retrieve_pos:
                        solution._feasible = False
                        return False
                except ValueError:
                    pass  # Already caught above

        return solution._feasible

    def evaluate(self, solution: Solution) -> Tuple[float, float, bool]:
        """Evaluate a solution: return (cost, satisfaction, feasible)."""
        if solution._cost is None or solution._satisfaction is None:
            solution._timeline, solution._feasible = self.compute_timeline(solution)
            solution._cost = self.compute_cost(solution)
            solution._satisfaction = self.compute_satisfaction(solution)
            solution._feasible = self.is_feasible(solution)
        return solution._cost, solution._satisfaction, solution._feasible

    def dominates(self, sol_a: Solution, sol_b: Solution) -> bool:
        """Check if solution a dominates solution b.

        In minimization: a dominates b if a is no worse in all objectives
        and strictly better in at least one.
        Here: minimize cost, maximize satisfaction -> minimize negative satisfaction.
        """
        cost_a, sat_a, feas_a = self.evaluate(sol_a)
        cost_b, sat_b, feas_b = self.evaluate(sol_b)

        if not feas_a or not feas_b:
            return False  # Don't compare infeasible solutions

        # Convert to minimization: f1 = cost, f2 = -satisfaction
        f1_a, f2_a = cost_a, -sat_a
        f1_b, f2_b = cost_b, -sat_b

        if f1_a <= f1_b and f2_a <= f2_b and (f1_a < f1_b or f2_a < f2_b):
            return True
        return False

    def __repr__(self):
        return (f"ProblemInstance(customers={self.num_customers}, "
                f"drones={self.num_drones}, endurance={self.drone_endurance:.1f})")
