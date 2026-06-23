"""
Benchmark Instance Generator

Extends the TSPTW benchmark instances (Dumas et al., 1995)
to create Mo-CRPTW-mD instances with:
- Flexible time windows [e, l] from hard windows [a, b]
- Drone eligibility (85% of nodes)
- Drone endurance filter (35% of feasible flights)
- Manhattan distance for truck, Euclidean distance for drone
"""

import math
import random
from typing import List, Tuple
from model import Customer, ProblemInstance


def generate_random_instance(
    num_customers: int,
    num_drones: int = 3,
    area_size: float = 100.0,
    tw_width: float = 80.0,
    tw_horizon: float = 480.0,
    wbli: float = 0.2,
    wbui: float = 0.2,
    drone_eligible_ratio: float = 0.85,
    endurance_ratio: float = 0.35,
    service_time_range: Tuple[float, float] = (5.0, 15.0),
    seed: int = 42
) -> ProblemInstance:
    """
    Generate a random test instance similar to extended TSPTW benchmarks.

    Args:
        num_customers: Number of customer nodes
        num_drones: Number of drones
        area_size: Size of the coordinate area (depot at corner)
        tw_width: Width of time windows
        tw_horizon: Max time for time windows
        wbli/wbui: Flexible window parameters
        drone_eligible_ratio: % of drone-eligible customers
        endurance_ratio: % of feasible flights for endurance setting
        service_time_range: (min, max) service time for truck customers
        seed: Random seed

    Returns:
        ProblemInstance
    """
    rng = random.Random(seed)

    # Depot at (0, 0)
    # Generate customer nodes randomly
    customers = []
    for i in range(1, num_customers + 1):
        x = rng.uniform(0, area_size)
        y = rng.uniform(0, area_size)

        # Time window [a, b]
        a = rng.uniform(0, tw_horizon - tw_width)
        b = a + tw_width

        # Flexible time window [e, l]
        e = a - wbli * tw_width
        l = b + wbui * tw_width

        service = rng.uniform(*service_time_range)
        eligible = rng.random() < drone_eligible_ratio

        customers.append(Customer(
            id=i, x=x, y=y,
            a=a, b=b, e=e, l=l,
            service_time=service,
            drone_eligible=eligible
        ))

    # Create instance
    inst = ProblemInstance(
        customers=customers,
        num_drones=num_drones,
        truck_cost_per_km=25.0,
        drone_cost_per_km=1.0,
        drone_endurance=float('inf'),  # Will be set below
        wbli=wbli,
        wbui=wbui
    )

    # Set drone endurance based on 35% feasible flights rule
    # (Same method as Moshref-Javadi et al.)
    all_distances = []
    for i in range(1, num_customers + 1):
        for j in range(1, num_customers + 1):
            if i != j:
                d = inst.drone_distance(i, j)
                all_distances.append(d)

    if all_distances:
        all_distances.sort()
        endurance_idx = min(int(len(all_distances) * endurance_ratio),
                           len(all_distances) - 1)
        inst.drone_endurance = all_distances[endurance_idx]
        # Ensure it's at least the max distance from depot
        max_depot_dist = max(
            inst.drone_distance(0, c.id) for c in customers
        )
        inst.drone_endurance = max(inst.drone_endurance, max_depot_dist * 1.5)

    return inst


def load_dumas_instance(filepath: str,
                        num_drones: int = 3,
                        wbli: float = 0.2,
                        wbui: float = 0.2) -> ProblemInstance:
    """
    Load a genuine Dumas TSPTW instance file and convert to Mo-CRPTW-mD.

    Dumas format:
    (index, x, y, ready_time, due_time, service_time)

    Args:
        filepath: Path to .txt file
        num_drones: Number of drones
        wbli/wbui: Flexible window parameters

    Returns:
        ProblemInstance
    """
    customers = []
    idx = 1

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                # Parse: idx x y a b [service]
                x = float(parts[1])
                y = float(parts[2])
                a = float(parts[3])
                b = float(parts[4])
                service = float(parts[5]) if len(parts) > 5 else 10.0

                width = b - a
                e = a - wbli * width
                l = b + wbui * width

                # 85% drone eligibility by default
                eligible = (idx % 7) != 0  # approx 85%

                customers.append(Customer(
                    id=idx, x=x, y=y,
                    a=a, b=b, e=e, l=l,
                    service_time=service,
                    drone_eligible=eligible
                ))
                idx += 1

    inst = ProblemInstance(
        customers=customers,
        num_drones=num_drones,
        truck_cost_per_km=25.0,
        drone_cost_per_km=1.0,
        drone_endurance=50.0,
        wbli=wbli,
        wbui=wbui
    )

    # Calculate drone endurance
    all_distances = []
    for i in range(1, len(customers) + 1):
        for j in range(1, len(customers) + 1):
            if i != j:
                all_distances.append(inst.drone_distance(i, j))

    if all_distances:
        all_distances.sort()
        idx = min(int(len(all_distances) * 0.35), len(all_distances) - 1)
        inst.drone_endurance = max(all_distances[idx], 20.0)

    return inst


def print_instance_info(inst: ProblemInstance):
    """Print summary of the problem instance"""
    print(f"Problem Instance: {inst.num_customers} customers, {inst.num_drones} drones")
    print(f"  Truck cost/km: {inst.truck_cost_per_km}")
    print(f"  Drone cost/km: {inst.drone_cost_per_km}")
    print(f"  Drone endurance: {inst.drone_endurance:.2f}")
    print(f"  Flexible window params: wbli={inst.wbli}, wbui={inst.wbui}")

    eligible = sum(1 for c in inst.customers if c.drone_eligible)
    print(f"  Drone-eligible: {eligible}/{inst.num_customers}")

    # Time window stats
    avg_width = sum(c.b - c.a for c in inst.customers) / inst.num_customers
    print(f"  Avg time window width: {avg_width:.2f}")
    print()
