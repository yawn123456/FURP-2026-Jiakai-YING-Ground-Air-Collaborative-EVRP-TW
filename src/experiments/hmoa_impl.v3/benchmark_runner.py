"""
Dumas TSPTW Benchmark Runner — HMOA Paper Reproduction

Runs HMOA on Dumas-style TSPTW instances matching the paper's experimental setup:
- 20 instances: n20w80 (x5), n40w80 (x5), n60w80 (x5), n80w80 (x5)
- Each instance run 15 independent times (paper: "15 times independently")
- Reports HV (Hypervolume) and CPU time matching paper Table II format
- Parameters: pop=200, crossover=0.8, mutation=0.3, restart=0.3, pls_kmax=5

Paper: Luo et al., "Hybrid Multi-Objective Optimization Approach With Pareto
       Local Search for Collaborative Truck-Drone Routing Problems Considering
       Flexible Time Windows", IEEE TITS, 2022
"""

import sys, os, time, math, random, json
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import generate_random_instance, load_dumas_instance
from model import ProblemInstance, Solution
from hmoa import hmoa
from evaluate import evaluate


def generate_dumas_instance_file(filename: str, n: int, width: float, seed: int):
    """
    Generate a Dumas TSPTW-style instance file.

    Dumas instances are characterized by:
    - Random customer coordinates (x, y) in [0, 100]
    - Time windows [a, b] centered around a nearest-neighbor route
    - a = arrival - width/2, b = a + width
    - Index starting from 1

    The paper's benchmark set:
    - n20w80: 20 customers, time window width = 80
    - n40w80: 40 customers, time window width = 80
    - n60w80: 60 customers, time window width = 80
    - n80w80: 80 customers, time window width = 80
    """
    rng = random.Random(seed)

    # Generate customer coordinates (uniform in [0, 100])
    coords = []
    for i in range(1, n + 1):
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)
        coords.append((i, x, y))

    # Build nearest-neighbor route to create feasible time windows
    depot = (0, 0.0, 0.0)
    unvisited = list(coords)
    route = [depot]
    current = depot

    while unvisited:
        best = min(unvisited, key=lambda c:
                   math.sqrt((c[1] - current[1])**2 + (c[2] - current[2])**2))
        route.append(best)
        unvisited.remove(best)
        current = best

    # Compute arrival times along the route
    times = [0.0]
    for i in range(1, len(route)):
        dx = route[i][1] - route[i-1][1]
        dy = route[i][2] - route[i-1][2]
        dist = math.sqrt(dx*dx + dy*dy)
        times.append(times[-1] + dist)

    # Assign time windows centered around arrival times
    lines = []
    for idx in range(1, len(route)):
        node_id, x, y = route[idx]
        arrival = times[idx]
        # Ready time a_i: center around arrival, ensure non-negative
        a = max(0, arrival - width / 2)
        b = a + width
        service = 10.0  # default service time
        lines.append(
            f"{int(node_id)} {x:.4f} {y:.4f} {a:.2f} {b:.2f} {service:.1f}"
        )

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


def create_instance_from_dumas(filepath: str, num_drones: int = 3,
                                wbli: float = 0.2, wbui: float = 0.2,
                                drone_eligible_ratio: float = 0.85,
                                endurance_ratio: float = 0.35,
                                truck_cost_ratio: float = 25.0) -> ProblemInstance:
    """
    Create a Mo-CRPTW-mD instance from a Dumas TSPTW file.

    Per paper Section V-B:
    - Flexible time windows: ei = ai - wbli*(bi-ai), li = bi + wbui*(bi-ai)
    - 85% drone-eligible
    - 35% endurance ratio
    - Truck cost = 25 × drone cost
    - Manhattan distance for truck, Euclidean for drone
    """
    from model import Customer, ProblemInstance

    customers = []
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                x = float(parts[1])
                y = float(parts[2])
                a = float(parts[3])
                b = float(parts[4])
                service = float(parts[5]) if len(parts) > 5 else 10.0

                tw_width = b - a
                e = a - wbli * tw_width
                l = b + wbui * tw_width

                # Paper: 85% drone-eligible
                eligible = (idx % 7) != 0  # ~85.7% eligible

                customers.append(Customer(
                    id=idx, x=x, y=y,
                    a=a, b=b, e=e, l=l,
                    service_time=service,
                    drone_eligible=eligible
                ))

    inst = ProblemInstance(
        customers=customers,
        num_drones=num_drones,
        truck_cost_per_km=truck_cost_ratio,
        drone_cost_per_km=1.0,
        drone_endurance=float('inf'),
        wbli=wbli,
        wbui=wbui
    )

    # Set drone endurance via 35% feasible flights rule
    # Paper: "we use the same method introduced in [32] and assume that
    # 35% of feasible flights can be satisfied by drones"
    all_distances = []
    for i in range(1, len(customers) + 1):
        for j in range(1, len(customers) + 1):
            if i != j:
                d = inst.drone_distance(i, j)
                all_distances.append(d)

    if all_distances:
        all_distances.sort()
        endurance_idx = min(
            int(len(all_distances) * endurance_ratio),
            len(all_distances) - 1
        )
        inst.drone_endurance = all_distances[endurance_idx]
        # Ensure at least max depot distance * 1.5
        max_depot_dist = max(
            inst.drone_distance(0, c.id) for c in customers
        )
        inst.drone_endurance = max(inst.drone_endurance, max_depot_dist * 1.5)

    return inst


def compute_hypervolume(pareto_front: List[Solution],
                         ref_point: Tuple[float, float]) -> float:
    """
    Compute Hypervolume (HV) of a Pareto front.

    Paper Section V-A: "HV represents the volume among a solution set
    and a reference point. We select the reference point by the worst
    objective values from all solutions obtained by all algorithms in
    all runs. The larger the HV, the better its diversity and convergence."

    Uses a simple sweep-line algorithm for 2D.
    f1 (cost) is minimized, f2 (satisfaction) is maximized.
    ref_point: (worst_f1, worst_f2) — dominated by all solutions.
    """
    if not pareto_front:
        return 0.0

    # Sort by f1 ascending
    sorted_pf = sorted(pareto_front, key=lambda s: s.cost)

    hv = 0.0
    prev_f1 = ref_point[0]
    max_f2 = ref_point[1]

    for sol in sorted_pf:
        # Rectangle width = f1 - prev_f1, height = satisfaction - ref_f2
        width = sol.cost - prev_f1
        height = sol.satisfaction - ref_point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = sol.cost

    return hv


def compute_c_metric(A: List[Solution], B: List[Solution]) -> float:
    """
    Compute C-metric C(A, B): fraction of solutions in B that are
    dominated by or equal to at least one solution in A.

    Paper Section V-A: C(A,B) represents the percentage of solutions
    in B that are dominated by any solution in A.
    """
    if not B:
        return 0.0

    dominated_count = 0
    for b_sol in B:
        for a_sol in A:
            # a dominates b: a has lower cost AND higher satisfaction
            # (or equal in one and better in the other)
            if (a_sol.cost <= b_sol.cost and a_sol.satisfaction >= b_sol.satisfaction
                    and (a_sol.cost < b_sol.cost or a_sol.satisfaction > b_sol.satisfaction)):
                dominated_count += 1
                break

    return dominated_count / len(B)


def run_single_instance(inst: ProblemInstance,
                         pop_size: int = 200, max_iter: int = 200,
                         crossover_rate: float = 0.8,
                         mutation_rate: float = 0.3,
                         restart_rate: float = 0.3,
                         pls_kmax: int = 5,
                         verbose: bool = False) -> Tuple[List[Solution], float]:
    """Run HMOA on a single instance and return PF + CPU time."""
    start = time.time()
    pf = hmoa(
        inst=inst,
        pop_size=pop_size,
        max_iterations=max_iter,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        restart_rate=restart_rate,
        pls_kmax=pls_kmax,
        verbose=verbose
    )
    elapsed = time.time() - start
    return pf, elapsed


def run_benchmark():
    """
    Full benchmark experiment matching paper Section V-B.

    Parameters (from paper):
    - pop_size=200, crossover=0.8, mutation=0.3, restart=0.3, pls_kmax=5
    - 3 drones, wbli=wbui=0.2
    - 85% drone-eligible, 35% endurance ratio
    - 15 independent runs per instance
    - 20 instances total (5 per size × 4 sizes)
    """
    print("=" * 80)
    print("  HMOA — BENCHMARK EXPERIMENTS (Dumas TSPTW Instances)")
    print("  Luo et al., IEEE Trans. Intelligent Transportation Systems, 2022")
    print("=" * 80)

    # --- Configuration (exactly per paper) ---
    CONFIG = {
        'pop_size': 200,
        'max_iter': 200,
        'crossover_rate': 0.8,
        'mutation_rate': 0.3,
        'restart_rate': 0.3,
        'pls_kmax': 5,
        'num_drones': 3,
        'wbli': 0.2,
        'wbui': 0.2,
        'drone_eligible_ratio': 0.85,
        'endurance_ratio': 0.35,
        'truck_cost_ratio': 25.0,
        'num_runs': 15,
    }

    print(f"\nConfiguration (paper Section V-B-2):")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # --- Generate Dumas TSPTW instances ---
    instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'dumas_instances')
    os.makedirs(instance_dir, exist_ok=True)

    # Paper: 4 sizes × 5 instances each = 20 instances
    instance_specs = []
    for n_cust, width in [(20, 80), (40, 80), (60, 80), (80, 80)]:
        for inst_idx in range(1, 6):
            seed = 100 + (n_cust // 20) * 20 + inst_idx
            name = f"n{n_cust}w{width}_{inst_idx:03d}"
            filepath = os.path.join(instance_dir, f"{name}.txt")

            if not os.path.exists(filepath):
                generate_dumas_instance_file(filepath, n_cust, width, seed)
                print(f"  Generated: {name}")

            instance_specs.append((name, filepath, n_cust))

    print(f"\nTotal instances: {len(instance_specs)}")
    print(f"  n20w80 × 5, n40w80 × 5, n60w80 × 5, n80w80 × 5")
    print(f"  Runs per instance: {CONFIG['num_runs']}")
    print(f"  Total runs: {len(instance_specs) * CONFIG['num_runs']}")

    # --- Run experiments ---
    all_results = {}  # instance_name -> list of (pf, time, hv)

    # Global reference point for HV (will be collected across all runs)
    all_objectives = []

    print(f"\n{'='*80}")
    print("  RUNNING EXPERIMENTS")
    print(f"{'='*80}\n")

    for inst_idx, (name, filepath, n_cust) in enumerate(instance_specs):
        print(f"[{inst_idx+1}/{len(instance_specs)}] {name} "
              f"({n_cust} customers, {CONFIG['num_runs']} runs)...")
        sys.stdout.flush()

        inst = create_instance_from_dumas(
            filepath,
            num_drones=CONFIG['num_drones'],
            wbli=CONFIG['wbli'],
            wbui=CONFIG['wbui'],
            drone_eligible_ratio=CONFIG['drone_eligible_ratio'],
            endurance_ratio=CONFIG['endurance_ratio'],
            truck_cost_ratio=CONFIG['truck_cost_ratio']
        )

        instance_results = []
        for run in range(CONFIG['num_runs']):
            random.seed(42 + run)
            pf, elapsed = run_single_instance(
                inst,
                pop_size=CONFIG['pop_size'],
                max_iter=CONFIG['max_iter'],
                crossover_rate=CONFIG['crossover_rate'],
                mutation_rate=CONFIG['mutation_rate'],
                restart_rate=CONFIG['restart_rate'],
                pls_kmax=CONFIG['pls_kmax'],
                verbose=False
            )

            # Collect objectives for global reference point
            for sol in pf:
                all_objectives.append((sol.cost, sol.satisfaction))

            instance_results.append({
                'run': run + 1,
                'pf_size': len(pf),
                'cpu_time': elapsed,
                'pf': pf
            })

        all_results[name] = instance_results

        # Print intermediate summary
        avg_time = sum(r['cpu_time'] for r in instance_results) / len(instance_results)
        avg_pf_size = sum(r['pf_size'] for r in instance_results) / len(instance_results)
        print(f"  Avg PF size: {avg_pf_size:.1f}, Avg CPU: {avg_time:.1f}s")

    # --- Compute global reference point for HV ---
    # Paper: "reference point by the worst objective values from all solutions"
    worst_f1 = max(o[0] for o in all_objectives) if all_objectives else 100000
    worst_f2 = min(o[1] for o in all_objectives) if all_objectives else 0
    # Slightly beyond worst for proper HV computation
    ref_point = (worst_f1 * 1.1, worst_f2 * 0.9)

    # --- Compute HV for each run ---
    print(f"\n{'='*80}")
    print("  HYPERVOLUME RESULTS")
    print(f"{'='*80}")
    print(f"  Reference point: (f1={ref_point[0]:.2f}, f2={ref_point[1]:.4f})")
    print()

    summary_by_size = {}

    for name, results in all_results.items():
        n = int(name.split('w')[0][1:])
        size_key = f"n{n}w80"

        hvs = []
        for r in results:
            hv = compute_hypervolume(r['pf'], ref_point)
            r['hv'] = hv
            hvs.append(hv)

        avg_hv = sum(hvs) / len(hvs)
        std_hv = (sum((h - avg_hv)**2 for h in hvs) / len(hvs)) ** 0.5
        avg_time = sum(r['cpu_time'] for r in results) / len(results)
        avg_pf = sum(r['pf_size'] for r in results) / len(results)

        if size_key not in summary_by_size:
            summary_by_size[size_key] = []
        summary_by_size[size_key].append({
            'name': name,
            'avg_hv': avg_hv,
            'std_hv': std_hv,
            'avg_time': avg_time,
            'avg_pf': avg_pf,
            'hvs': hvs
        })

        print(f"  {name:20s}  HV={avg_hv:.4f}({std_hv:.4f})  "
              f"Time={avg_time:6.1f}s  PF={avg_pf:5.1f}")

    # --- Print paper-style summary table ---
    print(f"\n{'='*80}")
    print("  SUMMARY BY INSTANCE SIZE (paper Table II format)")
    print(f"{'='*80}")
    print(f"{'Size':<12} {'Inst':<6} {'Avg HV':<12} {'Std HV':<12} "
          f"{'Avg Time(s)':<14} {'Avg PF':<8}")
    print("-" * 70)

    for size_key in ['n20w80', 'n40w80', 'n60w80', 'n80w80']:
        items = summary_by_size[size_key]
        size_avg_hv = sum(i['avg_hv'] for i in items) / len(items)
        size_avg_time = sum(i['avg_time'] for i in items) / len(items)
        size_avg_pf = sum(i['avg_pf'] for i in items) / len(items)

        for item in items:
            print(f"{'':12} {item['name'][-3:]:<6} "
                  f"{item['avg_hv']:<12.4f} {item['std_hv']:<12.4f} "
                  f"{item['avg_time']:<14.1f} {item['avg_pf']:<8.1f}")

        print(f"{size_key:<12} {'AVG':<6} "
              f"{size_avg_hv:<12.4f} {'':12} "
              f"{size_avg_time:<14.1f} {size_avg_pf:<8.1f}")
        print()

    # --- Extreme solutions summary ---
    print(f"{'='*80}")
    print("  EXTREME SOLUTIONS (Best across all runs)")
    print(f"{'='*80}")
    print(f"{'Instance':<20} {'Best Cost':<15} {'Best Sat':<15} "
          f"{'Compromise Cost':<18} {'Compromise Sat':<15}")
    print("-" * 85)

    for name, results in all_results.items():
        all_pf_sols = []
        for r in results:
            all_pf_sols.extend(r['pf'])

        if not all_pf_sols:
            continue

        best_cost = min(all_pf_sols, key=lambda s: s.cost)
        best_sat = max(all_pf_sols, key=lambda s: s.satisfaction)
        min_c = best_cost.cost
        max_s = best_sat.satisfaction

        # Best compromise (closest to ideal point)
        best_comp = min(all_pf_sols, key=lambda s:
            ((s.cost - min_c) / (max_s - min_c + 1))**2 +
            ((max_s - s.satisfaction) / (max_s + 1))**2)

        print(f"{name:<20} {best_cost.cost:<15.2f} {best_sat.satisfaction:<15.4f} "
              f"{best_comp.cost:<18.2f} {best_comp.satisfaction:<15.4f}")

    # --- Save results ---
    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'benchmark_results.json')
    # Convert to serializable format
    serializable = {}
    for name, results in all_results.items():
        serializable[name] = []
        for r in results:
            serializable[name].append({
                'run': r['run'],
                'pf_size': r['pf_size'],
                'cpu_time': r['cpu_time'],
                'hv': r.get('hv', 0),
            })

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    run_benchmark()
