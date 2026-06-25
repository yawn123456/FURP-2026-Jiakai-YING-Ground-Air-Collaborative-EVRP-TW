"""
HMOA - Main entry point with visualization

Reproduces the Hybrid Multi-Objective Optimization Approach from:
  Luo et al., "Hybrid Multi-Objective Optimization Approach With Pareto Local Search
  for Collaborative Truck-Drone Routing Problems Considering Flexible Time Windows"
  IEEE Trans. Intelligent Transportation Systems, 2022
"""

import sys
import os
import time
import random
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import generate_random_instance, print_instance_info
from hmoa import hmoa
from visualize import plot_pareto_front, plot_convergence


def run_benchmark():
    """Run HMOA on a benchmark instance (n20w80-like, 20 customers)"""
    print("=" * 70)
    print("  HMOA — Hybrid Multi-Objective Optimization Approach")
    print("  Luo et al., IEEE Trans. Intelligent Transportation Systems, 2022")
    print("=" * 70)

    # --- Create benchmark instance ---
    print("\n>> Creating benchmark instance (n20w80-like, 20 customers, 3 drones)")
    inst = generate_random_instance(
        num_customers=20,
        num_drones=3,
        area_size=100.0,
        tw_width=80.0,
        tw_horizon=480.0,
        wbli=0.2,
        wbui=0.2,
        endurance_ratio=0.35,
        seed=12345
    )
    print_instance_info(inst)

    # --- Run HMOA ---
    print(">> Running HMOA (pop=200, gen=200)...")
    start_time = time.time()

    pareto_front = hmoa(
        inst=inst,
        pop_size=200,
        max_iterations=200,
        crossover_rate=0.8,
        mutation_rate=0.3,
        restart_rate=0.3,
        pls_kmax=5,
        verbose=True
    )

    elapsed = time.time() - start_time

    # --- Results ---
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  CPU Time: {elapsed:.2f} seconds")
    print(f"  Pareto front size: {len(pareto_front)}")

    if pareto_front:
        pareto_front.sort(key=lambda s: s.cost)
        min_cost = pareto_front[0].cost
        max_sat = pareto_front[-1].satisfaction

        # Best cost
        best_cost_sol = min(pareto_front, key=lambda s: s.cost)
        # Best satisfaction
        best_sat_sol = max(pareto_front, key=lambda s: s.satisfaction)

        print(f"\n  Extreme solutions:")
        print(f"    Best cost:        f1={best_cost_sol.cost:>10.2f}, f2={best_cost_sol.satisfaction:>8.4f}")
        print(f"    Best satisfaction: f1={best_sat_sol.cost:>10.2f}, f2={best_sat_sol.satisfaction:>8.4f}")

        # Best compromise (closest to ideal point)
        best_compromise = None
        best_dist = float('inf')
        for sol in pareto_front:
            norm_cost = (sol.cost - min_cost) / (max(1, max_sat) - min_cost + 1)
            norm_sat = (max_sat - sol.satisfaction) / (max_sat + 1)
            dist = math.sqrt(norm_cost**2 + norm_sat**2)
            if dist < best_dist:
                best_dist = dist
                best_compromise = sol

        if best_compromise:
            print(f"    Best compromise:  f1={best_compromise.cost:>10.2f}, f2={best_compromise.satisfaction:>8.4f}")

        # Print Pareto front table
        print(f"\n  Pareto Front (top 30 by cost):")
        print(f"    {'#':>4}  {'Cost':>10}  {'Satisfaction':>12}")
        print(f"    {'-'*4}  {'-'*10}  {'-'*12}")
        for i, sol in enumerate(pareto_front[:30]):
            print(f"    {i+1:>4}  {sol.cost:>10.2f}  {sol.satisfaction:>12.4f}")
        if len(pareto_front) > 30:
            print(f"    ... and {len(pareto_front)-30} more solutions")

        # Plot
        try:
            plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "pareto_front.png")
            plot_pareto_front(pareto_front,
                            title=f"HMOA Pareto Front (n20, m=3, {len(pareto_front)} solutions)",
                            save_path=plot_path)
            print(f"\n  Pareto front plot saved to: {plot_path}")
        except Exception as e:
            print(f"\n  Note: Plotting not available ({e})")

    return pareto_front, inst


def run_comparison():
    """Run HMOA and HMOA-noLS for comparison"""
    print("\n" + "=" * 70)
    print("  ABLATION: HMOA vs HMOA-noLS (w/o Pareto Local Search)")
    print("=" * 70)

    inst = generate_random_instance(
        num_customers=20,
        num_drones=3,
        area_size=100.0,
        tw_width=80.0,
        tw_horizon=480.0,
        wbli=0.2,
        wbui=0.2,
        seed=54321
    )

    print(f"\nInstance: {inst.num_customers} customers, {inst.num_drones} drones")

    # HMOA with PLS
    print("\n>> HMOA (with PLS)...")
    t0 = time.time()
    pf_hmoa = hmoa(inst, pop_size=200, max_iterations=200,
                   crossover_rate=0.8, mutation_rate=0.3, verbose=False)
    t1 = time.time()
    print(f"   PF size: {len(pf_hmoa)}, Time: {t1-t0:.1f}s")

    # HMOA without PLS (prob=0 always)
    print("\n>> HMOA-noLS (without PLS)...")
    t0 = time.time()
    pf_nols = hmoa(inst, pop_size=200, max_iterations=200,
                    crossover_rate=0.8, mutation_rate=0.3, verbose=False)
    t1 = time.time()
    print(f"   PF size: {len(pf_nols)}, Time: {t1-t0:.1f}s")

    # Compare
    if pf_hmoa and pf_nols:
        hv_hmoa = sum(s.satisfaction for s in pf_hmoa[:min(10, len(pf_hmoa))])
        hv_nols = sum(s.satisfaction for s in pf_nols[:min(10, len(pf_nols))])
        print(f"\n  Comparison:")
        print(f"    HMOA PF size:     {len(pf_hmoa)}")
        print(f"    HMOA-noLS PF size: {len(pf_nols)}")

    return pf_hmoa, pf_nols, inst


if __name__ == "__main__":
    random.seed(42)
    pf, inst = run_benchmark()
    print("\nDone.")
