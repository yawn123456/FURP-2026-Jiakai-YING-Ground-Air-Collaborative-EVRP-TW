#!/usr/bin/env python
"""
HMOA: Hybrid Multi-Objective Optimization Approach with Pareto Local Search
for Collaborative Truck-Drone Routing Problems Considering Flexible Time Windows.

Main entry point for running experiments.

Paper: Luo et al., IEEE Trans. on Intelligent Transportation Systems,
       VOL. 23, NO. 8, AUGUST 2022.

Usage:
    # Run on synthetic benchmark instances
    python main.py --mode synthetic --customers 20 --width 80 --runs 5

    # Run on Dumas TSPTW instances
    python main.py --mode dumas --data_dir ./dumas_instances/

    # Run with single instance test
    python main.py --mode test --customers 20 --width 80
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import List, Dict

import config
from problem import Solution, ProblemInstance
from algorithm import hmoa
from metrics import (
    compute_hypervolume, compute_c_metric,
    get_reference_point, find_best_compromise_solution,
    find_extreme_solutions
)
from utils import (
    generate_synthetic_instance, generate_benchmark_instances,
    load_instance, save_instance, discover_dumas_instances,
    load_dumas_instance, plot_pareto_fronts, plot_compromise_analysis
)
from operators import assign_nodes


def run_single_instance(instance: ProblemInstance,
                        instance_name: str = "test",
                        runs: int = 1,
                        verbose: bool = True) -> Dict:
    """
    Run HMOA on a single problem instance.

    Returns:
        dict with results: Pareto fronts, metrics, timing
    """
    all_pfs = []
    all_times = []
    all_hv = []

    for run_idx in range(runs):
        seed = hash(f"{instance_name}_{run_idx}") % (2**31)
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Instance: {instance_name} | Run {run_idx + 1}/{runs}")
            print(f"  Customers: {instance.num_customers} | Drones: {instance.num_drones}")
            print(f"  Drone Endurance: {instance.drone_endurance:.2f} | LTL: {instance.ltl}")
            print(f"{'='*60}")

        start_time = time.time()
        pf, hv_hist, obj_hist = hmoa(
            instance,
            population_size=config.POPULATION_SIZE,
            max_iterations=config.MAX_ITERATIONS,
            random_seed=seed,
            verbose=verbose
        )
        elapsed = time.time() - start_time

        all_pfs.append(pf)
        all_times.append(elapsed)

        if verbose:
            print(f"\n  Run {run_idx + 1} completed in {elapsed:.1f}s")
            print(f"  PF size: {len(pf)}")
            if pf:
                costs = [s._cost for s in pf]
                sats = [s._satisfaction for s in pf]
                print(f"  Cost range: [{min(costs):.2f}, {max(costs):.2f}]")
                print(f"  Satisfaction range: [{min(sats):.4f}, {max(sats):.4f}]")

    # Compute HV for each run
    if runs > 1:
        ref_point = get_reference_point(all_pfs)
        for pf in all_pfs:
            hv = compute_hypervolume(pf, ref_point, normalize=True)
            all_hv.append(hv)
        if verbose:
            print(f"\n  Mean HV: {np.mean(all_hv):.4f} ± {np.std(all_hv):.4f}")
            print(f"  Mean Time: {np.mean(all_times):.1f}s ± {np.std(all_times):.1f}s")

    return {
        'instance_name': instance_name,
        'instance': instance,
        'pareto_fronts': all_pfs,
        'times': all_times,
        'hypervolumes': all_hv,
        'reference_point': ref_point if runs > 1 else None,
    }


def run_benchmark_experiment(sizes: List[int] = None,
                             widths: List[int] = None,
                             instances_per_size: int = None,
                             runs: int = None,
                             output_dir: str = None) -> Dict:
    """
    Run full benchmark experiment as described in Section V-B.

    Tests HMOA on all instance sizes and time window widths.
    """
    if sizes is None:
        sizes = config.BENCHMARK_SIZES
    if widths is None:
        widths = config.TIME_WINDOW_WIDTHS
    if instances_per_size is None:
        instances_per_size = config.INSTANCES_PER_SIZE
    if runs is None:
        runs = config.NUM_RUNS
    if output_dir is None:
        output_dir = config.OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # Generate instances
    print("\n" + "=" * 60)
    print("  Generating benchmark instances...")
    print("=" * 60)

    instances = generate_benchmark_instances(
        sizes=sizes,
        widths=widths,
        instances_per_size=instances_per_size,
        output_dir=os.path.join(output_dir, 'instances')
    )
    print(f"  Generated {len(instances)} instances")

    # Run experiments
    all_results = {}
    for name, instance in instances.items():
        print(f"\n{'#'*60}")
        print(f"#  Running HMOA on {name}")
        print(f"{'#'*60}")

        result = run_single_instance(
            instance, instance_name=name, runs=runs, verbose=True
        )
        all_results[name] = result

        # Plot Pareto front for this instance (best run)
        best_pf = max(result['pareto_fronts'],
                     key=lambda pf: compute_hypervolume(pf, result['reference_point'])
                     if result['reference_point'] else len(pf))

        plot_compromise_analysis(
            best_pf,
            title=f"Pareto Front: {name}",
            save_path=os.path.join(output_dir, 'plots', f"{name}_pareto.png")
        )

    # Summary statistics
    print_summary(all_results, output_dir)

    return all_results


def run_comparison_experiment(instance: ProblemInstance,
                              instance_name: str = "comparison",
                              runs: int = 15) -> Dict:
    """
    Run comparison between HMOA and HMOA-noLS (for ablation study).

    Section V-B.3 compares HMOA with HMOA without PLS component.
    """
    print(f"\n{'='*60}")
    print(f"  Comparison Experiment: {instance_name}")
    print(f"  HMOA vs HMOA-noLS (without PLS)")
    print(f"{'='*60}")

    # Save original KMAX
    original_kmax = config.KMAX

    # Run HMOA (with PLS)
    print("\n  Running HMOA (with PLS)...")
    results_hmoa = run_single_instance(instance, f"{instance_name}_HMOA", runs)

    # Run HMOA-noLS (disable PLS by setting KMAX=0)
    config.KMAX = 0
    print("\n  Running HMOA-noLS (without PLS)...")
    results_nols = run_single_instance(instance, f"{instance_name}_noLS", runs)
    config.KMAX = original_kmax

    # Compute C-metrics
    all_c_hmoa_vs_nols = []
    all_c_nols_vs_hmoa = []

    for pf_hmoa, pf_nols in zip(results_hmoa['pareto_fronts'], results_nols['pareto_fronts']):
        c1 = compute_c_metric(pf_hmoa, pf_nols)
        c2 = compute_c_metric(pf_nols, pf_hmoa)
        all_c_hmoa_vs_nols.append(c1)
        all_c_nols_vs_hmoa.append(c2)

    print(f"\n  C(HMOA, HMOA-noLS): {np.mean(all_c_hmoa_vs_nols):.4f} ± {np.std(all_c_hmoa_vs_nols):.4f}")
    print(f"  C(HMOA-noLS, HMOA): {np.mean(all_c_nols_vs_hmoa):.4f} ± {np.std(all_c_nols_vs_hmoa):.4f}")

    # Plot comparison
    plot_pareto_fronts(
        {'HMOA': results_hmoa['pareto_fronts'][np.argmax(results_hmoa['hypervolumes'])],
         'HMOA-noLS': results_nols['pareto_fronts'][np.argmax(results_nols['hypervolumes'])]},
        title=f"Pareto Front Comparison: {instance_name}",
        save_path=os.path.join(config.OUTPUT_DIR, 'plots', f"{instance_name}_comparison.png")
    )

    return {
        'hmoa': results_hmoa,
        'noLS': results_nols,
        'C_HMOA_vs_noLS': all_c_hmoa_vs_nols,
        'C_noLS_vs_HMOA': all_c_nols_vs_hmoa,
    }


def run_sensitivity_analysis(instance: ProblemInstance,
                             instance_name: str = "sensitivity",
                             runs: int = 5) -> Dict:
    """
    Run sensitivity analysis on algorithm parameters (Section V-C.5).

    Tests impact of: population size, crossover rate, mutation rate.
    """
    print(f"\n{'='*60}")
    print(f"  Sensitivity Analysis: {instance_name}")
    print(f"{'='*60}")

    results = {}

    # Test population sizes
    pop_sizes = [50, 100, 150, 200]
    print("\n  --- Population Size Analysis ---")
    for n in pop_sizes:
        original_pop = config.POPULATION_SIZE
        config.POPULATION_SIZE = n
        print(f"  Population size = {n}")
        r = run_single_instance(instance, f"{instance_name}_pop{n}", runs, verbose=False)
        results[f'pop_{n}'] = r
        config.POPULATION_SIZE = original_pop

    # Test crossover rates
    cr_values = [0.7, 0.8, 0.9, 1.0]
    print("\n  --- Crossover Rate Analysis ---")
    for cr in cr_values:
        original_cr = config.CROSSOVER_RATE
        config.CROSSOVER_RATE = cr
        print(f"  Crossover rate = {cr}")
        r = run_single_instance(instance, f"{instance_name}_cr{cr}", runs, verbose=False)
        results[f'cr_{cr}'] = r
        config.CROSSOVER_RATE = original_cr

    # Test mutation rates
    mr_values = [0.1, 0.2, 0.3, 0.4]
    print("\n  --- Mutation Rate Analysis ---")
    for mr in mr_values:
        original_mr = config.MUTATION_RATE
        config.MUTATION_RATE = mr
        print(f"  Mutation rate = {mr}")
        r = run_single_instance(instance, f"{instance_name}_mr{mr}", runs, verbose=False)
        results[f'mr_{mr}'] = r
        config.MUTATION_RATE = original_mr

    # Print summary
    print("\n  --- Sensitivity Analysis Results ---")
    for param, result in results.items():
        hvs = result['hypervolumes']
        if hvs:
            print(f"  {param}: HV = {np.mean(hvs):.4f} ± {np.std(hvs):.4f}")

    return results


def print_summary(all_results: Dict, output_dir: str):
    """Print and save summary statistics."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    lines = []
    lines.append("instance,n_customers,n_drones,mean_pf_size,mean_time,std_time,mean_hv,std_hv")
    for name, result in sorted(all_results.items()):
        inst = result['instance']
        mean_pf = np.mean([len(pf) for pf in result['pareto_fronts']])
        mean_time = np.mean(result['times'])
        std_time = np.std(result['times'])
        mean_hv = np.mean(result['hypervolumes']) if result['hypervolumes'] else 0
        std_hv = np.std(result['hypervolumes']) if result['hypervolumes'] else 0

        print(f"  {name:20s}: "
              f"PF={mean_pf:.0f}, "
              f"Time={mean_time:.1f}±{std_time:.1f}s, "
              f"HV={mean_hv:.4f}±{std_hv:.4f}")

        lines.append(f"{name},{inst.num_customers},{inst.num_drones},"
                    f"{mean_pf:.1f},{mean_time:.1f},{std_time:.1f},"
                    f"{mean_hv:.4f},{std_hv:.4f}")

    # Save CSV
    csv_path = os.path.join(output_dir, 'summary.csv')
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Summary saved to {csv_path}")


def print_instance_report(instance: ProblemInstance, name: str):
    """Print detailed instance report."""
    best_pf = None  # Will be set after run

    print(f"\n{'='*60}")
    print(f"  INSTANCE REPORT: {name}")
    print(f"{'='*60}")
    print(f"  Number of customers:         {instance.num_customers}")
    print(f"  Number of drones:            {instance.num_drones}")
    print(f"  Drone endurance (epsilon):   {instance.drone_endurance:.2f}")
    print(f"  Lower Truck Limit (LTL):     {instance.ltl}")
    print(f"  Depot location:              ({instance.depot_x:.1f}, {instance.depot_y:.1f})")
    print(f"  Truck cost per unit:         {instance.truck_cost_per_unit}")
    print(f"  Drone cost per unit:         {instance.drone_cost_per_unit}")
    print(f"  Truck speed:                 {instance.truck_speed}")
    print(f"  Drone speed:                 {instance.drone_speed}")

    # Statistics about time windows
    ready_times = [n.ready_time for n in instance.nodes.values()]
    due_times = [n.due_time for n in instance.nodes.values()]
    earliest_times = [n.earliest_time for n in instance.nodes.values()]
    latest_times = [n.latest_time for n in instance.nodes.values()]

    print(f"  Time windows:")
    print(f"    Ready time range:          [{min(ready_times):.1f}, {max(ready_times):.1f}]")
    print(f"    Due time range:            [{min(due_times):.1f}, {max(due_times):.1f}]")
    print(f"    Flexible earliest range:   [{min(earliest_times):.1f}, {max(earliest_times):.1f}]")
    print(f"    Flexible latest range:     [{min(latest_times):.1f}, {max(latest_times):.1f}]")

    drone_eligible = sum(1 for n in instance.nodes.values() if n.is_drone_eligible)
    print(f"  Drone-eligible nodes:        {drone_eligible}/{instance.num_customers} "
          f"({100*drone_eligible/instance.num_customers:.0f}%)")
    print(f"{'='*60}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='HMOA: Hybrid Multi-Objective Optimization for Truck-Drone Routing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'synthetic', 'dumas', 'comparison', 'sensitivity'],
                       help='Experiment mode (default: test)')

    # Instance parameters
    parser.add_argument('--customers', type=int, default=20,
                       help='Number of customers (default: 20)')
    parser.add_argument('--width', type=int, default=80,
                       help='Time window width (default: 80)')
    parser.add_argument('--drones', type=int, default=2,
                       help='Number of drones (default: 2)')
    parser.add_argument('--data_dir', type=str, default='./dumas_instances/',
                       help='Directory containing Dumas instance files')

    # Algorithm parameters
    parser.add_argument('--population', type=int, default=None,
                       help='Population size (default: from config)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Max iterations/generations (default: from config)')
    parser.add_argument('--runs', type=int, default=None,
                       help='Number of independent runs (default: from config)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: from config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--no_plot', action='store_true',
                       help='Disable plotting')

    args = parser.parse_args()

    # Apply CLI overrides to config
    if args.population is not None:
        config.POPULATION_SIZE = args.population
    if args.iterations is not None:
        config.MAX_ITERATIONS = args.iterations
    if args.runs is not None:
        config.NUM_RUNS = args.runs
    if args.output is not None:
        config.OUTPUT_DIR = args.output
    if args.seed is not None:
        config.RANDOM_SEED = args.seed

    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  HMOA - Hybrid Multi-Objective Optimization Approach")
    print("  Truck-Drone Collaborative Routing with Flexible Time Windows")
    print("  ============================================================")
    print(f"  Population size:   {config.POPULATION_SIZE}")
    print(f"  Max iterations:    {config.MAX_ITERATIONS}")
    print(f"  Crossover rate:    {config.CROSSOVER_RATE}")
    print(f"  Mutation rate:     {config.MUTATION_RATE}")
    print(f"  PLS kmax:          {config.KMAX}")
    print(f"  PLS heuristics:    {config.PLS_HEURISTICS}")
    print(f"  Number of runs:    {config.NUM_RUNS}")
    print("=" * 60)

    if args.mode == 'test':
        # Single test instance
        print(f"\n  Mode: TEST (single synthetic instance)")
        instance = generate_synthetic_instance(
            num_customers=args.customers,
            time_window_width=args.width,
            num_drones=args.drones,
            random_seed=config.RANDOM_SEED,
        )

        print_instance_report(instance, f"n{args.customers}w{args.width}")
        result = run_single_instance(instance, f"n{args.customers}w{args.width}",
                                     runs=config.NUM_RUNS)

        if not args.no_plot and result['pareto_fronts']:
            best_pf = result['pareto_fronts'][np.argmax(result['hypervolumes'])] if result['hypervolumes'] else result['pareto_fronts'][0]
            plot_compromise_analysis(
                best_pf,
                title=f"Pareto Front: n{args.customers}w{args.width}",
                save_path=os.path.join(output_dir, 'plots', f"n{args.customers}w{args.width}_pareto.png")
            )

    elif args.mode == 'synthetic':
        # Full benchmark on synthetic instances
        print(f"\n  Mode: SYNTHETIC BENCHMARK")
        sizes = config.BENCHMARK_SIZES
        widths = [args.width] if args.width else config.TIME_WINDOW_WIDTHS
        run_benchmark_experiment(
            sizes=sizes,
            widths=widths,
            instances_per_size=config.INSTANCES_PER_SIZE,
            runs=config.NUM_RUNS,
            output_dir=output_dir
        )

    elif args.mode == 'dumas':
        # Dumas TSPTW instances
        print(f"\n  Mode: DUMAS TSPTW INSTANCES")
        data_dir = args.data_dir

        if not os.path.isdir(data_dir):
            print(f"  ERROR: Data directory not found: {data_dir}")
            print(f"  Please download Dumas instances or use --mode synthetic")
            sys.exit(1)

        dumas_files = discover_dumas_instances(data_dir)

        if not dumas_files:
            print(f"  ERROR: No Dumas instance files found in {data_dir}")
            print(f"  Looking for patterns: n*w*.*, *.txt, *.dat")
            sys.exit(1)

        print(f"  Found {len(dumas_files)} Dumas instance files")

        for name, filepath in sorted(dumas_files.items()):
            try:
                instance = load_dumas_instance(
                    filepath,
                    num_drones=args.drones
                )
                print_instance_report(instance, name)
                run_single_instance(instance, name, runs=config.NUM_RUNS)
            except Exception as e:
                print(f"  ERROR loading {filepath}: {e}")
                continue

    elif args.mode == 'comparison':
        # Comparison: HMOA vs HMOA-noLS
        print(f"\n  Mode: COMPARISON (HMOA vs HMOA-noLS)")
        instance = generate_synthetic_instance(
            num_customers=args.customers,
            time_window_width=args.width,
            num_drones=args.drones,
            random_seed=config.RANDOM_SEED,
        )
        print_instance_report(instance, f"n{args.customers}w{args.width}")
        run_comparison_experiment(instance, f"n{args.customers}w{args.width}",
                                  runs=min(config.NUM_RUNS, 5))

    elif args.mode == 'sensitivity':
        # Sensitivity analysis
        print(f"\n  Mode: SENSITIVITY ANALYSIS")
        instance = generate_synthetic_instance(
            num_customers=args.customers,
            time_window_width=args.width,
            num_drones=args.drones,
            random_seed=config.RANDOM_SEED,
        )
        print_instance_report(instance, f"n{args.customers}w{args.width}")
        run_sensitivity_analysis(instance, f"n{args.customers}w{args.width}",
                                 runs=min(config.NUM_RUNS, 5))

    print("\n" + "=" * 60)
    print("  Experiment completed.")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
