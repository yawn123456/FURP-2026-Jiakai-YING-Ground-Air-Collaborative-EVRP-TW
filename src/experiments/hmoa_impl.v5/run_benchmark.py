#!/usr/bin/env python
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
"""
Run HMOA benchmark experiment strictly following the paper Section V-B.

Paper settings:
- 20 Dumas et al. (1995) TSPTW instances (n=20,40,60,80; w=20,40,60,80)
- 5 instances per size
- Population size = 100
- Max iterations = 500
- 15 independent runs per instance
- Metrics: Hypervolume (HV) and Coverage (C-metric)
- Comparison: HMOA vs HMOA-noLS
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, '.')
import config
from utils import load_dumas_instance, generate_synthetic_instance
from algorithm import hmoa
from metrics import (
    compute_hypervolume, compute_c_metric,
    get_reference_point, find_best_compromise_solution,
    find_extreme_solutions
)

os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)

# =============================================================================
# Configuration
# =============================================================================

config.POPULATION_SIZE = 100
config.MAX_ITERATIONS = 500
config.MAX_PF_SIZE = 200
config.NUM_RUNS = 15

# Select available Dumas instances
INSTANCE_LIST = [
    # (name, filepath)
    ('n20w60_001', 'dumas_instances/n20w60.001.txt'),
    ('n20w60_002', 'dumas_instances/n20w60.002.txt'),
    ('n20w60_003', 'dumas_instances/n20w60.003.txt'),
    ('n20w60_004', 'dumas_instances/n20w60.004.txt'),
    ('n20w60_005', 'dumas_instances/n20w60.005.txt'),
    ('n40w80_001', 'dumas_instances/n40w80.001.txt'),
    ('n40w80_002', 'dumas_instances/n40w80.002.txt'),
    ('n40w80_003', 'dumas_instances/n40w80.003.txt'),
    ('n40w80_004', 'dumas_instances/n40w80.004.txt'),
    ('n40w80_005', 'dumas_instances/n40w80.005.txt'),
    ('n60w40_001', 'dumas_instances/n60w40.001.txt'),
    ('n60w40_002', 'dumas_instances/n60w40.002.txt'),
    ('n60w40_003', 'dumas_instances/n60w40.003.txt'),
    ('n60w40_004', 'dumas_instances/n60w40.004.txt'),
    ('n60w40_005', 'dumas_instances/n60w40.005.txt'),
    ('n80w80_001', 'dumas_instances/n80w80.001.txt'),
    ('n80w80_002', 'dumas_instances/n80w80.002.txt'),
    ('n80w80_003', 'dumas_instances/n80w80.003.txt'),
    ('n80w80_004', 'dumas_instances/n80w80.004.txt'),
    ('n80w80_005', 'dumas_instances/n80w80.005.txt'),
]

# For quick test, uncomment the next line:
# INSTANCE_LIST = INSTANCE_LIST[:2]  # Test with first 2 instances only

NUM_RUNS = 15  # Paper: 15 runs
NUM_DRONES = 2

print("=" * 70)
print("  HMOA BENCHMARK EXPERIMENT")
print("  Following paper Section V-B")
print("=" * 70)
print(f"  Instances: {len(INSTANCE_LIST)}")
print(f"  Runs per instance: {NUM_RUNS}")
print(f"  Population size: {config.POPULATION_SIZE}")
print(f"  Max iterations: {config.MAX_ITERATIONS}")
print(f"  Max PF size: {config.MAX_PF_SIZE}")
print(f"  Drones: {NUM_DRONES}")
print("=" * 70)

all_results = {}

for instance_name, filepath in INSTANCE_LIST:
    print(f"\n{'#'*60}")
    print(f"#  Instance: {instance_name}")
    print(f"{'#'*60}")

    # Load instance
    try:
        instance = load_dumas_instance(filepath, num_drones=NUM_DRONES)
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        continue

    print(f"  Customers: {instance.num_customers}")
    print(f"  LTL: {instance.ltl}")
    print(f"  Drone endurance: {instance.drone_endurance:.1f}")

    # Run HMOA (with PLS)
    pf_hmoa_runs = []
    pf_nols_runs = []
    times_hmoa = []
    times_nols = []

    for run in range(NUM_RUNS):
        seed = hash(f"{instance_name}_hmoa_{run}") % (2**31)

        # HMOA (with PLS)
        print(f"  Run {run+1}/{NUM_RUNS} (HMOA)...", end=' ', flush=True)
        t0 = time.time()
        pf, _, _ = hmoa(instance, population_size=config.POPULATION_SIZE,
                        max_iterations=config.MAX_ITERATIONS,
                        random_seed=seed, verbose=False)
        elapsed = time.time() - t0
        pf_hmoa_runs.append(pf)
        times_hmoa.append(elapsed)
        print(f"PF={len(pf)}, {elapsed:.0f}s")

    # Run HMOA-noLS
    original_kmax = config.KMAX
    config.KMAX = 0  # Disable PLS

    for run in range(NUM_RUNS):
        seed = hash(f"{instance_name}_nols_{run}") % (2**31)

        print(f"  Run {run+1}/{NUM_RUNS} (HMOA-noLS)...", end=' ', flush=True)
        t0 = time.time()
        pf, _, _ = hmoa(instance, population_size=config.POPULATION_SIZE,
                        max_iterations=config.MAX_ITERATIONS,
                        random_seed=seed, verbose=False)
        elapsed = time.time() - t0
        pf_nols_runs.append(pf)
        times_nols.append(elapsed)
        print(f"PF={len(pf)}, {elapsed:.0f}s")

    config.KMAX = original_kmax

    # Compute metrics
    all_pfs = pf_hmoa_runs + pf_nols_runs
    ref_point = get_reference_point(all_pfs)

    hv_hmoa = [compute_hypervolume(pf, ref_point) for pf in pf_hmoa_runs]
    hv_nols = [compute_hypervolume(pf, ref_point) for pf in pf_nols_runs]

    # C-metric (average over all run pairs)
    c_hmoa_vs_nols = []
    c_nols_vs_hmoa = []
    for pf_h, pf_n in zip(pf_hmoa_runs, pf_nols_runs):
        c_hmoa_vs_nols.append(compute_c_metric(pf_h, pf_n))
        c_nols_vs_hmoa.append(compute_c_metric(pf_n, pf_h))

    # Store results
    result = {
        'instance': instance_name,
        'num_customers': instance.num_customers,
        'num_drones': NUM_DRONES,
        'hv_hmoa_mean': np.mean(hv_hmoa),
        'hv_hmoa_std': np.std(hv_hmoa),
        'hv_nols_mean': np.mean(hv_nols),
        'hv_nols_std': np.std(hv_nols),
        'c_hmoa_vs_nols_mean': np.mean(c_hmoa_vs_nols),
        'c_hmoa_vs_nols_std': np.std(c_hmoa_vs_nols),
        'c_nols_vs_hmoa_mean': np.mean(c_nols_vs_hmoa),
        'c_nols_vs_hmoa_std': np.std(c_nols_vs_hmoa),
        'time_hmoa_mean': np.mean(times_hmoa),
        'time_nols_mean': np.mean(times_nols),
    }

    all_results[instance_name] = result

    print(f"\n  --- Results for {instance_name} ---")
    print(f"  HV(HMOA):     {result['hv_hmoa_mean']:.4f} ± {result['hv_hmoa_std']:.4f}")
    print(f"  HV(noLS):     {result['hv_nols_mean']:.4f} ± {result['hv_nols_std']:.4f}")
    print(f"  C(HMOA,noLS): {result['c_hmoa_vs_nols_mean']:.4f} ± {result['c_hmoa_vs_nols_std']:.4f}")
    print(f"  C(noLS,HMOA): {result['c_nols_vs_hmoa_mean']:.4f} ± {result['c_nols_vs_hmoa_std']:.4f}")
    print(f"  Time(HMOA):   {result['time_hmoa_mean']:.1f}s")
    print(f"  Time(noLS):   {result['time_nols_mean']:.1f}s")

# Save results
output_path = 'output/benchmark_results.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# Print summary table
print("\n" + "=" * 80)
print("  SUMMARY TABLE (format similar to paper Table II)")
print("=" * 80)
print(f"{'Instance':<15} {'HV(HMOA)':<18} {'HV(noLS)':<18} {'C(HMOA,noLS)':<16} {'C(noLS,HMOA)':<16} {'Time(HMOA)':<12}")
print("-" * 80)
for name, r in sorted(all_results.items()):
    print(f"{name:<15} {r['hv_hmoa_mean']:.4f}±{r['hv_hmoa_std']:.3f}   "
          f"{r['hv_nols_mean']:.4f}±{r['hv_nols_std']:.3f}   "
          f"{r['c_hmoa_vs_nols_mean']:.4f}±{r['c_hmoa_vs_nols_std']:.3f}   "
          f"{r['c_nols_vs_hmoa_mean']:.4f}±{r['c_nols_vs_hmoa_std']:.3f}   "
          f"{r['time_hmoa_mean']:.0f}s")
print("=" * 80)
