#!/usr/bin/env python
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
"""
Dumas Benchmark Experiment — strictly following paper Section V-B.

Paper settings:
  - Population size = 100
  - Max iterations = 500
  - 15 independent runs per instance
  - 2 drones
  - 5 instances per size: n20w60, n40w80, n60w40, n80w80
  - Compare HMOA vs HMOA-noLS
  - Metrics: HV, C-metric, computation time

Runtime estimation:
  n=20: ~2 min/run  → 5×15×2 = 150 runs × 2 min = 5 hours
  n=40: ~15 min/run → 5×15×2 = 150 runs × 15 min = 37 hours
  n=60: ~60 min/run → impractical with full settings

Strategy: adaptive parameters for larger instances.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, '.')
import config
from utils import load_dumas_instance
from algorithm import hmoa
from metrics import (
    compute_hypervolume, compute_c_metric,
    get_reference_point, find_best_compromise_solution
)

# =============================================================================
# Experiment Configuration
# =============================================================================

# Instance selection: 5 per size, picking sets with complete data
INSTANCES = [
    # (label, filepath)
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

# Paper parameters
NUM_DRONES = 2
NUM_RUNS = 15       # 论文: 15 runs
POP_SIZE = 100      # 论文: population = 100
MAX_ITER = 500      # 论文: iter = 500
KMAX_PLS = 5        # 论文: kmax = 5

# Adaptive strategy for larger instances (to keep runtime manageable)
# For n >= 60: reduce iterations and PLS intensity
ADAPTIVE_CONFIG = {
    20: {'iter': 500, 'kmax': 5, 'pop': 100},
    40: {'iter': 500, 'kmax': 5, 'pop': 100},
    60: {'iter': 300, 'kmax': 3, 'pop': 80},
    80: {'iter': 200, 'kmax': 2, 'pop': 60},
}

OUTPUT_DIR = 'output/dumas_experiment'
RESUME_FILE = os.path.join(OUTPUT_DIR, 'progress.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'pfs'), exist_ok=True)

# =============================================================================
# Main Experiment
# =============================================================================

def run_instance_experiment(instance_name, filepath, n_size, adaptive_cfg):
    """Run HMOA and HMOA-noLS on one instance, NUM_RUNS times each."""
    print(f"\n{'='*70}")
    print(f"  INSTANCE: {instance_name} (n={n_size})")
    print(f"  Config: pop={adaptive_cfg['pop']}, iter={adaptive_cfg['iter']}, "
          f"kmax={adaptive_cfg['kmax']}, runs={NUM_RUNS}")
    print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    # Load instance
    instance = load_dumas_instance(filepath, num_drones=NUM_DRONES)
    print(f"  Customers={instance.num_customers}, LTL={instance.ltl}, "
          f"Endurance={instance.drone_endurance:.1f}")

    pop = adaptive_cfg['pop']
    max_iter = adaptive_cfg['iter']
    kmax = adaptive_cfg['kmax']

    pf_hmoa_runs = []
    pf_nols_runs = []
    times_hmoa = []
    times_nols = []

    # Save PF for best HV run
    best_hv_hmoa = -float('inf')
    best_pf_hmoa = None
    best_hv_nols = -float('inf')
    best_pf_nols = None

    # ----- HMOA with PLS -----
    config.POPULATION_SIZE = pop
    config.MAX_ITERATIONS = max_iter
    config.MAX_PF_SIZE = 300
    config.KMAX = kmax

    print(f"\n  --- HMOA (with PLS, kmax={kmax}) ---")
    for run in range(NUM_RUNS):
        seed = hash(f"{instance_name}_hmoa_{run}") % (2**31)
        sys.stdout.write(f"    Run {run+1:2d}/{NUM_RUNS}... ")
        sys.stdout.flush()
        t0 = time.time()

        try:
            pf, _, _ = hmoa(instance, pop, max_iter, seed, verbose=False)
            elapsed = time.time() - t0
            pf_hmoa_runs.append(pf)
            times_hmoa.append(elapsed)

            # Track best PF
            costs = [s._cost for s in pf if s._cost is not None]
            sats = [s._satisfaction for s in pf if s._satisfaction is not None]
            print(f"PF={len(pf)}, cost=[{min(costs):.0f},{max(costs):.0f}], "
                  f"sat=[{min(sats):.1f},{max(sats):.1f}], {elapsed:.0f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            elapsed = time.time() - t0
            pf_hmoa_runs.append([])
            times_hmoa.append(elapsed)

    # ----- HMOA-noLS -----
    config.KMAX = 0

    print(f"\n  --- HMOA-noLS (without PLS) ---")
    for run in range(NUM_RUNS):
        seed = hash(f"{instance_name}_nols_{run}") % (2**31)
        sys.stdout.write(f"    Run {run+1:2d}/{NUM_RUNS}... ")
        sys.stdout.flush()
        t0 = time.time()

        try:
            pf, _, _ = hmoa(instance, pop, max_iter, seed, verbose=False)
            elapsed = time.time() - t0
            pf_nols_runs.append(pf)
            times_nols.append(elapsed)

            costs = [s._cost for s in pf if s._cost is not None]
            sats = [s._satisfaction for s in pf if s._satisfaction is not None]
            print(f"PF={len(pf)}, cost=[{min(costs):.0f},{max(costs):.0f}], "
                  f"sat=[{min(sats):.1f},{max(sats):.1f}], {elapsed:.0f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            elapsed = time.time() - t0
            pf_nols_runs.append([])
            times_nols.append(elapsed)

    config.KMAX = kmax  # Restore

    # ----- Compute Metrics -----
    print(f"\n  --- Metrics ---")
    all_pfs = [pf for pf in pf_hmoa_runs + pf_nols_runs if pf]
    if not all_pfs:
        print("  No valid PFs to compare!")
        return None

    ref_point = get_reference_point([all_pfs])

    # HV
    hv_hmoa = []
    hv_nols = []
    for pf in pf_hmoa_runs:
        if pf:
            hv_hmoa.append(compute_hypervolume(pf, ref_point))
    for pf in pf_nols_runs:
        if pf:
            hv_nols.append(compute_hypervolume(pf, ref_point))

    # C-metric
    c_hn = []
    c_nh = []
    for pf_h, pf_n in zip(pf_hmoa_runs, pf_nols_runs):
        if pf_h and pf_n:
            c_hn.append(compute_c_metric(pf_h, pf_n))
            c_nh.append(compute_c_metric(pf_n, pf_h))

    result = {
        'instance': instance_name,
        'n_customers': instance.num_customers,
        'num_drones': NUM_DRONES,
        'config': adaptive_cfg,
        'num_runs': NUM_RUNS,
        'hv_hmoa_mean': float(np.mean(hv_hmoa)) if hv_hmoa else None,
        'hv_hmoa_std': float(np.std(hv_hmoa)) if hv_hmoa else None,
        'hv_nols_mean': float(np.mean(hv_nols)) if hv_nols else None,
        'hv_nols_std': float(np.std(hv_nols)) if hv_nols else None,
        'c_hmoa_vs_nols_mean': float(np.mean(c_hn)) if c_hn else None,
        'c_hmoa_vs_nols_std': float(np.std(c_hn)) if c_hn else None,
        'c_nols_vs_hmoa_mean': float(np.mean(c_nh)) if c_nh else None,
        'c_nols_vs_hmoa_std': float(np.std(c_nh)) if c_nh else None,
        'time_hmoa_mean': float(np.mean(times_hmoa)),
        'time_hmoa_std': float(np.std(times_hmoa)),
        'time_nols_mean': float(np.mean(times_nols)),
        'time_nols_std': float(np.std(times_nols)),
        'ref_point': (float(ref_point[0]), float(ref_point[1])),
        'timestamp': datetime.now().isoformat(),
    }

    # Print result
    print(f"    HV(HMOA):     {result['hv_hmoa_mean']:.4f} ± {result['hv_hmoa_std']:.4f}" if result['hv_hmoa_mean'] else "    HV(HMOA): N/A")
    print(f"    HV(noLS):     {result['hv_nols_mean']:.4f} ± {result['hv_nols_std']:.4f}" if result['hv_nols_mean'] else "    HV(noLS): N/A")
    print(f"    C(HMOA,noLS): {result['c_hmoa_vs_nols_mean']:.4f} ± {result['c_hmoa_vs_nols_std']:.4f}" if result['c_hmoa_vs_nols_mean'] else "    C(HMOA,noLS): N/A")
    print(f"    C(noLS,HMOA): {result['c_nols_vs_hmoa_mean']:.4f} ± {result['c_nols_vs_hmoa_std']:.4f}" if result['c_nols_vs_hmoa_mean'] else "    C(noLS,HMOA): N/A")
    print(f"    Time(HMOA):   {result['time_hmoa_mean']:.0f} ± {result['time_hmoa_std']:.0f}s")
    print(f"    Time(noLS):   {result['time_nols_mean']:.0f} ± {result['time_nols_std']:.0f}s")

    # Save best PFs
    if hv_hmoa:
        best_idx = np.argmax(hv_hmoa)
        pf_path = os.path.join(OUTPUT_DIR, 'pfs', f'{instance_name}_hmoa_best.json')
        save_pf(pf_hmoa_runs[best_idx], pf_path)
    if hv_nols:
        best_idx = np.argmax(hv_nols)
        pf_path = os.path.join(OUTPUT_DIR, 'pfs', f'{instance_name}_nols_best.json')
        save_pf(pf_nols_runs[best_idx], pf_path)

    return result


def save_pf(pf, filepath):
    """Save Pareto front to JSON."""
    data = []
    for sol in pf:
        if sol._cost is not None and sol._satisfaction is not None:
            data.append({
                'cost': float(sol._cost),
                'satisfaction': float(sol._satisfaction),
                'truck_route': sol.truck_route,
                'drone_deliveries': [(d.drone_id, d.launch_node, d.customer, d.retrieve_node)
                                     for d in sol.drone_deliveries]
            })
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_progress():
    """Load completed instances from progress file."""
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE) as f:
            return json.load(f)
    return {'completed': [], 'results': {}}


def save_progress(progress):
    """Save progress to file."""
    with open(RESUME_FILE, 'w') as f:
        json.dump(progress, f, indent=2, default=str)


def print_final_summary(all_results):
    """Print final summary table in paper format."""
    print("\n\n" + "=" * 100)
    print("  FINAL SUMMARY TABLE")
    print("  (Format follows paper Table II & III)")
    print("=" * 100)

    header = (f"{'Instance':<16} {'n':<5} {'HV(HMOA)':<20} {'HV(noLS)':<20} "
              f"{'C(H,N)':<16} {'C(N,H)':<16} {'T(HMOA)':<10} {'T(noLS)':<10}")
    print(header)
    print("-" * 100)

    for name, r in sorted(all_results.items()):
        hv_h = f"{r['hv_hmoa_mean']:.4f}±{r['hv_hmoa_std']:.3f}" if r['hv_hmoa_mean'] else "N/A"
        hv_n = f"{r['hv_nols_mean']:.4f}±{r['hv_nols_std']:.3f}" if r['hv_nols_mean'] else "N/A"
        c_hn = f"{r['c_hmoa_vs_nols_mean']:.4f}±{r['c_hmoa_vs_nols_std']:.3f}" if r['c_hmoa_vs_nols_mean'] else "N/A"
        c_nh = f"{r['c_nols_vs_hmoa_mean']:.4f}±{r['c_nols_vs_hmoa_std']:.3f}" if r['c_nols_vs_hmoa_mean'] else "N/A"
        th = f"{r['time_hmoa_mean']:.0f}s" if r['time_hmoa_mean'] else "N/A"
        tn = f"{r['time_nols_mean']:.0f}s" if r['time_nols_mean'] else "N/A"

        print(f"{name:<16} {r['n_customers']:<5} {hv_h:<20} {hv_n:<20} "
              f"{c_hn:<16} {c_nh:<16} {th:<10} {tn:<10}")

    print("=" * 100)

    # Summary by size
    print("\n--- Summary by Instance Size ---")
    by_size = {}
    for name, r in all_results.items():
        n = r['n_customers']
        if n not in by_size:
            by_size[n] = {'c_hn': [], 'c_nh': [], 'hv_h': [], 'hv_n': []}
        if r['c_hmoa_vs_nols_mean'] is not None:
            by_size[n]['c_hn'].append(r['c_hmoa_vs_nols_mean'])
            by_size[n]['c_nh'].append(r['c_nols_vs_hmoa_mean'])
        if r['hv_hmoa_mean'] is not None:
            by_size[n]['hv_h'].append(r['hv_hmoa_mean'])
            by_size[n]['hv_n'].append(r['hv_nols_mean'])

    for n in sorted(by_size.keys()):
        stats = by_size[n]
        if stats['c_hn']:
            print(f"  n={n}: "
                  f"C(H,N)={np.mean(stats['c_hn']):.4f}, "
                  f"C(N,H)={np.mean(stats['c_nh']):.4f}, "
                  f"HV(H)={np.mean(stats['hv_h']):.2f}, "
                  f"HV(N)={np.mean(stats['hv_n']):.2f}")


# =============================================================================
# Run Experiment
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  HMOA DUMAS BENCHMARK EXPERIMENT")
    print("  Paper: Section V-B")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"  Total instances: {len(INSTANCES)}")
    print(f"  Runs per instance: {NUM_RUNS}")
    print(f"  Drones: {NUM_DRONES}")
    print(f"  Adaptive config: {ADAPTIVE_CONFIG}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Load progress for resume capability
    progress = load_progress()
    completed = set(progress['completed'])
    all_results = progress['results']

    if completed:
        print(f"\n  Resuming from previous run. {len(completed)} instances already completed.")
        print(f"  Completed: {sorted(completed)}")

    # Run experiments
    for instance_name, filepath in INSTANCES:
        if instance_name in completed:
            print(f"\n  Skipping {instance_name} (already completed)")
            continue

        # Determine instance size
        n_size = int(instance_name.split('w')[0][1:])  # Extract N from "n20w60_001"
        cfg = ADAPTIVE_CONFIG.get(n_size, ADAPTIVE_CONFIG[20])

        result = run_instance_experiment(instance_name, filepath, n_size, cfg)

        if result:
            all_results[instance_name] = result
            completed.add(instance_name)
            progress['completed'] = list(completed)
            progress['results'] = all_results
            save_progress(progress)
            print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Saved progress: "
                  f"{len(completed)}/{len(INSTANCES)} instances done")

    # Final summary
    print_final_summary(all_results)

    # Save final results
    final_path = os.path.join(OUTPUT_DIR, 'final_results.json')
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFinal results saved to {final_path}")
    print(f"Experiment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
