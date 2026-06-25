"""
Fast Dumas TSPTW Benchmark — 1 run per instance size
(Full paper parameters, reduced runs for quick validation)
"""
import sys, os, time, random, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_runner import (
    generate_dumas_instance_file, create_instance_from_dumas,
    run_single_instance, compute_hypervolume
)
from nsga2_utils import non_dominated_sort

CONFIG = {
    'pop_size': 200, 'max_iterations': 200,
    'crossover_rate': 0.8, 'mutation_rate': 0.3,
    'restart_rate': 0.3, 'pls_kmax': 5,
    'num_drones': 3, 'wbli': 0.2, 'wbui': 0.2,
    'drone_eligible_ratio': 0.85, 'endurance_ratio': 0.35,
    'truck_cost_ratio': 25.0,
}

INSTANCES = [
    ('n20w80_001', 20, 80, 201),
    ('n40w80_001', 40, 80, 301),
    ('n60w80_001', 60, 80, 401),
    ('n80w80_001', 80, 80, 501),
]

print("=" * 80)
print("  HMOA - DUMAS TSPTW BENCHMARK (Paper Parameters)")
print("  Luo et al., IEEE Trans. Intelligent Transportation Systems, 2022")
print("=" * 80)
print(f"  Config: pop={CONFIG['pop_size']}, gen={CONFIG['max_iterations']}, "
      f"crossover={CONFIG['crossover_rate']}, mutation={CONFIG['mutation_rate']}")
print(f"  restart_alpha={CONFIG['restart_rate']}, pls_kmax={CONFIG['pls_kmax']}")
print(f"  drones={CONFIG['num_drones']}, wbli=wbui={CONFIG['wbli']}, "
      f"drone_eligible={CONFIG['drone_eligible_ratio']}")
print()

instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'dumas_instances')
os.makedirs(instance_dir, exist_ok=True)

all_results = {}
all_objectives = []

for inst_idx, (name, n_cust, tw_width, seed) in enumerate(INSTANCES):
    filepath = os.path.join(instance_dir, f'{name}.txt')
    if not os.path.exists(filepath):
        generate_dumas_instance_file(filepath, n_cust, tw_width, seed)

    inst = create_instance_from_dumas(
        filepath, num_drones=CONFIG['num_drones'],
        wbli=CONFIG['wbli'], wbui=CONFIG['wbui'],
        drone_eligible_ratio=CONFIG['drone_eligible_ratio'],
        endurance_ratio=CONFIG['endurance_ratio'],
        truck_cost_ratio=CONFIG['truck_cost_ratio']
    )

    eligible = sum(1 for c in inst.customers if c.drone_eligible)
    print(f"[{inst_idx+1}/4] {name}: {n_cust} customers, "
          f"drone-eligible={eligible}/{n_cust}, endurance={inst.drone_endurance:.1f}")
    sys.stdout.flush()

    random.seed(42)
    start = time.time()
    pf, elapsed = run_single_instance(
        inst, pop_size=CONFIG['pop_size'], max_iter=CONFIG['max_iterations'],
        crossover_rate=CONFIG['crossover_rate'], mutation_rate=CONFIG['mutation_rate'],
        restart_rate=CONFIG['restart_rate'], pls_kmax=CONFIG['pls_kmax'],
        verbose=False
    )

    costs = [s.cost for s in pf]
    sats = [s.satisfaction for s in pf]
    for sol in pf:
        all_objectives.append((sol.cost, sol.satisfaction))

    all_results[name] = {
        'pf': pf, 'pf_size': len(pf), 'cpu_time': elapsed,
        'best_cost': min(costs) if costs else float('inf'),
        'best_sat': max(sats) if sats else 0.0,
    }

    print(f"  PF={len(pf)}, Time={elapsed:.1f}s, "
          f"BestCost={min(costs) if costs else 0:.1f}, "
          f"BestSat={max(sats) if sats else 0:.1f}")
    sys.stdout.flush()

# === RESULTS ===
worst_f1 = max(o[0] for o in all_objectives) if all_objectives else 100000
worst_f2 = min(o[1] for o in all_objectives) if all_objectives else 0
ref_point = (worst_f1 * 1.1, worst_f2 * 0.9)

print()
print("=" * 80)
print("  RESULTS vs PAPER (Table II)")
print("=" * 80)
print(f"{'Instance':<16} {'HV':<12} {'Time(s)':<12} {'PF':<8} "
      f"{'Best Cost':<14} {'Best Sat':<12}")
print("-" * 80)

for name in ['n20w80_001', 'n40w80_001', 'n60w80_001', 'n80w80_001']:
    if name not in all_results:
        continue
    r = all_results[name]
    hv = compute_hypervolume(r['pf'], ref_point)
    print(f"  {name:<16} {hv:<12.4f} {r['cpu_time']:<12.1f} {r['pf_size']:<8} "
          f"{r['best_cost']:<14.1f} {r['best_sat']:<12.4f}")

print()
print("  Paper n20w80: HV~0.8-0.9, Time~20-60s (Table II)")
print("  Paper n40w80: HV~0.7-0.9, Time~60-200s")
print("  Paper n60w80: HV~0.6-0.9, Time~120-400s")
print("  Paper n80w80: HV~0.6-0.8, Time~200-600s")
print()

# === PARETO FRONT DETAILS ===
print("=" * 80)
print("  PARETO FRONT SOLUTIONS")
print("=" * 80)

for name in ['n20w80_001', 'n40w80_001', 'n60w80_001', 'n80w80_001']:
    if name not in all_results:
        continue
    pf = all_results[name]['pf']
    pf.sort(key=lambda s: s.cost)
    min_c = pf[0].cost
    max_s = max(s.satisfaction for s in pf)
    best_comp = min(pf, key=lambda s:
        ((s.cost - min_c) / (max_s - min_c + 1))**2 +
        ((max_s - s.satisfaction) / (max_s + 1))**2)

    print(f"\n  {name} ({len(pf)} solutions):")
    print(f"    Best Cost:       f1={min_c:.2f}")
    print(f"    Best Sat:        f2={max_s:.4f}")
    print(f"    Best Compromise: f1={best_comp.cost:.2f}, f2={best_comp.satisfaction:.4f}")
    print(f"    PF (first 10):")
    for i, s in enumerate(pf[:10]):
        print(f"      {i+1:>2}. f1={s.cost:>10.2f}, f2={s.satisfaction:>8.4f}")

# Save results
results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'dumas_benchmark_fast.json')
serializable = {}
for name, r in all_results.items():
    serializable[name] = {
        'pf_size': r['pf_size'], 'cpu_time': r['cpu_time'],
        'best_cost': r['best_cost'], 'best_sat': r['best_sat'],
        'hv': compute_hypervolume(r['pf'], ref_point),
    }
with open(results_file, 'w') as f:
    json.dump(serializable, f, indent=2)

print(f"\n{'='*80}")
print(f"  Results saved: {results_file}")
print(f"{'='*80}")
