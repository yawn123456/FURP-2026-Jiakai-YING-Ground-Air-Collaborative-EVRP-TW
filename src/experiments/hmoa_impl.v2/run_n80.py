"""Run HMOA on n80w80 Dumas instance with paper parameters."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_runner import (
    generate_dumas_instance_file, create_instance_from_dumas, run_single_instance
)
from model import Solution

CONFIG = {
    'pop_size': 200, 'max_iterations': 200,
    'crossover_rate': 0.8, 'mutation_rate': 0.3,
    'restart_rate': 0.3, 'pls_kmax': 5,
    'num_drones': 3, 'wbli': 0.2, 'wbui': 0.2,
    'drone_eligible_ratio': 0.85, 'endurance_ratio': 0.35,
    'truck_cost_ratio': 25.0,
}

instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dumas_instances')
os.makedirs(instance_dir, exist_ok=True)

name = 'n80w80_001'
filepath = os.path.join(instance_dir, f'{name}.txt')
generate_dumas_instance_file(filepath, 80, 80, 501)

inst = create_instance_from_dumas(
    filepath, num_drones=CONFIG['num_drones'],
    wbli=CONFIG['wbli'], wbui=CONFIG['wbui'],
    drone_eligible_ratio=CONFIG['drone_eligible_ratio'],
    endurance_ratio=CONFIG['endurance_ratio'],
    truck_cost_ratio=CONFIG['truck_cost_ratio']
)

eligible = sum(1 for c in inst.customers if c.drone_eligible)
print(f"n80w80_001: {inst.num_customers} customers, drone-eligible={eligible}/80")
print(f"Drone endurance: {inst.drone_endurance:.1f}")
print(f"Config: pop=200, gen=200, crossover=0.8, mutation=0.3, "
      f"restart=0.3, PLS kmax=5")
print(f"Starting HMOA... (this may take 30-60 minutes)")
sys.stdout.flush()

random.seed(42)
start = time.time()
pf, elapsed = run_single_instance(
    inst, pop_size=CONFIG['pop_size'], max_iter=CONFIG['max_iterations'],
    crossover_rate=CONFIG['crossover_rate'], mutation_rate=CONFIG['mutation_rate'],
    restart_rate=CONFIG['restart_rate'], pls_kmax=CONFIG['pls_kmax'],
    verbose=True
)

costs = [s.cost for s in pf]
sats = [s.satisfaction for s in pf]

print()
print("=" * 70)
print("  n80w80_001 RESULTS")
print("=" * 70)
print(f"  CPU Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  Pareto Front size: {len(pf)}")
print(f"  Best Cost:        f1={min(costs) if costs else 0:.2f}")
print(f"  Best Satisfaction: f2={max(sats) if sats else 0:.4f}")
print()
print(f"  Pareto Front:")
print(f"  {'#':>3}  {'Cost':>10}  {'Satisfaction':>12}")
print(f"  {'-'*3}  {'-'*10}  {'-'*12}")
for i, s in enumerate(sorted(pf, key=lambda x: x.cost)[:20]):
    print(f"  {i+1:>3}  {s.cost:>10.2f}  {s.satisfaction:>12.4f}")
if len(pf) > 20:
    print(f"  ... and {len(pf)-20} more solutions")
