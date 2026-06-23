"""
HMOA 50-customer benchmark test
"""
import sys, os, time, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import generate_random_instance
from hmoa import hmoa
from visualize import plot_pareto_front

random.seed(42)

print("=" * 60)
print("HMOA Test: 50 customers, 3 drones")
print("=" * 60)
sys.stdout.flush()

inst = generate_random_instance(
    num_customers=50, num_drones=3, area_size=100.0,
    tw_width=80.0, tw_horizon=480.0,
    wbli=0.2, wbui=0.2,
    endurance_ratio=0.35, seed=12345
)

print(f"Customers: {inst.num_customers}")
print(f"Drones: {inst.num_drones}")
print(f"Drone endurance: {inst.drone_endurance:.2f}")
print(f"Drone-eligible: {sum(1 for c in inst.customers if c.drone_eligible)}/{inst.num_customers}")
print(f"Pop size: 100, Max generations: 100")
sys.stdout.flush()

start = time.time()
pf = hmoa(inst, pop_size=100, max_iterations=100,
          crossover_rate=0.8, mutation_rate=0.3,
          restart_rate=0.3, pls_kmax=5, verbose=True)
elapsed = time.time() - start

print(f"\n{'='*60}")
print("RESULTS")
print('='*60)
print(f"CPU Time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
print(f"Pareto front size: {len(pf)}")
sys.stdout.flush()

pf.sort(key=lambda s: s.cost)
bc = min(pf, key=lambda s: s.cost)
bs = max(pf, key=lambda s: s.satisfaction)

print(f"\nBest Cost:        f1={bc.cost:>10.2f}, f2={bc.satisfaction:.4f}")
print(f"Best Satisfaction: f1={bs.cost:>10.2f}, f2={bs.satisfaction:.4f}")

min_c, max_s = pf[0].cost, pf[-1].satisfaction
best_comp = min(pf, key=lambda s: ((s.cost-min_c)/(max_s-min_c+1))**2 + ((max_s-s.satisfaction)/(max_s+1))**2)
print(f"Best Compromise:  f1={best_comp.cost:>10.2f}, f2={best_comp.satisfaction:.4f}")

print(f"\nPareto Front ({len(pf)} solutions):")
print(f"| # | Cost | Satisfaction |")
print(f"|---|------|-------------|")
for i, s in enumerate(pf):
    print(f"| {i+1} | {s.cost:.2f} | {s.satisfaction:.4f} |")

plot_pareto_front(pf, title=f'HMOA Pareto Front (n50, m=3, {len(pf)} solutions)',
                  save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pareto_front_n50.png'))
print(f"\nPlot saved to pareto_front_n50.png")
sys.stdout.flush()
