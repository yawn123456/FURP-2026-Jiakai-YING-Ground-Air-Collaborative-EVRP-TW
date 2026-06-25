"""
Quick test script — validate the algorithm with small instance
"""

import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import generate_random_instance, print_instance_info
from hmoa import hmoa

random.seed(42)

# Small test: 10 customers, 2 drones (paper parameters: pop=200, gen=200, crossover=0.8, mutation=0.3)
inst = generate_random_instance(
    num_customers=10, num_drones=2, area_size=50.0,
    tw_width=60.0, tw_horizon=300.0, wbli=0.2, wbui=0.2,
    seed=42
)
print_instance_info(inst)

start = time.time()
pf = hmoa(inst, pop_size=200, max_iterations=200,
          crossover_rate=0.8, mutation_rate=0.3,
          restart_rate=0.3, pls_kmax=5, verbose=True)
elapsed = time.time() - start

print(f"\nResults: {len(pf)} solutions in {elapsed:.1f}s")
pf.sort(key=lambda s: s.cost)
for i, s in enumerate(pf[:10]):
    print(f"  {i+1:>3}: cost={s.cost:.2f}, satisfaction={s.satisfaction:.4f}")
