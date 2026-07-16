"""Single-instance worker — runs 15 HMOA + 15 noLS on one instance."""
import sys, os, time, json

# Fix 1: Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Fix 2: Limit OpenBLAS threads BEFORE numpy import
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

name, fpath = sys.argv[1], sys.argv[2]
iter_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
outdir = sys.argv[4] if len(sys.argv) > 4 else 'output/parallel_w80'

import config
config.POPULATION_SIZE = 200  # 论文: 主种群规模固定为200
if iter_override:
    config.MAX_ITERATIONS = iter_override
config.MAX_PF_SIZE = 500
config.KMAX = 5
config.CROSSOVER_RATE = 0.8
config.MUTATION_RATE = 0.3

from utils import load_dumas_instance
from algorithm import hmoa
from metrics import compute_hypervolume, compute_c_metric, get_reference_point

inst = load_dumas_instance(fpath, num_drones=config.DEFAULT_DRONE_COUNT)
NRUNS = 15

# HMOA
config.KMAX = 5
pf_h, t_h = [], []
for r in range(NRUNS):
    seed = hash(f'{name}_h_{r}') % (2**31)
    t0 = time.time()
    pf, _, _ = hmoa(inst, config.POPULATION_SIZE, config.MAX_ITERATIONS, seed, verbose=False)
    e = time.time() - t0
    pf_h.append(pf); t_h.append(e)
    c = [s._cost for s in pf if s._cost is not None]
    s = [s._satisfaction for s in pf if s._satisfaction is not None]
    if c:
        print(f'{name} HMOA {r+1}/{NRUNS}: PF={len(pf)}, cost=[{min(c):.0f},{max(c):.0f}], sat=[{min(s):.1f},{max(s):.1f}], {e:.0f}s')
    else:
        print(f'{name} HMOA {r+1}/{NRUNS}: PF={len(pf)} (empty), {e:.0f}s')

# noLS
config.KMAX = 0
pf_n, t_n = [], []
for r in range(NRUNS):
    seed = hash(f'{name}_n_{r}') % (2**31)
    t0 = time.time()
    pf, _, _ = hmoa(inst, config.POPULATION_SIZE, config.MAX_ITERATIONS, seed, verbose=False)
    e = time.time() - t0
    pf_n.append(pf); t_n.append(e)
    c = [s._cost for s in pf if s._cost is not None]
    s = [s._satisfaction for s in pf if s._satisfaction is not None]
    if c:
        print(f'{name} noLS {r+1}/{NRUNS}: PF={len(pf)}, cost=[{min(c):.0f},{max(c):.0f}], sat=[{min(s):.1f},{max(s):.1f}], {e:.0f}s')
    else:
        print(f'{name} noLS {r+1}/{NRUNS}: PF={len(pf)} (empty), {e:.0f}s')

# Metrics
all_sols = []
for pf in pf_h + pf_n: all_sols.extend(pf)
ref = get_reference_point([all_sols])
hv_h = [compute_hypervolume(pf, ref) for pf in pf_h if pf]
hv_n = [compute_hypervolume(pf, ref) for pf in pf_n if pf]
c_hn = [compute_c_metric(h, n) for h, n in zip(pf_h, pf_n) if h and n]
c_nh = [compute_c_metric(n, h) for h, n in zip(pf_h, pf_n) if h and n]

result = {
    'name': name,
    'n': inst.num_customers,
    'hv_hmoa_mean': float(np.mean(hv_h)), 'hv_hmoa_std': float(np.std(hv_h)),
    'hv_nols_mean': float(np.mean(hv_n)), 'hv_nols_std': float(np.std(hv_n)),
    'c_hn_mean': float(np.mean(c_hn)), 'c_hn_std': float(np.std(c_hn)),
    'c_nh_mean': float(np.mean(c_nh)), 'c_nh_std': float(np.std(c_nh)),
    'time_hmoa_mean': float(np.mean(t_h)), 'time_nols_mean': float(np.mean(t_n)),
}

os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/{name}.json', 'w') as f:
    json.dump(result, f, indent=2)

# Save best PFs
if hv_h:
    best = pf_h[int(np.argmax(hv_h))]
    json.dump([{'cost': float(s._cost), 'sat': float(s._satisfaction)} for s in best if s._cost],
              open(f'{outdir}/pfs_{name}_hmoa.json', 'w'), indent=2)
if hv_n:
    best = pf_n[int(np.argmax(hv_n))]
    json.dump([{'cost': float(s._cost), 'sat': float(s._satisfaction)} for s in best if s._cost],
              open(f'{outdir}/pfs_{name}_nols.json', 'w'), indent=2)

print(f'DONE {name}: C(H,N)={np.mean(c_hn):.4f}, C(N,H)={np.mean(c_nh):.4f}')
