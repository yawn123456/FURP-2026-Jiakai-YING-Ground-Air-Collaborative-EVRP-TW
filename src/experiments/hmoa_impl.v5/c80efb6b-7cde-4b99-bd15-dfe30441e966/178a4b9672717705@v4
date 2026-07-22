#!/usr/bin/env python
"""Parallel HMOA experiment on Dumas w80 instances — paper parameters."""
import sys, os, time, json, numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
config.POPULATION_SIZE = 100
config.MAX_ITERATIONS = 500
config.MAX_PF_SIZE = 10000
config.KMAX = 5
config.CROSSOVER_RATE = 0.8
config.MUTATION_RATE = 0.3

from utils import load_dumas_instance
from algorithm import hmoa
from metrics import compute_hypervolume, compute_c_metric, get_reference_point

INSTANCES = [
    ('n20w80_001', 'dumas_instances/n20w80.001.txt'),
    ('n20w80_002', 'dumas_instances/n20w80.002.txt'),
    ('n20w80_003', 'dumas_instances/n20w80.003.txt'),
    ('n20w80_004', 'dumas_instances/n20w80.004.txt'),
    ('n20w80_005', 'dumas_instances/n20w80.005.txt'),
    ('n40w80_001', 'dumas_instances/n40w80.001.txt'),
    ('n40w80_002', 'dumas_instances/n40w80.002.txt'),
    ('n40w80_003', 'dumas_instances/n40w80.003.txt'),
    ('n40w80_004', 'dumas_instances/n40w80.004.txt'),
    ('n40w80_005', 'dumas_instances/n40w80.005.txt'),
    ('n60w80_001', 'dumas_instances/n60w80.001.txt'),
    ('n60w80_002', 'dumas_instances/n60w80.002.txt'),
    ('n60w80_003', 'dumas_instances/n60w80.003.txt'),
    ('n60w80_004', 'dumas_instances/n60w80.004.txt'),
    ('n60w80_005', 'dumas_instances/n60w80.005.txt'),
    ('n80w80_001', 'dumas_instances/n80w80.001.txt'),
    ('n80w80_002', 'dumas_instances/n80w80.002.txt'),
    ('n80w80_003', 'dumas_instances/n80w80.003.txt'),
    ('n80w80_004', 'dumas_instances/n80w80.004.txt'),
    ('n80w80_005', 'dumas_instances/n80w80.005.txt'),
]

NRUNS = 15
OUTDIR = 'output/parallel_w80'
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f'{OUTDIR}/pfs', exist_ok=True)

def run_one(instance_name, run_idx, seed_offset=0):
    """Run HMOA once on a given instance. Returns (pf, time, instance_name, run_idx)."""
    seed = hash(f'{instance_name}_{run_idx}') % (2**31) + seed_offset
    t0 = time.time()
    pf, _, _ = hmoa(load_dumas_instance(INSTANCES_DICT[instance_name], num_drones=2),
                    100, 500, seed, verbose=False)
    elapsed = time.time() - t0
    return pf, elapsed, instance_name, run_idx

def run_batch(args):
    """Run 15 HMOA + 15 noLS for one instance. Returns (name, pf_hmoa, t_hmoa, pf_nols, t_nols)."""
    name, fpath = args
    inst = load_dumas_instance(fpath, num_drones=2)

    # HMOA with PLS
    config.KMAX = 5
    pf_h, t_h = [], []
    for r in range(NRUNS):
        seed = hash(f'{name}_h_{r}') % (2**31)
        t0 = time.time()
        pf, _, _ = hmoa(inst, 100, 500, seed, verbose=False)
        e = time.time() - t0
        pf_h.append(pf); t_h.append(e)
        if pf:
            c = [s._cost for s in pf if s._cost]; s = [s._satisfaction for s in pf if s._satisfaction]
            print(f'  [{datetime.now():%H:%M:%S}] {name} HMOA run{r+1:2d}: PF={len(pf)}, cost=[{min(c):.0f},{max(c):.0f}], sat=[{min(s):.1f},{max(s):.1f}], {e:.0f}s')

    # HMOA-noLS
    config.KMAX = 0
    pf_n, t_n = [], []
    for r in range(NRUNS):
        seed = hash(f'{name}_n_{r}') % (2**31)
        t0 = time.time()
        pf, _, _ = hmoa(inst, 100, 500, seed, verbose=False)
        e = time.time() - t0
        pf_n.append(pf); t_n.append(e)
        if pf:
            c = [s._cost for s in pf if s._cost]; s = [s._satisfaction for s in pf if s._satisfaction]
            print(f'  [{datetime.now():%H:%M:%S}] {name} noLS run{r+1:2d}: PF={len(pf)}, cost=[{min(c):.0f},{max(c):.0f}], sat=[{min(s):.1f},{max(s):.1f}], {e:.0f}s')

    return name, pf_h, t_h, pf_n, t_n

if __name__ == '__main__':
    # Build instance dict
    INSTANCES_DICT = dict(INSTANCES)
    NAMES = [n for n, _ in INSTANCES]

    # Build work batches: one per instance
    batches = [(name, fpath) for name, fpath in INSTANCES]

    n_workers = min(cpu_count(), len(INSTANCES))
    print('=' * 70)
    print(f'  HMOA vs HMOA-noLS PARALLEL EXPERIMENT — Paper Parameters')
    print(f'  Instances: {len(INSTANCES)} (n=20,40,60,80 × 5)')
    print(f'  Runs/instance: {NRUNS} × 2 (HMOA + noLS) = {len(INSTANCES) * NRUNS * 2} total')
    print(f'  Workers: {n_workers}')
    print(f'  Params: pop=100, iter=500, kmax=5, pc=0.8, pm=0.3')
    print(f'  Started: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print('=' * 70)

    t_start = time.time()
    all_results = {}

    with Pool(processes=n_workers) as pool:
        for name, pf_h, t_h, pf_n, t_n in pool.imap_unordered(run_batch, batches):
            # Compute reference point
            all_sols = []
            for pf in pf_h + pf_n:
                all_sols.extend(pf)
            ref = get_reference_point([all_sols]) if all_sols else (1, -1)

            # HV
            hv_h = [compute_hypervolume(pf, ref) for pf in pf_h if pf]
            hv_n = [compute_hypervolume(pf, ref) for pf in pf_n if pf]

            # C-metric (pairwise across runs)
            c_hn = [compute_c_metric(h, n) for h, n in zip(pf_h, pf_n) if h and n]
            c_nh = [compute_c_metric(n, h) for h, n in zip(pf_h, pf_n) if h and n]

            # Cost/sat ranges from HMOA
            costs_h = [s._cost for pf in pf_h for s in pf if s._cost]
            sats_h  = [s._satisfaction for pf in pf_h for s in pf if s._satisfaction]

            all_results[name] = {
                'n': int(name.split('w')[0][1:]),
                'pf_hmoa_sizes': [len(pf) for pf in pf_h],
                'pf_nols_sizes': [len(pf) for pf in pf_n],
                'hv_hmoa_mean': float(np.mean(hv_h)), 'hv_hmoa_std': float(np.std(hv_h)),
                'hv_nols_mean': float(np.mean(hv_n)), 'hv_nols_std': float(np.std(hv_n)),
                'c_hn_mean': float(np.mean(c_hn)), 'c_hn_std': float(np.std(c_hn)),
                'c_nh_mean': float(np.mean(c_nh)), 'c_nh_std': float(np.std(c_nh)),
                'time_hmoa_mean': float(np.mean(t_h)), 'time_nols_mean': float(np.mean(t_n)),
                'time_total': float(np.sum(t_h) + np.sum(t_n)),
                'cost_min': float(min(costs_h)), 'cost_max': float(max(costs_h)),
                'sat_min': float(min(sats_h)), 'sat_max': float(max(sats_h)),
            }

            # Save best PFs
            if hv_h:
                best_h = pf_h[np.argmax(hv_h)]
                json.dump([{'cost': float(s._cost), 'sat': float(s._satisfaction)} for s in best_h if s._cost],
                          open(f'{OUTDIR}/pfs/{name}_hmoa_best.json', 'w'), indent=2)
            if hv_n:
                best_n = pf_n[np.argmax(hv_n)]
                json.dump([{'cost': float(s._cost), 'sat': float(s._satisfaction)} for s in best_n if s._cost],
                          open(f'{OUTDIR}/pfs/{name}_nols_best.json', 'w'), indent=2)

            # Save incremental results
            with open(f'{OUTDIR}/results.json', 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

            elapsed = time.time() - t_start
            n_done = len(all_results)
            winner = 'HMOA' if np.mean(c_hn) > np.mean(c_nh) else ('noLS' if np.mean(c_nh) > np.mean(c_hn) else 'TIE')
            print(f'  [{datetime.now():%H:%M:%S}] COMPLETED: {name} ({n_done}/{len(INSTANCES)}), '
                  f'C(H,N)={np.mean(c_hn):.3f}, C(N,H)={np.mean(c_nh):.3f} [{winner}], '
                  f'total: {elapsed:.0f}s')

    # Final summary
    t_total = time.time() - t_start
    print(f'\n{"="*90}')
    print(f'  FINAL RESULTS — HMOA vs HMOA-noLS (w=80, pop=100, iter=500, kmax=5)')
    print(f'  Total time: {t_total:.0f}s ({t_total/3600:.1f}h)')
    print(f'{"="*90}')
    print(f'{"Instance":<16} {"C(H,N)":<14} {"C(N,H)":<14} {"W":<6} {"HV(H)":<14} {"HV(N)":<14} {"T(H)":<8} {"T(N)":<8}')
    print('-'*90)
    for name, r in sorted(all_results.items()):
        w = 'HMOA' if r['c_hn_mean'] > r['c_nh_mean'] else ('noLS' if r['c_nh_mean'] > r['c_hn_mean'] else 'TIE')
        print(f'{name:<16} {r["c_hn_mean"]:.4f}+/-{r["c_hn_std"]:.3f} '
              f'{r["c_nh_mean"]:.4f}+/-{r["c_nh_std"]:.3f} {w:<6} '
              f'{r["hv_hmoa_mean"]:.2f}       {r["hv_nols_mean"]:.2f}       '
              f'{r["time_hmoa_mean"]:.0f}s      {r["time_nols_mean"]:.0f}s')

    # By size
    print(f'\n--- By Size (cf. paper Table II & III) ---')
    for n_size in [20, 40, 60, 80]:
        sub = {k: v for k, v in all_results.items() if v['n'] == n_size}
        if sub:
            c_hn = np.mean([v['c_hn_mean'] for v in sub.values()])
            c_nh = np.mean([v['c_nh_mean'] for v in sub.values()])
            hv_h = np.mean([v['hv_hmoa_mean'] for v in sub.values()])
            hv_n = np.mean([v['hv_nols_mean'] for v in sub.values()])
            t_h = np.mean([v['time_hmoa_mean'] for v in sub.values()])
            t_n = np.mean([v['time_nols_mean'] for v in sub.values()])
            print(f'  n={n_size}: C(H,N)={c_hn:.4f}, C(N,H)={c_nh:.4f}, '
                  f'HV(H)={hv_h:.1f}, HV(N)={hv_n:.1f}, T(H)={t_h:.0f}s, T(N)={t_n:.0f}s')

    print(f'\nResults saved to {OUTDIR}/results.json')
    print(f'Completed: {datetime.now():%Y-%m-%d %H:%M:%S}')
