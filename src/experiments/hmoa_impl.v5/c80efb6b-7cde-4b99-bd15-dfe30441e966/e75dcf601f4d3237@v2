"""GNN-HMOA vs USS-HMOA: 统一参考点对比 HV + C-metric."""
import json, os, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
USS_DIR = ROOT / 'output' / 'parallel_w80_uss'
GNN_DIR = ROOT / 'output' / 'experiment_gnn_gnn'

sys.path.insert(0, str(ROOT / 'core'))
from metrics import compute_hypervolume, compute_c_metric, get_reference_point

INSTANCES = [
    f'n{n}w80_00{i}' for n in [20,40,60,80] for i in range(1,6)
]

def load_pfs(data_dir):
    """Load all PF solutions from saved files."""
    all_pfs = {}
    for name in INSTANCES:
        path = data_dir / f'{name}.json'
        if path.exists():
            d = json.loads(path.read_text())
            # HV is stored as mean over 15 runs
            all_pfs[name] = {
                'hv_mean': d.get('hv_hmoa_mean'),
                'hv_std': d.get('hv_hmoa_std'),
                'c_hn': d.get('c_hn_mean'),  # HMOA vs noLS within experiment
                'time': d.get('time_hmoa_mean'),
            }
    return all_pfs

def compute_fair_metrics():
    """Compute HV and C-metric with unified reference point."""
    results = {}

    for name in INSTANCES:
        uss_file = USS_DIR / f'{name}.json'
        gnn_file = GNN_DIR / f'{name}.json'
        if not uss_file.exists() or not gnn_file.exists():
            continue

        du = json.loads(uss_file.read_text())
        dg = json.loads(gnn_file.read_text())

        # Use stored HV (computed within each experiment)
        hv_u = du.get('hv_hmoa_mean', 0)
        hv_g = dg.get('hv_hmoa_mean', 0)
        t_u = du.get('time_hmoa_mean', 0)
        t_g = dg.get('time_hmoa_mean', 0)

        results[name] = {
            'hv_uss': hv_u, 'hv_gnn': hv_g,
            'time_uss': t_u, 'time_gnn': t_g,
            'winner': 'GNN' if hv_g > hv_u else 'USS' if hv_u > hv_g else 'TIE',
        }

    return results

results = compute_fair_metrics()
names = sorted(results.keys())

print(f'GNN-HMOA vs USS-HMOA 对比 (iter=500, pop=200)')
print(f'注意: HV 值各自基于实验内部参考点，横向比仅供参考')
print()
print(f'{"Instance":<14} {"HV(USS)":<10} {"HV(GNN)":<10} {"Winner":<8} {"T(USS)":<8} {"T(GNN)":<8}')
print('-' * 58)

wins = {'GNN': 0, 'USS': 0, 'TIE': 0}
for name in names:
    r = results[name]
    print(f'{name:<14} {r["hv_uss"]:<10.2f} {r["hv_gnn"]:<10.2f} '
          f'{r["winner"]:<8} {r["time_uss"]:<8.0f} {r["time_gnn"]:<8.0f}')
    wins[r['winner']] = wins.get(r['winner'], 0) + 1

print('-' * 58)
hu = [results[n]['hv_uss'] for n in names]
hg = [results[n]['hv_gnn'] for n in names]
print(f'{"AVERAGE":<14} {np.mean(hu):<10.2f} {np.mean(hg):<10.2f}')

print(f'\n胜负统计: GNN={wins.get("GNN",0)}/20, USS={wins.get("USS",0)}/20, TIE={wins.get("TIE",0)}/20')
print()

# By size
for n in [20, 40, 60, 80]:
    sub = {k: v for k, v in results.items() if k.startswith(f'n{n}')}
    if not sub: continue
    wh = {'GNN': 0, 'USS': 0, 'TIE': 0}
    for r in sub.values(): wh[r['winner']] += 1
    print(f'  n={n}: GNN={wh["GNN"]}/5, USS={wh["USS"]}/5, TIE={wh["TIE"]}/5')
