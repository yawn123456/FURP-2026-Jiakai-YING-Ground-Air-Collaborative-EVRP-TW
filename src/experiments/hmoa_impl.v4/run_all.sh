#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUTF8=1
export PATH="/d/anaconda/envs/vrp_env312:/d/anaconda/envs/vrp_env312/Scripts:$PATH"

OUTDIR="output/parallel_w80"
mkdir -p "$OUTDIR/pfs"

MAX_PROCS=6  # Limit parallel workers to avoid OOM

echo "=== HMOA Parallel Experiment ==="
echo "Instances: 20 (n=20,40,60,80 × 5 w80)"
echo "Workers: $MAX_PROCS"
echo "Started: $(date)"

# All instances
INSTANCES=(
  "n20w80_001 dumas_instances/n20w80.001.txt"
  "n20w80_002 dumas_instances/n20w80.002.txt"
  "n20w80_003 dumas_instances/n20w80.003.txt"
  "n20w80_004 dumas_instances/n20w80.004.txt"
  "n20w80_005 dumas_instances/n20w80.005.txt"
  "n40w80_001 dumas_instances/n40w80.001.txt"
  "n40w80_002 dumas_instances/n40w80.002.txt"
  "n40w80_003 dumas_instances/n40w80.003.txt"
  "n40w80_004 dumas_instances/n40w80.004.txt"
  "n40w80_005 dumas_instances/n40w80.005.txt"
  "n60w80_001 dumas_instances/n60w80.001.txt"
  "n60w80_002 dumas_instances/n60w80.002.txt"
  "n60w80_003 dumas_instances/n60w80.003.txt"
  "n60w80_004 dumas_instances/n60w80.004.txt"
  "n60w80_005 dumas_instances/n60w80.005.txt"
  "n80w80_001 dumas_instances/n80w80.001.txt"
  "n80w80_002 dumas_instances/n80w80.002.txt"
  "n80w80_003 dumas_instances/n80w80.003.txt"
  "n80w80_004 dumas_instances/n80w80.004.txt"
  "n80w80_005 dumas_instances/n80w80.005.txt"
)

# Launch workers with limited parallelism
running=0
for entry in "${INSTANCES[@]}"; do
  name=$(echo $entry | cut -d' ' -f1)
  fpath=$(echo $entry | cut -d' ' -f2)
  
  # Skip if already done
  if [ -f "$OUTDIR/${name}.json" ]; then
    echo "SKIP $name (already done)"
    continue
  fi
  
  echo "START $name"
  python -X utf8 worker.py "$name" "$fpath" > "$OUTDIR/${name}.log" 2>&1 &
  
  running=$((running + 1))
  if [ $running -ge $MAX_PROCS ]; then
    wait -n  # Wait for any child to finish
    running=$((running - 1))
  fi
done

# Wait for remaining
wait

echo "=== All workers done ==="
echo "Completed: $(date)"

# Merge results
python -X utf8 -c "
import json, os, numpy as np
outdir = '$OUTDIR'
results = {}
for f in sorted(os.listdir(outdir)):
    if f.endswith('.json') and not f.startswith('pfs_') and f != 'results.json':
        with open(os.path.join(outdir, f)) as fp:
            r = json.load(fp)
            results[r['name']] = r

# Summary
print()
print('='*90)
print('  FINAL RESULTS — HMOA vs HMOA-noLS (w=80, pop=100, iter=500, kmax=5)')
print('='*90)
print(f'{\"Instance\":<16} {\"C(H,N)\":<14} {\"C(N,H)\":<14} {\"W\":<6} {\"HV(H)\":<14} {\"HV(N)\":<14} {\"T(H)\":<8} {\"T(N)\":<8}')
print('-'*90)
for name, r in sorted(results.items()):
    w = 'HMOA' if r['c_hn_mean'] > r['c_nh_mean'] else ('noLS' if r['c_nh_mean'] > r['c_hn_mean'] else 'TIE')
    print(f'{name:<16} {r[\"c_hn_mean\"]:.4f}+/-{r[\"c_hn_std\"]:.3f} '
          f'{r[\"c_nh_mean\"]:.4f}+/-{r[\"c_nh_std\"]:.3f} {w:<6} '
          f'{r[\"hv_hmoa_mean\"]:.2f}       {r[\"hv_nols_mean\"]:.2f}       '
          f'{r[\"time_hmoa_mean\"]:.0f}s      {r[\"time_nols_mean\"]:.0f}s')

for n in [20, 40, 60, 80]:
    sub = {k:v for k,v in results.items() if v['n']==n}
    if sub:
        hn = np.mean([v['c_hn_mean'] for v in sub.values()])
        nh = np.mean([v['c_nh_mean'] for v in sub.values()])
        print(f'n={n}: C(H,N)={hn:.4f}, C(N,H)={nh:.4f}')

with open(os.path.join(outdir, 'results.json'), 'w') as fp:
    json.dump(results, fp, indent=2)
print(f'\nResults saved to {outdir}/results.json')
"
