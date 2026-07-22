"""Run GNN-HMOA on all 20 instances, 5 at a time."""
import subprocess, sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'experiment_gnn_gnn'
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / 'pfs').mkdir(exist_ok=True)

WORKER = ROOT / 'core' / 'worker.py'
PYTHON = sys.executable

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

BATCH_SIZE = 20  # 全部一起跑
MAX_PARALLEL = 20
t_start = time.time()
done, fail = 0, 0

print(f'GNN-HMOA experiment — {len(INSTANCES)} instances, {MAX_PARALLEL} parallel')
print(f'Output: {OUTDIR}')
print()

for batch_start in range(0, len(INSTANCES), BATCH_SIZE):
    batch = INSTANCES[batch_start:batch_start + BATCH_SIZE]
    scale = batch[0][0].split('_')[0]
    print(f'[{time.strftime("%H:%M:%S")}] Batch: {scale} ({len(batch)} instances)')

    procs = []
    for name, fpath in batch:
        abs_fpath = str(ROOT / fpath)
        cmd = [PYTHON, '-u', str(WORKER), name, abs_fpath, '500', str(OUTDIR)]
        env = {**os.environ, 'SKIP_NOLS': '1', 'CUDA_VISIBLE_DEVICES': '-1'}
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        procs.append((name, p))

    for name, p in procs:
        p.wait()
        if p.returncode == 0:
            done += 1
            out = OUTDIR / f'{name}.json'
            if out.exists():
                import json
                d = json.loads(out.read_text())
                print(f'  [{time.strftime("%H:%M:%S")}] DONE {name}: '
                      f'HV={d["hv_hmoa_mean"]:.2f}, C(H,N)={d["c_hn_mean"]:.3f}, '
                      f'T={d["time_hmoa_mean"]:.0f}s')
        else:
            fail += 1
            print(f'  [{time.strftime("%H:%M:%S")}] FAIL {name} (code {p.returncode})')

elapsed = time.time() - t_start
print(f'\nDone: {done}/{len(INSTANCES)}, Failed: {fail}, Time: {elapsed:.0f}s ({elapsed/60:.1f}min)')
