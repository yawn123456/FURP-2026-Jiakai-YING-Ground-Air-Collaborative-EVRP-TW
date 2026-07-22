#!/usr/bin/env python
"""
并行 GNN 训练数据生成器。

使用 subprocess 并行运行 HMOA+USS 数据生成 worker，
每个 worker 处理一个实例（5 次 HMOA 运行），保存全部解的结构信息。
"""

import sys, os, time, json, traceback, subprocess, threading
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'core'))
os.chdir(ROOT)

import config

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

OUTDIR = os.path.join(ROOT, 'output', 'gnn_training_data')
WORKER_SCRIPT = os.path.join(ROOT, 'gnn_worker.py')
MAX_WORKERS = 20

os.makedirs(OUTDIR, exist_ok=True)


def run_worker(instance_name: str, fpath: str):
    """Launch gnn_worker.py as a subprocess."""
    cmd = [sys.executable, '-u', WORKER_SCRIPT, instance_name, fpath, OUTDIR]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, 'PYTHONUTF8': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        )
        output_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(f'  [{instance_name}] {line}', flush=True)
        proc.wait(timeout=10800)  # 3h timeout
        success = proc.returncode == 0
        return instance_name, success, '\n'.join(output_lines[-5:])
    except subprocess.TimeoutExpired:
        proc.kill()
        return instance_name, False, "TIMEOUT (3h)"
    except Exception as e:
        return instance_name, False, str(e)


def merge_dataset(outdir, instances):
    """Merge all individual instance .npz files into one dataset.npz."""
    import numpy as np

    all_x, all_y = [], []
    all_ptr = []
    all_names = []
    all_meta = []
    offset = 0
    total_graphs = 0

    for name, _ in instances:
        npz_path = os.path.join(outdir, f'{name}.npz')
        meta_path = os.path.join(outdir, f'{name}_meta.json')
        marker = os.path.join(outdir, f'{name}_done.marker')

        if not os.path.exists(npz_path):
            print(f'  [SKIP] {name} — .npz not found')
            continue
        if not os.path.exists(marker):
            print(f'  [SKIP] {name} — not completed')
            continue

        data = np.load(npz_path, allow_pickle=True)
        x_chunk = data['x']
        y_chunk = data['y']
        ptr_chunk = data['graph_ptr']

        n_graphs = len(ptr_chunk)
        n_nodes = len(x_chunk)

        # Adjust pointers: add offset
        ptr_chunk[:, 0] += offset
        offset += n_nodes

        all_x.append(x_chunk)
        all_y.append(y_chunk)
        all_ptr.append(ptr_chunk)
        all_names.extend([name] * n_graphs)
        total_graphs += n_graphs

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                all_meta.append(meta)

        print(f'  [OK]   {name}: {n_graphs} graphs, {n_nodes} nodes')

    if not all_x:
        print('  WARNING: no data to merge!')
        return

    x_all = np.concatenate(all_x, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    ptr_all = np.concatenate(all_ptr, axis=0)

    np.savez(os.path.join(outdir, 'dataset.npz'),
             x=x_all, y=y_all, graph_ptr=ptr_all,
             instance_names=np.array(all_names, dtype=object))

    # Summary metadata
    labeled = (y_all >= 0).sum()
    truck = (y_all == 0).sum()
    drone = (y_all == 1).sum()

    summary = {
        'total_graphs': int(total_graphs),
        'total_nodes': int(len(x_all)),
        'labeled_nodes': int(labeled),
        'truck_nodes': int(truck),
        'drone_nodes': int(drone),
        'unlabeled_nodes': int((y_all == -1).sum()),
        'instances': all_meta,
        'config': {
            'population_size': config.POPULATION_SIZE,
            'max_iterations': config.MAX_ITERATIONS,
            'kmax': config.KMAX,
            'zeta': config.ZETA,
        },
        'generated_at': datetime.now().isoformat(),
    }
    with open(os.path.join(outdir, 'metadata.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\n  合并完成: {total_graphs} 图, {len(x_all)} 节点')
    print(f'  标签: {truck} 卡车, {drone} 无人机, {labeled} 总计')
    print(f'  保存至: {outdir}/dataset.npz')


if __name__ == '__main__':
    # Skip already completed
    todo = []
    for name, fpath in INSTANCES:
        marker = os.path.join(OUTDIR, f'{name}_done.marker')
        if os.path.exists(marker):
            print(f'  SKIP {name} (already done)')
        else:
            todo.append((name, fpath))

    if not todo:
        print('所有实例已完成!')
        sys.exit(0)

    n_total = len(todo)
    print('=' * 70)
    print('  GNN 训练数据生成 (并行)')
    print(f'  Instances: {len(todo)} (of {len(INSTANCES)} total)')
    print(f'  Runs/instance: 5 × HMOA (pop=200, iter=500, USS on)')
    print(f'  Workers: {MAX_WORKERS}')
    print(f'  Output: {OUTDIR}')
    print(f'  Started: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print('=' * 70)

    t_start = time.time()
    done, fail = 0, 0
    active = {}  # proc -> (name, fpath)

    # Launch all workers at once (up to MAX_WORKERS)
    for name, fpath in todo[:MAX_WORKERS]:
        cmd = [sys.executable, '-u', WORKER_SCRIPT, name, fpath, OUTDIR]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, 'PYTHONUTF8': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        )
        active[proc] = (name, fpath)

    pending = todo[MAX_WORKERS:]  # remaining to start

    # Stream output from all workers
    while active:
        for proc in list(active.keys()):
            name, fpath = active[proc]
            # Read available lines (non-blocking-ish)
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                print(f'  [{name}] {line.rstrip()}', flush=True)

            ret = proc.poll()
            if ret is not None:
                # Process finished
                success = ret == 0
                if success:
                    done += 1
                else:
                    fail += 1
                elapsed = time.time() - t_start
                status = 'DONE' if success else 'FAIL'
                print(f'  [{datetime.now():%H:%M:%S}] {status} {name} '
                      f'({done + fail}/{n_total}), elapsed={elapsed:.0f}s')
                del active[proc]

                # Launch a replacement if there are pending instances
                if pending:
                    next_name, next_path = pending.pop(0)
                    cmd = [sys.executable, '-u', WORKER_SCRIPT, next_name, next_path, OUTDIR]
                    new_proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True,
                        env={**os.environ, 'PYTHONUTF8': '1', 'OPENBLAS_NUM_THREADS': '1',
                             'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                    )
                    active[new_proc] = (next_name, next_path)
                break  # re-iterate active dict after modification
        else:
            time.sleep(0.5)

    t_total = time.time() - t_start

    # Merge all .npz into dataset.npz
    print(f'\n  --- 合并数据集 ---')
    merge_dataset(OUTDIR, INSTANCES)

    print(f'\n{"=" * 70}')
    print(f'  完成! 成功={done}, 失败={fail}')
    print(f'  总时间: {t_total:.0f}s ({t_total/3600:.1f}h)')
    print(f'  Output: {OUTDIR}')
    print(f'  Next:   python gnn_train.py')
    print(f'{"=" * 70}')
