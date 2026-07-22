"""
GNN Training Data Generator — 从 HMOA + USS 解中提取监督信号。

流程：
  1. 对每个 Dumas 实例运行 HMOA（启用 USS）
  2. 从最终 Pareto Front 提取每个解的节点分配（卡车=0 / 无人机=1）
  3. 构建 PyG Data 对象并保存为可训练数据集

节点特征 (6维)：
  [norm_x, norm_y, norm_early, norm_late, norm_service, drone_accessible]

标签：
  0 = 卡车节点（在 truck_route 中）
  1 = 无人机节点（在 drone_deliveries 中）
"""

import sys, os, json, time, random
import numpy as np
import torch
from typing import List, Optional
from datetime import datetime

# 路径设置
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'core'))
os.chdir(ROOT)

import config
from utils import load_dumas_instance
from algorithm import hmoa
from problem import Solution, ProblemInstance

# =============================================================================
# USS 配置 (启用混合可行/不可行选择)
# =============================================================================
config.POPULATION_SIZE = 200
config.MAX_ITERATIONS = 500
config.MAX_PF_SIZE = 500
config.KMAX = 5
config.CROSSOVER_RATE = 0.8
config.MUTATION_RATE = 0.3
config.ZETA = 0.2  # USS 参数

# =============================================================================
# 实例列表
# =============================================================================
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

NUM_RUNS_PER_INSTANCE = 5   # 每个实例跑 5 次 HMOA 收集多样解
OUTPUT_DIR = os.path.join(ROOT, 'output', 'gnn_training_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 特征提取
# =============================================================================

def extract_node_features(instance: ProblemInstance) -> np.ndarray:
    """
    提取每个客户节点的 6 维特征，形状 (N, 6)。

    特征归一化策略（每个实例独立）：
      - x, y: 归一化到 [0, 1]
      - 时间特征: 用整个实例的 min/max 缩放到 [0, 1]
      - drone_accessible: 0/1 保持不变
    """
    n = instance.num_customers
    features = np.zeros((n, 6), dtype=np.float32)

    # 收集所有坐标和时间的 min/max
    xs, ys = [], []
    early_times, late_times, service_times = [], [], []

    for node in instance.nodes.values():
        xs.append(node.x)
        ys.append(node.y)
        early_times.append(node.earliest_time)
        late_times.append(node.latest_time)
        service_times.append(node.service_time)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    e_min, e_max = min(early_times), max(early_times)
    l_min, l_max = min(late_times), max(late_times)
    s_min, s_max = min(service_times), max(service_times)

    # 避免除零
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    e_range = e_max - e_min if e_max > e_min else 1.0
    l_range = l_max - l_min if l_max > l_min else 1.0
    s_range = s_max - s_min if s_max > s_min else 1.0

    for i, node in instance.nodes.items():
        idx = i - 1  # node ID 从 1 开始 → 0-index
        features[idx, 0] = (node.x - x_min) / x_range
        features[idx, 1] = (node.y - y_min) / y_range
        features[idx, 2] = (node.earliest_time - e_min) / e_range
        features[idx, 3] = (node.latest_time - l_min) / l_range
        features[idx, 4] = (node.service_time - s_min) / s_range
        features[idx, 5] = 1.0 if node.is_drone_eligible else 0.0

    return features


def extract_node_labels(solution: Solution, num_customers: int) -> np.ndarray:
    """
    从解中提取节点标签。

    标签:
      0 = 卡车节点 (在 truck_route 中)
      1 = 无人机节点 (在 drone_deliveries 中)

    注意: 有些节点可能同时出现在 truck_route 和 drone_deliveries？
    理论上不会，但以防万一：如果同时出现则优先标为卡车。
    有些节点可能未覆盖（不在任何 route 中）→ 标为 -1（忽略）。
    """
    labels = np.full(num_customers, -1, dtype=np.int64)  # -1 = unassigned

    truck_set = set(solution.truck_route)
    drone_set = {d.customer for d in solution.drone_deliveries}

    for node_id in range(1, num_customers + 1):
        idx = node_id - 1
        if node_id in truck_set:
            labels[idx] = 0
        elif node_id in drone_set:
            labels[idx] = 1
        # else: 保持 -1（该解未覆盖此节点）

    return labels


def instance_to_graph(
    instance: ProblemInstance,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    构建一个图的训练样本。

    Returns
    -------
    dict with:
      x: (N, 6) 特征
      y: (N,) 标签 (-1 表示忽略)
      n: 客户节点数
      instance_name: 实例名
    """
    return {
        'x': features,           # (N, 6)
        'y': labels,             # (N,)
        'n': instance.num_customers,
    }


# =============================================================================
# 数据生成主函数
# =============================================================================

def generate_gnn_data(
    instances: list,
    runs_per_instance: int = 5,
    output_dir: str = OUTPUT_DIR,
) -> dict:
    """
    运行 HMOA(+USS) 并生成 GNN 训练数据。

    Returns
    -------
    dataset: dict {instance_name: [graph_dict, ...]}
    """
    dataset = {}
    total_graphs = 0
    total_labeled = 0

    print('=' * 70)
    print('  GNN 训练数据生成')
    print(f'  HMOA + USS (zeta={config.ZETA})')
    print(f'  Instances: {len(instances)} × {runs_per_instance} runs')
    print(f'  pop={config.POPULATION_SIZE}, iter={config.MAX_ITERATIONS}')
    print(f'  Started: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print('=' * 70)

    for inst_name, filepath in instances:
        print(f'\n{"-" * 60}')
        print(f'  [{datetime.now():%H:%M:%S}] 处理: {inst_name}')

        instance = load_dumas_instance(filepath, num_drones=config.DEFAULT_DRONE_COUNT)
        features = extract_node_features(instance)
        print(f'    客户节点: {instance.num_customers}, 特征形状: {features.shape}')

        instance_graphs = []

        for run in range(runs_per_instance):
            seed = hash(f'{inst_name}_gnn_{run}') % (2 ** 31)
            t0 = time.time()

            # 运行 HMOA (USS 在 update2_population 内部启用)
            pf, _, _ = hmoa(instance, config.POPULATION_SIZE,
                            config.MAX_ITERATIONS, seed, verbose=False)

            elapsed = time.time() - t0

            # 从 PF 中提取每个解的标签
            valid_solutions = 0
            for sol in pf:
                labels = extract_node_labels(sol, instance.num_customers)
                labeled_count = (labels >= 0).sum()

                # 只保留有足够标签的解（>50% 节点被覆盖）
                if labeled_count >= instance.num_customers * 0.5:
                    graph = instance_to_graph(instance, features, labels)
                    graph['instance'] = inst_name
                    graph['run'] = run
                    graph['cost'] = float(sol._cost) if sol._cost is not None else None
                    graph['satisfaction'] = float(sol._satisfaction) if sol._satisfaction is not None else None
                    instance_graphs.append(graph)
                    valid_solutions += 1

            print(f'     Run {run + 1:2d}: PF={len(pf)}, '
                  f'有效图={valid_solutions}, {elapsed:.0f}s')

        dataset[inst_name] = instance_graphs
        total_graphs += len(instance_graphs)
        for g in instance_graphs:
            total_labeled += (g['y'] >= 0).sum()

        # 增量保存
        save_dataset(dataset, output_dir)
        print(f'  [{datetime.now():%H:%M:%S}] 累积: {total_graphs} 图, '
              f'{total_labeled} 标签节点')

    print(f'\n{"=" * 70}')
    print(f'  生成完成!')
    print(f'  总图数: {total_graphs}')
    print(f'  总标签: {total_labeled}')
    print(f'  保存至: {output_dir}')
    print(f'  Time: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'{"=" * 70}')

    return dataset


def save_dataset(dataset: dict, output_dir: str):
    """将数据集保存为 .npz 文件（高效 numpy 格式）和元数据 JSON。"""
    # 构建 numpy 数组
    all_x, all_y = [], []
    all_instance_names = []
    all_graph_indices = []  # 每个图的节点数
    all_metadata = []

    for inst_name, graphs in dataset.items():
        for graph in graphs:
            all_x.append(graph['x'])
            all_y.append(graph['y'])
            all_instance_names.append(graph['instance'])
            all_graph_indices.append(graph['n'])
            all_metadata.append({
                'instance': graph['instance'],
                'run': graph['run'],
                'n': graph['n'],
                'cost': graph.get('cost'),
                'satisfaction': graph.get('satisfaction'),
            })

    if not all_x:
        # 空数据集 → 清空文件
        np.savez(os.path.join(output_dir, 'dataset.npz'),
                 x=np.zeros((0, 6), dtype=np.float32),
                 y=np.zeros((0,), dtype=np.int64),
                 graph_indices=np.zeros((0,), dtype=np.int64),
                 instance_names=np.array([], dtype=object))
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump({'graphs': [], 'total': 0}, f, indent=2)
        return

    # 拼接所有图的特征和标签
    x_all = np.concatenate(all_x, axis=0)  # (total_nodes, 6)
    y_all = np.concatenate(all_y, axis=0)  # (total_nodes,)

    # 记录每个图的起始索引和节点数
    # graph_ptr 形状 (num_graphs, 2): [start_idx, num_nodes]
    ptrs = []
    offset = 0
    for n in all_graph_indices:
        ptrs.append([offset, n])
        offset += n
    graph_ptr = np.array(ptrs, dtype=np.int64)

    # 保存
    np.savez(os.path.join(output_dir, 'dataset.npz'),
             x=x_all, y=y_all, graph_ptr=graph_ptr,
             instance_names=np.array(all_instance_names, dtype=object))

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'total_graphs': len(all_x),
            'total_nodes': int(len(x_all)),
            'labeled_nodes': int((y_all >= 0).sum()),
            'truck_nodes': int((y_all == 0).sum()),
            'drone_nodes': int((y_all == 1).sum()),
            'unlabeled_nodes': int((y_all == -1).sum()),
            'graphs': all_metadata,
            'config': {
                'population_size': config.POPULATION_SIZE,
                'max_iterations': config.MAX_ITERATIONS,
                'kmax': config.KMAX,
                'zeta': config.ZETA,
                'runs_per_instance': NUM_RUNS_PER_INSTANCE,
            },
            'generated_at': datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print(f'    已保存 {len(all_x)} 图 → {output_dir}/dataset.npz')


# =============================================================================
# 加载数据集
# =============================================================================

def load_gnn_dataset(data_dir: str = OUTPUT_DIR) -> dict:
    """从 .npz 文件加载数据集。"""
    npz_path = os.path.join(data_dir, 'dataset.npz')
    meta_path = os.path.join(data_dir, 'metadata.json')

    if not os.path.exists(npz_path):
        print(f'错误: 未找到数据集 {npz_path}')
        return {'x': None, 'y': None, 'graph_ptr': None, 'metadata': None}

    data = np.load(npz_path, allow_pickle=True)
    metadata = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    return {
        'x': data['x'],
        'y': data['y'],
        'graph_ptr': data['graph_ptr'],
        'instance_names': data['instance_names'],
        'metadata': metadata,
    }


# =============================================================================
# 统计信息
# =============================================================================

def print_dataset_stats(data_dir: str = OUTPUT_DIR):
    """打印数据集的统计信息。"""
    data = load_gnn_dataset(data_dir)
    if data['x'] is None:
        return

    meta = data['metadata']
    x, y, ptr = data['x'], data['y'], data['graph_ptr']

    print(f'\n{"=" * 50}')
    print(f'  GNN 数据集统计')
    print(f'{"=" * 50}')
    print(f'  总图数:      {meta.get("total_graphs", "?")}')
    print(f'  总节点数:    {meta.get("total_nodes", len(x))}')
    print(f'  标签节点数:  {meta.get("labeled_nodes", (y >= 0).sum())}')
    print(f'  ├─ 卡车:     {meta.get("truck_nodes", (y == 0).sum())}')
    print(f'  └─ 无人机:   {meta.get("drone_nodes", (y == 1).sum())}')
    if meta.get('unlabeled_nodes'):
        print(f'  未标签节点:  {meta["unlabeled_nodes"]}')
    print()

    # 按实例大小统计
    if ptr is not None and len(ptr) > 0:
        sizes = ptr[:, 1]
        print(f'  图大小分布:')
        for n in sorted(set(sizes)):
            count = (sizes == n).sum()
            print(f'    n={n}: {count} 图')

    # 标签分布
    labeled = y[y >= 0]
    if len(labeled) > 0:
        print(f'\n  标签分布:')
        print(f'    卡车 (0):  {(labeled == 0).sum()} ({100 * (labeled == 0).sum() / len(labeled):.1f}%)')
        print(f'    无人机 (1): {(labeled == 1).sum()} ({100 * (labeled == 1).sum() / len(labeled):.1f}%)')


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    # 生成数据
    dataset = generate_gnn_data(INSTANCES, runs_per_instance=NUM_RUNS_PER_INSTANCE)

    # 打印统计
    print_dataset_stats(OUTPUT_DIR)

    print(f'\n数据生成完成!')
    print(f'训练脚本: python gnn_train.py')
