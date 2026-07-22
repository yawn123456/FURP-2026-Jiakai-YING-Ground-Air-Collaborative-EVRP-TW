"""
按实例规模（n=20, 40, 60, 80）将数据集拆分为独立文件夹，
每个图保存为独立的 PyG Data 对象，方便 torch_geometric.loader.DataLoader 加载。
"""

import sys, os, json, numpy as np
import torch
from torch_geometric.data import Data
from gnn_model import build_knn_graph

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'output', 'gnn_training_data', 'dataset.npz')
DST = os.path.join(ROOT, 'output', 'gnn_training_data', 'by_scale')
K = 8  # KNN 图的 K 值
os.makedirs(DST, exist_ok=True)

def reorganize():
    print(f'加载数据集: {SRC}')
    data = np.load(SRC, allow_pickle=True)
    x_all = data['x']                # (total_nodes, 6)
    y_all = data['y']                # (total_nodes,)
    ptr = data['graph_ptr']          # (num_graphs, 2): [start, n]
    names = data['instance_names']   # (num_graphs,)

    total_graphs = len(ptr)
    print(f'总图数: {total_graphs}, 总节点数: {len(x_all)}')
    print(f'KNN K={K}, 正在预计算 edge_index...')

    # 按规模分组: n20, n40, n60, n80
    scale_bins = {}
    for i in range(total_graphs):
        start, n = ptr[i]
        # 从 instance_name 解析规模，例如 'n20w80_001' → 'n20'
        inst = names[i]
        scale_key = inst.split('w')[0]  # 'n20', 'n40', ...
        scale_bins.setdefault(scale_key, []).append(i)

    for scale_key, indices in sorted(scale_bins.items()):
        scale_dir = os.path.join(DST, scale_key)
        os.makedirs(scale_dir, exist_ok=True)
        print(f'\n{scale_key}: {len(indices)} 图')

        for idx in indices:
            start, n = ptr[idx]
            x = torch.from_numpy(x_all[start:start + n].copy()).float()  # (n, 6)
            y = torch.from_numpy(y_all[start:start + n].copy()).long()   # (n,)
            pos = x[:, :2].clone()                                       # (n, 2)

            # 预计算 KNN edge_index 并存入 Data
            # 单个图: batch 为全零向量
            batch = torch.zeros(n, dtype=torch.long)
            edge_index = build_knn_graph(pos, k=K, batch=batch)

            # 构建 PyG Data 对象（包含预计算的 edge_index）
            graph = Data(x=x, y=y, pos=pos, edge_index=edge_index)

            # 保存为 .pt 文件
            fname = f'graph_{idx:06d}.pt'
            torch.save(graph, os.path.join(scale_dir, fname))

        # 写入元数据
        meta = {
            'scale': scale_key,
            'num_nodes': int(ptr[indices[0]][1]),  # all same scale
            'num_graphs': len(indices),
            'graph_indices': indices,
        }
        with open(os.path.join(scale_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'  → {scale_dir}  ({len(indices)} 个 .pt 文件)')

    # 汇总
    print('\n' + '=' * 50)
    print('拆分完成!')
    for scale_key in sorted(scale_bins):
        n_files = len(os.listdir(os.path.join(DST, scale_key))) - 1  # -1 for metadata.json
        print(f'  {scale_key}: {n_files} 图')
    print(f'保存至: {DST}')


if __name__ == '__main__':
    reorganize()
