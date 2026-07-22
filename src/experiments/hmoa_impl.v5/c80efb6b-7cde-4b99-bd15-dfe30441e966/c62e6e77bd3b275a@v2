"""
GNN Inference — 用训练好的 GAT 模型预测节点分配（卡车/无人机）。

用法:
  python gnn_inference.py <instance_file> [--model output/gnn_models/best_model.pt]

输出:
  - 打印每个节点的预测 (T=卡车, D=无人机)
  - 可选: 保存预测结果为 JSON
"""

import sys, os, json, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'core'))

from gnn_classifier import GATNodeClassifier, build_pyg_data
from utils import load_dumas_instance
import config


def predict(
    model: GATNodeClassifier,
    instance_file: str,
    device: torch.device = torch.device('cpu'),
    return_logits: bool = False,
):
    """
    对实例进行节点分类预测。

    Parameters
    ----------
    model : GATNodeClassifier
        训练好的模型
    instance_file : str
        Dumas 实例文件路径
    device : torch.device
        计算设备
    return_logits : bool
        是否同时返回 logits

    Returns
    -------
    preds : np.ndarray  shape=(N,)
        0 = 卡车, 1 = 无人机
    probs : np.ndarray  shape=(N, 2)
        每个类的概率
    instance : ProblemInstance
    features : np.ndarray  shape=(N, 6)
    logits (可选) : np.ndarray
    """
    instance = load_dumas_instance(instance_file,
                                   num_drones=config.DEFAULT_DRONE_COUNT)

    # 提取特征（与训练时一致）
    n = instance.num_customers
    features = np.zeros((n, 6), dtype=np.float32)

    xs = [node.x for node in instance.nodes.values()]
    ys = [node.y for node in instance.nodes.values()]
    early = [node.earliest_time for node in instance.nodes.values()]
    late = [node.latest_time for node in instance.nodes.values()]
    service = [node.service_time for node in instance.nodes.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    e_min, e_max = min(early), max(early)
    l_min, l_max = min(late), max(late)
    s_min, s_max = min(service), max(service)

    x_r = x_max - x_min if x_max > x_min else 1.0
    y_r = y_max - y_min if y_max > y_min else 1.0
    e_r = e_max - e_min if e_max > e_min else 1.0
    l_r = l_max - l_min if l_max > l_min else 1.0
    s_r = s_max - s_min if s_max > s_min else 1.0

    for i, node in instance.nodes.items():
        idx = i - 1
        features[idx, 0] = (node.x - x_min) / x_r
        features[idx, 1] = (node.y - y_min) / y_r
        features[idx, 2] = (node.earliest_time - e_min) / e_r
        features[idx, 3] = (node.latest_time - l_min) / l_r
        features[idx, 4] = (node.service_time - s_min) / s_r
        features[idx, 5] = 1.0 if node.is_drone_eligible else 0.0

    # 构建图
    x_tensor = torch.from_numpy(features).float().to(device)
    pos_tensor = x_tensor[:, :2].clone()
    batch_tensor = torch.zeros(n, dtype=torch.long, device=device)

    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor, pos_tensor, batch_tensor)
        probs = F.softmax(logits, dim=-1)

    preds = logits.argmax(dim=-1).cpu().numpy()
    probs_np = probs.cpu().numpy()
    logits_np = logits.cpu().numpy()

    result = (preds, probs_np, instance, features)
    if return_logits:
        result = result + (logits_np,)
    return result


def print_predictions(preds, probs, instance):
    """打印预测结果。"""
    print(f'\n  Node  │ Truck_prob  Drone_prob  │ Assignment')
    print(f'  ───────┼──────────────────────────┼────────────')
    for i, node in instance.nodes.values():
        idx = i - 1
        t_prob = probs[idx, 0]
        d_prob = probs[idx, 1]
        label = 'T ← TRUCK' if preds[idx] == 0 else 'D ← DRONE'
        print(f'  {i:<5} │ {t_prob:.4f}      {d_prob:.4f}    │ {label}')

    n_truck = (preds == 0).sum()
    n_drone = (preds == 1).sum()
    print(f'\n  摘要: {n_truck} 卡车节点, {n_drone} 无人机节点'
          f'  (共 {len(preds)} 客户)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN 节点分类推理')
    parser.add_argument('instance', type=str, help='Dumas 实例文件路径')
    parser.add_argument('--model', type=str,
                        default=os.path.join(ROOT, 'output', 'gnn_models', 'best_model.pt'),
                        help='模型权重路径')
    parser.add_argument('--save', type=str, default=None,
                        help='保存预测结果为 JSON')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = GATNodeClassifier().to(device)
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f'  模型加载成功: {args.model}')

    # 预测
    from torch.nn import functional as F
    preds, probs, instance, features = predict(
        model, args.instance, device, return_logits=False)

    print_predictions(preds, probs, instance)

    if args.save:
        output = []
        for i, node in instance.nodes.values():
            idx = i - 1
            output.append({
                'node_id': i,
                'x': node.x,
                'y': node.y,
                'prediction': int(preds[idx]),
                'label': 'truck' if preds[idx] == 0 else 'drone',
                'truck_prob': float(probs[idx, 0]),
                'drone_prob': float(probs[idx, 1]),
            })
        with open(args.save, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\n  结果已保存至: {args.save}')
