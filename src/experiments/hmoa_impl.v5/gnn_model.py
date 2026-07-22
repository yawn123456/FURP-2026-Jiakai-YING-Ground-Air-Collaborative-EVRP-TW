"""
GAT 节点分类模型 — 预测卡车/无人机节点分配。

架构：
  GATConv(6 → 64, heads=4, concat) → 256  → ELU → Dropout(0.2)
  GATConv(256 → 128, heads=1, concat) → 128 → ELU → Dropout(0.2)
  global_mean_pool → 聚合为图级标量 → 拼接到每个节点
  Linear(128 + 1, 1) → 单 Logit（二分类）

输入:  (N, 6) 节点特征, (2, E) edge_index, (N,) batch索引
输出:  (N, 1) 每个节点的 Logit（正值→卡车，负值→无人机，由训练决定）

注意: edge_index 应在数据预处理时通过 build_knn_graph() 预计算并存入 Data 对象，
      训练时直接传递，避免每 epoch 重复构建 KNN 图。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# =============================================================================
# 独立工具函数：构建 KNN 图（用于数据预处理，不参与模型前向）
# =============================================================================

def build_knn_graph(pos: torch.Tensor, k: int, batch: torch.Tensor | None = None) -> torch.Tensor:
    """
    手动构建 KNN 图（无需 pyg-lib 依赖）。

    Parameters
    ----------
    pos : torch.Tensor  shape=(N, 2)
        XY 坐标。
    k : int
        最近邻数（包含自环，实际邻居 = k）。
    batch : torch.Tensor or None  shape=(N,)
        每个节点所属图的索引。None 表示所有节点属于同一图。

    Returns
    -------
    edge_index : LongTensor  shape=(2, N * k_eff)
    """
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    device = pos.device
    edge_list = []

    for b in batch.unique():
        mask = batch == b
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        local_pos = pos[mask]
        n_local = local_pos.size(0)

        dist = torch.cdist(local_pos, local_pos, p=2)
        k_eff = min(k, n_local)
        _, nbr_idx = torch.topk(dist, k_eff, dim=1, largest=False)

        global_src = idx.unsqueeze(1).expand(-1, k_eff)  # (n_local, k)
        global_dst = idx[nbr_idx]                         # (n_local, k)

        # stack → (2, n_local, k), reshape → (2, n_local * k)
        edge = torch.stack([global_src, global_dst], dim=0)  # (2, n, k)
        edge = edge.reshape(2, -1)                            # (2, n*k)
        edge_list.append(edge)

    return torch.cat(edge_list, dim=1).to(device)


# =============================================================================
# GAT 模型
# =============================================================================

class GATNodeClassifier(nn.Module):
    """
    图注意力网络节点分类器。

    Parameters
    ----------
    in_channels : int
        输入特征维度（默认 6）。
    hidden_dim : int
        第二层 GAT 隐藏维度（默认 128）。
    num_heads : int
        第一层注意力头数（默认 4）。
    dropout : float
        Dropout 比率（默认 0.2）。
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout_p = dropout

        # --- Layer 1: 多头 GAT ---
        # GATConv(6 → 64, heads=4) → 64×4 = 256
        self.conv1 = GATConv(in_channels, hidden_dim // 2, heads=num_heads, concat=True)
        conv1_out = (hidden_dim // 2) * num_heads  # = 256
        self.ln1 = nn.LayerNorm(conv1_out)

        # --- Layer 2: 单头 GAT ---
        self.conv2 = GATConv(conv1_out, hidden_dim, heads=1, concat=False)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # --- Output: 节点嵌入 + 全局上下文 → 单 Logit ---
        self.out_proj = nn.Linear(hidden_dim + 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape=(N, 6)
            归一化节点特征。
        edge_index : torch.Tensor  shape=(2, E)
            预计算的边索引（由 build_knn_graph 生成）。
        batch : torch.Tensor  shape=(N,)
            每个节点所属图的索引（0 … B-1）。

        Returns
        -------
        logits : torch.Tensor  shape=(N, 1)
            每个节点的二分类 Logit。
        """
        # --- Layer 1: 多头 GAT + LayerNorm + ELU + Dropout ---
        x = self.conv1(x, edge_index)          # (N, 256)
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # --- Layer 2: 单头 GAT + LayerNorm + ELU + Dropout ---
        x = self.conv2(x, edge_index)          # (N, 128)
        x = self.ln2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # --- 全局上下文 (Global Context) ---
        graph_emb = global_mean_pool(x, batch)            # (B, 128)
        graph_scalar = graph_emb.mean(dim=-1, keepdim=True)  # (B, 1)
        ctx = graph_scalar[batch]                         # (N, 1)

        # --- 拼接 + 输出 ---
        x = torch.cat([x, ctx], dim=-1)                  # (N, 129)
        logits = self.out_proj(x)                         # (N, 1)

        return logits


# =============================================================================
# 快速自测
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATNodeClassifier(in_channels=6, hidden_dim=128, num_heads=4, dropout=0.2).to(device)

    # 模拟 3 个图: n=20, 40, 50
    batch_sizes = [20, 40, 50]
    total = sum(batch_sizes)
    x = torch.randn(total, 6, device=device)
    pos = torch.rand(total, 2, device=device)
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(batch_sizes)]).to(device)

    # 预计算 edge_index（取代模型内部构建）
    edge_index = build_knn_graph(pos, k=8, batch=batch)

    logits = model(x, edge_index, batch)

    print(f'输入节点: {total}')
    print(f'输出形状: {logits.shape}  ← 期望 (N, 1)')
    print(f'参数量:   {sum(p.numel() for p in model.parameters()):,}')

    # 模拟二分类
    preds = (logits > 0).squeeze(-1)
    truck, drone = (preds == 1).sum().item(), (preds == 0).sum().item()
    print(f'预测: 卡车={truck}, 无人机={drone}')
    print('模型结构正确!')
