# HMOA — Hybrid Multi-Objective Optimization for Truck-Drone Routing

混合多目标卡车-无人机协同路径规划（Mo-CRPTW-mD），基于 IEEE TITS 2022 论文。

## 项目结构

```
hmoa_impl.v4/
│
├── core/                          # HMOA 核心算法
│   ├── algorithm.py               #   主算法 + PLS + USS + GNN桥接
│   ├── config.py                  #   参数配置
│   ├── metrics.py                 #   HV, C-metric
│   ├── operators.py               #   N1-N6, 交叉, 变异, AssignNodes, Repair
│   ├── operators_pls.py           #   PLS 专用算子
│   ├── problem.py                 #   Solution, ProblemInstance
│   ├── utils.py                   #   Dumas实例加载, 可视化, 合成实例
│   └── worker.py                  #   单实例 worker
│
├── gnn_*.py                       # GNN 核心代码（根目录）
│   ├── gnn_model.py               #   GAT 节点分类器 (36K参数)
│   ├── gnn_bridge.py              #   GNN-HMOA 桥接
│   ├── gnn_train.py               #   训练脚本（支持自适应权重）
│   ├── gnn_dataloader.py          #   GPU 优化 DataLoader
│   ├── gnn_inference.py           #   推理脚本
│   ├── gnn_data.py                #   训练数据生成
│   └── gnn_worker.py              #   数据生成 worker
│
├── experiments/                   # 实验脚本
│   ├── experiment.py              #   iter=500 基线
│   ├── experiment_uss.py          #   USS 改进
│   ├── experiment_dynamic.py      #   动态变异率
│   ├── experiment_hybrid.py       #   混合成本
│   ├── generate_gnn_data.py       #   GNN数据并行生成
│   └── run_gnn_experiment.py      #   GNN-HMOA 对比
│
├── dumas_instances/               # 405 个 TSPTW 实例
│   ├── n20w80.001.txt ~ .005.txt  #   原始 66 个 + 新增 339 个
│   └── sources/                   #   按来源分类
│       ├── original/              #      原始 Dumas (100个)
│       ├── DumasEtAl/             #      Dumas et al. (30个)
│       ├── GendreauEtAl/          #      Gendreau et al. (150个)
│       ├── DaSilvaUrrutia/        #      DaSilva & Urrutia (45个)
│       └── OhlmannThomas/         #      Ohlmann & Thomas (30个)
│
├── output/                        # 实验结果
│   ├── parallel_w80_*/            #   HMOA 基线实验
│   ├── gnn_training_data/         #   GNN v1 训练数据 (w=80)
│   ├── gnn_data_w60/              #   GNN v2 训练数据 (w=60)
│   ├── gnn_data_n100/             #   GNN n100 训练数据
│   ├── gnn_models/                #   旧模型 (w=80)
│   ├── gnn_models_w60/            #   新模型 (w=60, 推荐)
│   ├── experiment_gnn_gnn/        #   GNN+USS 对比
│   ├── experiment_gnn_w60_on_w80/ #   w=60→w=80 泛化
│   ├── hmoa_w40/                  #   w=40 基线
│   ├── gnn_w40/                   #   GNN w=40 测试
│   └── grid_search/               #   网格搜索结果
│
├── .claude/
│   └── skills/hmoa.md             # Claude Code skill
│
├── docs/
│   ├── EXPERIMENT_LOG.md          # 完整实验日志
│   └── GNN_SUMMARY.md             # GNN 实验总结
│
├── compare_gnn_uss.py             # 对比分析脚本
├── main.py                        # CLI 入口
└── README.md                      # 本文件
```

## 论文

- **题目**: Hybrid Multi-Objective Optimization Approach With Pareto Local Search for Collaborative Truck-Drone Routing Problems Considering Flexible Time Windows
- **作者**: Qizhang Luo, Guohua Wu, Bin Ji, Ling Wang, Ponnuthurai Nagaratnam Suganthan
- **期刊**: IEEE Trans. on Intelligent Transportation Systems, 2022

## 推荐参数

| 参数 | 值 |
|------|-----|
| 种群大小 | 200 |
| 迭代次数 | 500 |
| GNN 初始化 | 50%（前一半种群） |
| USS | 启用 (ζ=0.2) |
| PLS kmax | 5 |
| 交叉率 | 0.8 |
| 变异率 | 0.3 |
| 无人机数 | 3 |
| pos_weight | 自适应公式 |

## 环境

```bash
conda activate vrp_env312
# Python 3.12, PyTorch 2.13+cu121, PyTorch Geometric 2.8
# NVIDIA RTX 3060 6GB (训练用)
```

## 快速开始

```bash
# 完整实验
python experiments/run_gnn_experiment.py

# 训练 GNN
python gnn_train.py --model-dir output/gnn_models_w60

# 单实例预测
python gnn_inference.py dumas_instances/n20w80.001.txt

# 快速测试
python -c "import sys; sys.path.insert(0,'core'); from utils import load_dumas_instance; from algorithm import hmoa; inst=load_dumas_instance('dumas_instances/n20w80.001.txt',num_drones=3); pf,_,_=hmoa(inst,50,50,42,verbose=True)"
```

## 实验结果

| 配置 | 胜率 | 说明 |
|:-----|:----:|:-----|
| **GNN 50% + USS** | **11/20** 🥇 | 最优配置 |
| USS 单独 | — | 改进基线 |
| GNN(w=60)→w=80 | 10/20 | 泛化有效 |
| GNN(noUSS) | 9/20 | 可用 |
| GNN 100% | 0/20 ❌ | 多样性崩溃 |

详见 `docs/EXPERIMENT_LOG.md` 和 `docs/GNN_SUMMARY.md`。

## 自适应 pos_weight

GNN 训练时无需手动指定 `--pos-weight`，自动根据规模查表插值：

| 规模 | 权重 |
|:----:|:----:|
| n20 | 1.45 |
| n40 | 1.00 |
| n60 | 1.20 |
| n80 | 1.45 |
| n100 | 1.20 |
| >100 | 1.20 |
