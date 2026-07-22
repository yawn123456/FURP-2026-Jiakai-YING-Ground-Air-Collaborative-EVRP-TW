---
name: hmoa
description: HMOA 项目 — 混合多目标卡车-无人机路径规划
---

# HMOA 项目指南

## 项目结构

```
hmoa_impl.v4/
├── core/                        # HMOA 核心算法
│   ├── algorithm.py             #   主算法 (hmoa), 非支配排序, PLS
│   ├── config.py                #   参数配置
│   ├── metrics.py               #   HV, C-metric 计算
│   ├── operators.py             #   N1-N6 算子, 交叉, 变异, 修复
│   ├── operators_pls.py         #   PLS 专用算子 (N4,N5,N6)
│   ├── problem.py               #   问题模型 (Solution, ProblemInstance)
│   ├── utils.py                 #   实例加载, 可视化
│   └── worker.py                #   单实例 worker (被 experiments 调用)
│
├── experiments/                 # 实验脚本
│   ├── experiment.py            #   iter=500 基线
│   ├── experiment_uss.py        #   USS 改进
│   ├── experiment_dynamic.py    #   动态变异率
│   ├── experiment_hybrid.py     #   混合成本
│   ├── generate_gnn_data.py     #   GNN 训练数据生成 (并行)
│   └── run_gnn_experiment.py    #   GNN-HMOA 对比实验
│
├── gnn_*.py                     # GNN 相关 (根目录)
│   ├── gnn_model.py             #   GAT 模型定义 (36K 参数)
│   ├── gnn_bridge.py            #   GNN-HMOA 桥接
│   ├── gnn_dataloader.py        #   GPU 优化 DataLoader
│   ├── gnn_train.py             #   训练脚本
│   ├── gnn_inference.py         #   推理脚本
│   ├── gnn_data.py              #   数据生成
│   ├── gnn_reorganize.py        #   按规模拆分数据
│   └── gnn_worker.py            #   数据生成 worker
│
├── dumas_instances/             # Dumas TSPTW 实例
├── output/                      # 实验结果
│   ├── parallel_w80_iter500/    #   HMOA 基线 (06-26)
│   ├── parallel_w80_uss/        #   USS 改进 (07-15)
│   ├── gnn_training_data/       #   GNN 训练数据 (07-20)
│   ├── gnn_models/              #   GAT 模型权重 (07-21)
│   ├── experiment_gnn_gnn/      #   GNN+USS 对比 (07-21)
│   └── gnn_nouss_g/             #   GNN 无 USS (07-21)
│
├── docs/
│   ├── README.md
│   ├── EXPERIMENT_LOG.md        # 实验日志
│   └── GNN_SUMMARY.md           # GNN 实验总结
│
└── main.py                      # CLI 入口
```

## 常用命令

### 训练 GNN 模型
```
python gnn_train.py                          # 训练全部4个规模
python gnn_train.py --scale n80             # 只训练 n80
python gnn_train.py --pos-weight 1.45       # 指定权重
```

### GNN-HMOA 对比实验
```
python experiments/run_gnn_experiment.py     # 20实例并行
```

### 单实例预测
```
python gnn_inference.py dumas_instances/n20w80.001.txt
```

## 实验记录规范

所有实验必须记录到 `EXPERIMENT_LOG.md`，包含：

- **运行日期** — 实验执行日期
- **配置参数** — iter, pop, USS, GNN 比例等
- **运行次数** — 每实例 runs, 并行数
- **结果数据** — HV, C-metric, 时间, 成本/满意度范围
- **结论** — 胜率统计, 推荐/不推荐

实验目录用 `output/` 统一管理，命名规则：
- `parallel_w80_<实验名>/` — 原始 HMOA 实验
- `experiment_<实验名>/` — 改进实验
- `gnn_nouss_<标签>/` — GNN 相关实验

## 最佳配置

| 参数 | 值 |
|------|-----|
| population_size | 200 |
| max_iterations | 500 |
| GNN init ratio | 50% |
| USS | enabled (zeta=0.2) |
| PLS kmax | 5 |
| pos_weight | 1.45 |
| crossover_rate | 0.8 |
| mutation_rate | 0.3 |

## 实验结果摘要

| 配置 | GNN 胜率 |
|:----|:---------:|
| GNN 50% + USS | **11/20** 🥇 |
| GNN 50% (no USS) | 9/20 |
| GNN 100% + USS | 0/20 ❌ |

## 关键文件

- `core/algorithm.py` — `hmoa()` 主函数, 第 429 行
- `core/operators.py` — `assign_nodes()` 初始化, 第 34 行
- `gnn_bridge.py` — `GNNBridge` 桥接类
- `gnn_model.py` — `GATNodeClassifier` 模型

## 踩坑记录

### 1. 模型文件覆盖（最严重）
**问题：** GNN 训练脚本默认保存到 `output/gnn_models/`，不同实验的模型互相覆盖。
**后果：** w=80 模型被 w=60 训练覆盖，历史实验结果无法复现。
**规则：** 每次训练必须指定 `--model-dir` 参数，存到独立目录。
```
python gnn_train.py --model-dir output/gnn_models_w60/
```

### 2. Worker 文件路径问题
**问题：** worker.py 执行 `os.chdir()` 切换到 `core/` 目录，相对路径失效。
**后果：** 文件找不到、输出目录写错位置。
**规则：** 所有路径必须传绝对路径，不能用相对路径。
```bash
# 正确
python core/worker.py name "$ROOT/dumas_instances/n20w80.001.txt" 500 "$ROOT/output/exp/"

# 错误（worker 会 chdir 到 core/，找不到文件）
python core/worker.py name "dumas_instances/n20w80.001.txt" 500 "output/exp/"
```

### 3. CUDA 多进程冲突
**问题：** 多个 worker 同时加载 GNN 模型到 GPU 导致 CUDA 崩溃。
**后果：** worker 全部报错退出，0 实例完成。
**规则：** 批量运行时必须禁用 GPU：
```bash
CUDA_VISIBLE_DEVICES=-1 python core/worker.py ...
```

### 4. GNN 桥接导入路径
**问题：** algorithm.py 从 `core/` 目录运行，找不到根目录的 `gnn_bridge.py`。
**后果：** GNN 初始化静默回退为随机初始化，实验不生效。
**规则：** algorithm.py 已修复，自动将父目录加入 sys.path。

### 5. Bash 文件名展开（Glob）
**问题：** 在 bash 循环中使用 `for f in *.txt` 时，glob 可能不展开。
**后果：** 变量得到的是字面字符串 `*.txt`，worker 报文件不存在。
**规则：** 用 `$(pwd)/dumas_instances/n20w80.00$i.txt` 显式指定路径。

### 6. 实验数据集划分
**问题：** 训练/验证集按图随机拆分，同一实例的解出现在两边。
**后果：** val_acc 被高估，模型泛化能力不真实。
**规则：** 必须按实例拆分，保证验证集实例在训练中完全不可见。
