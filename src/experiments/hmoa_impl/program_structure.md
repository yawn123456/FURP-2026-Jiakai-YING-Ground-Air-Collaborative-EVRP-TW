# 程序结构说明

## 总览

```
hmoa_impl/
│
├── model.py              # 数据模型层
├── evaluate.py           # 目标函数计算
├── initialization.py     # 初始解生成 (Algorithm 2)
├── repair.py             # 修复启发式 (Algorithm 4)
├── neighborhood.py       # 邻域算子 N1-N6
├── genetic_ops.py        # 遗传操作 (交叉 + 变异)
├── nsga2_utils.py        # NSGA-II 工具 (非支配排序 + 拥挤距离)
├── duplication.py        # 去重策略
├── pls.py                # Pareto 局部搜索 (Algorithm 3)
├── hmoa.py               # HMOA 主算法入口 (Algorithm 1)
├── benchmark.py          # 测试实例生成
├── visualize.py          # Pareto 前沿可视化
│
├── main.py               # 20客户标准测试
├── test_n50.py           # 50客户测试
├── quick_test.py         # 10客户快速验证
│
├── pareto_front.png              # 20客户 Pareto 图 (旧版)
├── pareto_front_n20_retest.png   # 20客户 Pareto 图 (优化后)
├── pareto_front_n50.png          # 50客户 Pareto 图
│
├── HMOA_RESULTS.md              # 20/50客户实验结果
├── experiment_reflection.md     # 实验反思总结 → 已被替换为 Weekly Progress Log
└── program_structure.md         # 本文件
```

---

## 各文件详细说明

### 核心算法层

#### `model.py` — 数据模型

| 类/函数 | 说明 |
|---------|------|
| `Customer` | 客户节点：坐标、硬时间窗 [a,b]、柔性时间窗 [e,l]、服务时间、无人机可达性 |
| `DroneSpec` | 无人机规格：ID、续航里程、单位成本 |
| `DroneFlight` | 一次无人机配送：`<drone_id, launch_idx, customer_id, land_idx>` |
| `Solution` | **5部分染色体解**：`truck_route + launch_idx + drone_customers + land_idx + drone_ids` |
| `ProblemInstance` | 问题实例：客户列表、无人机数、成本参数、距离矩阵预计算 |

**关键设计**：5部分染色体编码 (论文 Fig. 2)，每个无人机配送列 `<启动位置, 客户, 降落位置, 无人机ID>` 形成完整的协同配送方案。

---

#### `evaluate.py` — 目标函数 (Eq.1-3)

| 函数 | 说明 |
|------|------|
| `evaluate()` | 计算 **f1 (运输成本)** = 卡车距离×成本 + Σ(无人机飞行距离×成本)；**f2 (客户满意度)** = Σ φⱼ(到达时间) |
| `_compute_arrival_times()` | 模拟卡车路线时间线，计算每个卡车服务客户的到达时刻 |
| `_estimate_node_time()` | 估计卡车在路线中某位置的时间（用于无人机客户满意度估算） |

**客户满意度函数 φ(t)** (Eq.2)：
- 在 [aᵢ, bᵢ] 内到达 → 1.0
- 在 [eᵢ, aᵢ) 线性上升，在 (bᵢ, lᵢ] 线性下降
- 在 [eᵢ, lᵢ] 之外 → 0.0

---

#### `initialization.py` — 初始解生成 (Algorithm 2)

| 函数 | 说明 |
|------|------|
| `nearest_neighbor_tw()` | 带时间窗的最近邻启发式：每一步选择时间窗可行且最近的未访问节点 |
| `assign_nodes()` | **AssignNodes 算法**：随机划分客户为卡车组 Ct 和无人机组 Cd；用 NearestNeighborTW 生成卡车路线；贪心为 Cd 中每个节点找最优无人机飞行 |

**LTL (Lower Truck Limit)** (Eq.34)：
```
LTL = ceil((|C| - m) / (m + 1))
```

**Bug 修复**：非无人机可达节点从 Cd 移入 Ct，确保所有客户被分配。

---

#### `repair.py` — 修复启发式 (Algorithm 4)

| 函数 | 说明 |
|------|------|
| `repair()` | 输入不可行节点列表，为每个节点找所有可行`<启动, 客户, 降落, 无人机>`组合；用贪心法（最少候选优先 + 最低成本优先）逐一分配 |
| `_remove_assignments()` | 从解中移除与不可行节点相关的无人机配送 |

---

#### `neighborhood.py` — 邻域算子 (Sec. IV-D)

| 算子 | 类型 | 操作 |
|------|------|------|
| `n1_truck_to_drone()` | N1 | 删除卡车路线中最贵（两侧距离和最大）的无人机可达客户，用 Repair 重新分配给无人机 |
| `n2_drone_to_truck()` | N2 | 删除最贵的无人机配送，将客户插入卡车路线最便宜位置 |
| `n3_swap()` | N3 | 随机交换一个卡车客户和一个无人机客户 |
| `n4_2opt()` | N4 | 在卡车路线上随机选两个位置，反转中间段 |
| `n5_greedy_deletion_reinsertion()` | N5 | 删除最贵的无人机配送并用 Repair 重新分配 |
| `n6_random_deletion_reinsertion()` | N6 | 删除随机无人机配送并随机选择可行新位置 |

**用途分配**：
- N1, N2, N3 → **多模式变异**（改变卡车/无人机分配）
- N4, N5, N6 → **PLS 局部搜索**（固定分配优化路线）

---

#### `genetic_ops.py` — 遗传操作 (Sec. IV-E)

| 函数 | 说明 |
|------|------|
| `crossover()` | **单点交叉**：从两个父代染色体的卡车路线中各选一个客户节点交换；用 Repair 修复重复 |
| `multi_mode_mutation()` | **多模式变异**：根据当前状态（卡车节点数 ≤ LTL 或无人机配送数 < m）从 N1/N2/N3 中选一个执行 |

---

#### `nsga2_utils.py` — NSGA-II 工具

| 函数 | 说明 |
|------|------|
| `dominates()` | 判断解 a 是否支配解 b（f1 更小且 f2 更大） |
| `non_dominated_sort()` | 快速非支配排序 (NSGA-II)，返回各前沿的索引列表 |
| `crowding_distance()` | 计算同一前沿中解的拥挤距离 |
| `tournament_selection()` | 二元锦标赛选择：优先选 rank 小者，同 rank 选拥挤距离大者 |
| `select_next_population()` | NSGA-II 精英选择策略：逐前沿选取，最后部分按拥挤距离选取 |

---

#### `duplication.py` — 去重策略 (Sec. IV-F)

| 函数 | 说明 |
|------|------|
| `remove_duplication()` | 按目标值签名检测重复解；以概率 β=0.3 触发多模式变异（从 PF 选父代），否则调用 AssignNodes 生成新解 |

---

#### `pls.py` — Pareto 局部搜索 (Algorithm 3)

| 函数 | 说明 |
|------|------|
| `pareto_local_search()` | 输入 Pareto 前沿 PF；迭代 kmax=5 次；每轮对 PF 中每个解依次应用 N4、N5、N6；新非支配解加入 PF 和下一轮探索队列 PL |

**优化点**：仅探索新加入 PF 的解，避免重复计算。

---

#### `hmoa.py` — HMOA 主算法 (Algorithm 1)

| 函数 | 说明 |
|------|------|
| `update_pf()` | 用 F1 前沿更新外部 Pareto 存档 PF（保留非支配解） |
| `hmoa()` | **主循环**：初始化 → 每代(交叉→变异→合并→去重→非支配排序→精英选择→自适应 PLS 触发) → 输出最终 PF |

**自适应 PLS 触发**：
- `prob` 从 0 开始
- 每代增加 `1/max_iter`
- 当 `random() < prob` 时触发 PLS

---

### 辅助工具层

| 文件 | 功能 |
|------|------|
| `benchmark.py` | 生成随机 TSPTW 风格测试实例。支持指定客户数、无人机数、时间窗宽度、无人机可达比例、续航限制等 |
| `visualize.py` | 绘制 Pareto 前沿散点图，标注最优成本解、最优满意度解、最佳折衷解 |

### 入口脚本层

| 文件 | 用途 | 参数 |
|------|------|------|
| `main.py` | 20客户标准测试 + 对比优化前后结果 | pop=100, iter=100 |
| `test_n50.py` | 50客户大规模测试 | pop=100, iter=100 |
| `quick_test.py` | 10客户快速验证（<1s） | pop=50, iter=30 |

---

## 数据流

```
benchmark.py ──→ ProblemInstance
                      │
          initialization.py (AssignNodes)
                      │
                      ↓
                P[pop_size]  ← 初始种群
                      │
              ┌──────┴──────┐
              │ 遗传操作     │
              │ genetic_ops │
              │ crossover   │
              │ mutation    │
              └──────┬──────┘
                     │
                     ↓ Qt (子代)
              ┌──────┴──────┐
              │ Rt = Pt ∪ Qt│
              └──────┬──────┘
                     ↓
              duplication.py 去重
                     ↓
              nsga2_utils.py 非支配排序
                     ↓
              ┌──────┴──────┐
              │  prob < rand?│──否──→ Pt+1
              └──────┬──────┘
                     │是
                     ↓
              pls.py (PLS)
              N4, N5, N6
                     ↓
               更新 PF → Pt+1
                     │
              ┌──────┴──────┐
             循环直到 max_iter
                     ↓
              输出 Pareto Front
```

---

## 算法对应关系

| 论文组件 | 实现文件 | 算法/函数 |
|---------|---------|----------|
| Algorithm 1: HMOA 框架 | `hmoa.py` | `hmoa()` |
| Algorithm 2: AssignNodes | `initialization.py` | `assign_nodes()` |
| Algorithm 3: Pareto Local Search | `pls.py` | `pareto_local_search()` |
| Algorithm 4: Repair | `repair.py` | `repair()` |
| Section IV-D: 邻域算子 N1-N6 | `neighborhood.py` | `n1_` ~ `n6_` |
| Section IV-E: 遗传操作 | `genetic_ops.py` | `crossover()`, `multi_mode_mutation()` |
| Section IV-F: 去重 | `duplication.py` | `remove_duplication()` |
| NSGA-II (精英策略) | `nsga2_utils.py` | `non_dominated_sort()`, `crowding_distance()` |
| Eq.1-3: 目标函数 | `evaluate.py` | `evaluate()` |
| Eq.34: LTL | `initialization.py` | `math.ceil((n_cust - m) / (m + 1))` |

---

## 实验入口

```bash
# 快速验证 (10客户, <1s)
python quick_test.py

# 标准测试 (20客户, ~14s)
python main.py

# 大规模测试 (50客户, ~38s)
python test_n50.py
```
