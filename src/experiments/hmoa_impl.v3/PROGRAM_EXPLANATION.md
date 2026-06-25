# HMOA 程序架构说明

> **论文**: Luo et al., "Hybrid Multi-Objective Optimization Approach With Pareto Local Search for Collaborative Truck-Drone Routing Problems Considering Flexible Time Windows", IEEE TITS, 2022.

---

## 文件结构与模块关系

```
hmoa_impl/
├── model.py           # 问题模型：数据结构定义
├── evaluate.py        # 解评估：目标函数 f1(成本) + f2(满意度)
├── initialization.py  # 初始化：贪心初始解生成 (Algorithm 2)
├── neighborhood.py    # 邻域算子：N1~N6 六种启发式
├── repair.py          # 修复启发式 (Algorithm 4)
├── genetic_ops.py     # 遗传算子：交叉 + 多模式变异
├── nsga2_utils.py     # NSGA-II 工具：非支配排序、拥挤距离、锦标赛选择
├── duplication.py     # 去重策略 (RemoveDuplication)
├── pls.py             # Pareto局部搜索 (Algorithm 3)
├── hmoa.py            # HMOA 主框架 (Algorithm 1)
├── benchmark.py       # 基准实例生成 (Dumas TSPTW 扩展)
├── visualize.py       # 可视化工具
├── main.py            # 入口：运行基准测试 + 消融实验
├── quick_test.py      # 快速验证 (n=10)
├── test_n50.py        # 中规模测试 (n=50)
├── benchmark_runner.py # 完整基准实验运行器
├── run_dumas_benchmark.py # Dumas 基准快速运行
└── run_n80.py         # n80 单独运行
```

---

## 1. 问题模型 — `model.py`

### 核心数据结构

```
Customer               Solution               ProblemInstance
├── id                 ├── truck_route[5]      ├── customers[]
├── x, y               │   Part 1: [0,1,5,2,   ├── num_drones
├── a, b (硬时间窗)    │            3,0]        ├── truck_dist[][] (曼哈顿)
├── e, l (柔性时间窗)  ├── launch_idx[]         ├── drone_dist[][] (欧几里得)
├── service_time       │   Part 2              ├── truck_cost_per_km (25.0)
├── drone_eligible     ├── drone_customers[]    ├── drone_cost_per_km (1.0)
│                      │   Part 3              ├── drone_endurance
│  satisfaction(t):    ├── land_idx[]           ├── wbli, wbui
│    [e,a): 线性上升   │   Part 4              └── node_map
│    [a,b]: =1.0       └── drone_ids[]
│    (b,l]: 线性下降       Part 5
│    否则:  =0.0
```

### 5 部分染色体编码

```
例子: <1, 3, 2, 1> 表示无人机1从卡车路线位置1(仓库)发射，
      服务客户3，在位置2(客户5)回收。

Part 1: [0, 5, 8, 3, 0]  ← 卡车路线（位置索引）
Part 2: [1, 0, 2, 1, 3]  ← 无人机发射位置
Part 3: [3, 7, 6, 4, 2]  ← 无人机服务客户
Part 4: [2, 4, 3, 3, 4]  ← 无人机回收位置
Part 5: [0, 1, 0, 1, 0]  ← 无人机编号

  ┌──────┬──────┬──────┬──────┬──────┐
  │发射位 │客户ID │回收位 │无人机 │ 含义 │
  ├──────┼──────┼──────┼──────┼──────┤
  │  1   │  3   │  2   │  0   │ 无人机0：从位置1发射→服务客户3→在位置2回收  │
  │  0   │  7   │  4   │  1   │ 无人机1：从仓库发射→服务客户7→在位置4回收    │
  └──────┴──────┴──────┴──────┴──────┘
```

### 柔性时间窗满意度函数 (Eq. 2)

```
满意度 φ(t)
  1.0 ┤          ┌──────┐
      │         /        \
  0.5 ┤        /          \
      │       /            \
  0.0 ┤──────┘              └──────
      ├─────┼─────┼─────┼────────→ 到达时间 t
      e     a     b     l
      ←容忍→ ←核心窗口→ ←容忍→
```

---

## 2. 解评估 — `evaluate.py`

**目标函数**（Eq. 1, 3）:

| 目标 | 公式 | 方向 |
|------|------|------|
| **f1** | 卡车距离×25 + Σ(无人机飞行距离×1) | ↓ 最小化 |
| **f2** | Σ 每个客户的满意度 φ(arrival_time) | ↑ 最大化 |

**关键逻辑**:
1. 沿卡车路线模拟行驶，累加旅行时间，记录到达时间
2. 无人机客户：`到达时间 = 发射点卡车时间 + 无人机飞行距离`
3. 对每个客户调用 `Customer.satisfaction(arrival_time)` 计算满意度

---

## 3. 初始化 — `initialization.py` (Algorithm 2: AssignNodes)

```
输入: C(所有客户), m(无人机数), n(种群大小)
输出: P(初始种群)

过程:
1. 计算 LTL = ⌈(|C| - m) / (m + 1)⌉  (Eq. 34)
   → 卡车路线最少客户数

2. while |P| < n:
   a) 随机分割客户: Ct(卡车组, ≥LTL个), Cd(无人机组)
      → Cd 只能包含 drone_eligible 客户

   b) NearestNeighborTW(Ct)
      → 最近邻+时间窗启发式生成卡车路线

   c) 贪心分配 Cd 中客户到无人机:
      for each drone d:
        for each unassigned customer j in Cd:
          寻找所有可行的 <发射位置i, 客户j, 回收位置k>
          选择成本最低的分配
      若无法分配 → 移入卡车路线，重生成路线

   d) 确保所有客户被服务
      添加到 P
```

**复杂度**: O(n × m × |Ct|³ × |Cd|)，对大规模实例 (n≥60) 是主要瓶颈。

---

## 4. 邻域算子 — `neighborhood.py` (Section IV-D)

### 六种启发式邻域

| 算子 | 名称 | 操作 | 用途 |
|------|------|------|------|
| **N1** | Truck→Drone | 删除卡车路线中最昂贵的客户，用无人机服务 | 变异, 增加无人机使用 |
| **N2** | Drone→Truck | 删除最昂贵的无人机分配，插入卡车路线 | 变异, 减少无人机使用 |
| **N3** | Swap | 交换一个卡车客户和一个无人机客户 | 变异, 探索混合方案 |
| **N4** | 2-Opt | 反转卡车路线的子段 | PLS, 优化卡车路线 |
| **N5** | Greedy-Deletion-Reinsertion | 删除最贵无人机分配，用Repair重分配 | PLS, 优化无人机分配 |
| **N6** | Random-Deletion-Reinsertion | 随机删除无人机分配，随机重分配 | PLS, 多样化探索 |

### 算子分配

```
N1, N2, N3 → Multi-Mode Mutation (多样性)
N4, N5, N6 → Pareto Local Search  (收敛性)
```

---

## 5. 修复启发式 — `repair.py` (Algorithm 4)

**作用**: 当邻域操作导致解不可行时，尝试修复。

```
输入: 不可行客户节点列表 Cin, 当前解 p
输出: 修复后的解 p'

过程:
1. 从 p 中删除与 Cin 相关的所有无人机分配
2. 对每个不可行客户 j:

   枚举所有可行的 <发射位置i, 客户j, 回收位置k, 无人机d>:
     - 满足续航约束: d_launch + d_return ≤ endurance
     - 不与现有分配冲突

3. 贪心分配:
   - 选择"选择最少"的客户优先分配
   - 为该客户选择成本最低的飞行方案
   - 移除冲突的候选方案
   - 重复直到所有客户被分配

4. 若无法完全修复 → 返回原始解
```

**复杂度**: O(m × |Ct|² × |Cin|)，对大规模实例和大不可行集很慢。

---

## 6. 遗传算子 — `genetic_ops.py` (Section IV-E)

### 6.1 交叉 (One-Point Crossover)

```
论文规定: "a random and COMMON crossover position will be selected
          from Part 1 or Part 2 in two chromosomes"

过程:
1. 选择共同交叉位置 pos (Part 1: 卡车路线)
2. 交换两个父代在 pos 处的客户节点
3. 检测重复节点
4. 若有重复 → Repair 修复
5. 返回两个子代
```

### 6.2 多模式变异 (Multi-Mode Mutation)

```
规则 (论文):
- 卡车路线长度 < LTL → 仅 N2 或 N3 (不能再减少卡车客户)
- 无人机分配数 < m  → 仅 N1 或 N3 (不能再减少无人机客户)
- 其他情况         → 等概率选择 N1, N2, N3

N1, N2, N3 被分配相同触发概率 (论文: "same rates")
```

---

## 7. NSGA-II 工具 — `nsga2_utils.py`

### 7.1 支配关系

```
解A 支配 解B ⟺ (f1_A ≤ f1_B) ∧ (f2_A ≥ f2_B) ∧ (f1_A < f1_B ∨ f2_A > f2_B)
             成本更低        满意度更高       至少一个严格更好
```

### 7.2 快速非支配排序 (Fast Non-Dominated Sort)

```
1. 对每个解 i，计算:
   - domination_count[i]: 支配 i 的解数
   - dominated_set[i]: 被 i 支配的解集合

2. Front 0: domination_count = 0 的解

3. 构建后续 front:
   for each p in current_front:
     for each q in dominated_set[p]:
       domination_count[q] -= 1
       if domination_count[q] == 0 → 加入 next_front
```

### 7.3 拥挤距离 (Crowding Distance)

```
作用: 衡量解的密度，用于保持种群多样性

对每个目标:
  1. 按目标值排序
  2. 边界点 → distance = ∞ (始终保留)
  3. 中间点 → distance += (next_obj - prev_obj) / (max_obj - min_obj)
```

### 7.4 二元锦标赛选择

```
从种群中随机选2个解:
  - 优先选 front rank 更小的
  - rank 相同时选 crowding distance 更大的
```

---

## 8. 去重策略 — `duplication.py` (Section IV-F)

```
问题: 复杂约束 + 启发式常返回不变解 → 种群中积累重复解

策略: RemoveDuplication(Rt, PF)
  对每个重复解:
    以概率 restart_rate (α=0.3):
      → Multi-Mode Mutation 扰动随机PF解
    以概率 1-α (0.7):
      → AssignNodes 创建全新初始解

论文动机:
  - 变异扰动: 搜索PF解的大邻域空间
  - 新解: 有利于多样性，但远离PF
  - 只改变重复解 → 保持算法鲁棒性
```

---

## 9. Pareto 局部搜索 — `pls.py` (Algorithm 3)

```
论文核心策略:
  "New solutions may be further improved, thereby only new
   solutions to PF would be explored in the local search."

参数: kmax=5 (论文: "to reduce computational time")

过程:
1. PF' ← 当前代新增到PF的解 (仅探索"新"解)

2. while k ≤ kmax and PF' ≠ ∅:
     PL ← ∅
     for each p in PF':
       for each Ni in {N4, N5, N6}:
         p' ← Ni(p)
         if p' 支配 p:          ← 论文条件 (line 8)
           Update2(p', PF)      ← 若不被PF支配，加入PF
           if p' 加入PF:
             Update2(p', PL)    ← 加入下一轮探索列表
     PF' ← PL
     k += 1

Update2 算子:
  - 若 p' 不被 target 中任何解支配 → 添加 p'
  - 删除 target 中被 p' 支配的所有解
```

---

## 10. HMOA 主框架 — `hmoa.py` (Algorithm 1)

```
Algorithm 1: Framework of HMOA
═══════════════════════════════════════════

初始化:
  prob ← 0, PF ← ∅
  LTL = ⌈(|C| - m) / (m + 1)⌉
  P1 ← AssignNodes(C, m, n)    ← 贪心初始种群

主循环 (t = 1 → iter):
  ┌─────────────────────────────────────────┐
  │ Step 4:   Qt ← GeneticOperation(Pt)      │  交叉(锦标赛选择)+变异
  │ Step 5:   Rt ← Qt ∪ Pt                  │  合并种群
  │ Step 6:   Rt ← RemoveDuplication(Rt, PF) │  去重+替换重复解
  │ Step 7:   NonDominatedSort(Rt)           │  NSGA-II 排序
  │ Step 8-19: 构建 Pt+1                     │  精英选择
  │           if i=1:                         │
  │             PF ← Update1(PF, F1)          │  更新外部PF
  │             if rand < prob:              │  自适应PLS触发
  │               PF ← PLS(PF)               │  prob = t/iter
  │               F1 ← PF                    │  用PF替换F1
  │ Step 20:  prob += 1/iter                │  递增触发概率
  └─────────────────────────────────────────┘

最终: 合并 P + PF，提取非支配解 = 最终PF

自适应PLS策略 (论文核心):
  - prob 从 0 开始，每代增加 1/iter
  - 早期(prob小): 注重多样性，PLS少触发
  - 后期(prob→1.0): 注重收敛，PLS频繁触发
```

---

## 11. 基准实例 — `benchmark.py`

### 实例生成 (Dumas TSPTW 扩展)

```
论文设置:
  - 扩展自 Dumas et al. (1995) TSPTW 基准
  - 4个规模 × 5个数据集 = 20个实例
  - n20w80, n40w80, n60w80, n80w80
    (n=客户数, w=时间窗宽度)

生成步骤:
1. 随机生成客户坐标 (x,y) ∈ [0,100]
2. 用最近邻路线计算到达时间
3. 时间窗居中: a = arrival - width/2, b = a + width
4. 柔性窗: e = a - wbli×(b-a), l = b + wbui×(b-a)
5. 85%客户为 drone-eligible
6. 续航: 35%分位数 (Moshref-Javadi et al. 方法)
7. 服务时间: uniform(5, 15)
8. 卡车距离: Manhattan; 无人机距离: Euclidean
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| truck_cost_per_km | 25.0 | 卡车成本 = 25×无人机成本 |
| drone_cost_per_km | 1.0 | 基准单位 |
| drone_eligible_ratio | 0.85 | 85%客户可被无人机服务 |
| endurance_ratio | 0.35 | 续航覆盖35%的可行飞行 |
| wbli, wbui | 0.2 | 柔性时间窗容忍参数 |

---

## 12. 可视化 — `visualize.py`

- `plot_pareto_front()`: 绘制 Pareto 前沿 (f1 vs f2)
  - 蓝色线+点: 非支配解
  - 红色星: 最佳折衷解
  - 绿色方块: 最优成本解
  - 橙色三角: 最优满意度解
- `plot_convergence()`: 绘制收敛曲线 (成本/满意度/PF规模 vs 代数)

---

## 13. 入口文件

| 文件 | 用途 | 配置 |
|------|------|------|
| `main.py` | 主入口：n20 基准 + HMOA vs HMOA-noLS | pop=200, gen=200 |
| `quick_test.py` | 快速验证 (n=10) | pop=200, gen=200 |
| `test_n50.py` | 中规模测试 (n=50) | pop=200, gen=200 |
| `benchmark_runner.py` | 完整基准 (20实例×15运行) | 论文参数 |
| `run_dumas_benchmark.py` | Dumas快速基准 (4实例×1运行) | 论文参数 |
| `run_n80.py` | n80单独运行 | 论文参数 |

---

## 14. 完整数据流

```
                    ┌──────────────┐
                    │ ProblemInstance │ (客户, 无人机, 距离矩阵)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  assign_nodes │ Algorithm 2: 贪心初始种群
                    └──────┬───────┘
                           │ P (200个解)
                    ┌──────▼───────┐
                    │   evaluate    │ f1(成本), f2(满意度)
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │   HMOA 主循环 (200代)    │ Algorithm 1
              │                        │
              │  ┌──────────────────┐   │
              │  │ 二元锦标赛选择    │   │
              │  │ One-Point交叉    │   │ → Qt (子代)
              │  │ Multi-Mode变异   │   │
              │  └──────────────────┘   │
              │           │             │
              │  Rt = Qt ∪ Pt           │
              │           │             │
              │  RemoveDuplication      │
              │  NonDominatedSort       │
              │  CrowdingDistance       │
              │           │             │
              │  ┌───────▼────────┐     │
              │  │ Update1(PF,F1) │     │
              │  │ rand<prob?     │     │
              │  │  PLS(PF)      │─────┤→ Algorithm 3
              │  │  F1 ← PF     │     │   N4,N5,N6
              │  └───────┬────────┘     │
              │           │             │
              │  Pt+1 ← 精英选择        │
              │  prob += 1/200          │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │  最终 PF      │ 非支配排序 → Pareto前沿
                    └──────────────┘
```
