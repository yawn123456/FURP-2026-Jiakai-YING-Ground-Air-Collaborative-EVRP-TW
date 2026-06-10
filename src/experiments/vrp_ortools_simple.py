"""
车辆路径问题 (VRP) - OR-Tools 实现
Week 1 Lab: Baseline Smoke Test
- 4 辆车，容量约束
- 欧氏距离矩阵
- 路径可视化
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(num_customers=20, num_vehicles=4, capacity=100, seed=42):
    """生成随机 VRP 测试数据"""
    rng = np.random.default_rng(seed)

    # 仓库在原点，客户坐标在 [-50, 50] 范围内
    depot = (0.0, 0.0)
    locations = [depot]  # index 0 = depot
    demands = [0]        # 仓库需求为 0

    for _ in range(num_customers):
        x = rng.uniform(-50, 50)
        y = rng.uniform(-50, 50)
        locations.append((x, y))
        demands.append(int(rng.integers(5, 30)))

    # 计算距离矩阵（欧氏距离，四舍五入为整数）
    n = len(locations)
    distance_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                dx = locations[i][0] - locations[j][0]
                dy = locations[i][1] - locations[j][1]
                distance_matrix[i][j] = int(round(math.hypot(dx, dy)))

    data = {
        'distance_matrix': distance_matrix.tolist(),
        'demands': demands,
        'num_vehicles': num_vehicles,
        'vehicle_capacity': capacity,
        'depot': 0,
        'locations': locations,
        'num_customers': num_customers,
    }
    return data


def print_solution(data, manager, routing, solution):
    """打印求解结果"""
    total_distance = 0
    total_load = 0
    routes = []

    print(f"{'='*60}")
    print(f"  VRP OR-Tools 求解结果")
    print(f"{'='*60}")
    print(f"  车辆数: {data['num_vehicles']}")
    print(f"  车辆容量: {data['vehicle_capacity']}")
    print(f"  客户数: {data['num_customers']}")
    print(f"{'='*60}")

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        route_nodes = []

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_nodes.append(node_index)
            route_load += data['demands'][node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        route_nodes.append(manager.IndexToNode(index))  # depot
        routes.append(route_nodes)

        total_distance += route_distance
        total_load += route_load

        print(f"\n  车辆 {vehicle_id + 1} 路线:")
        print(f"    路径: ", end="")
        for i, n in enumerate(route_nodes):
            if n == 0:
                print("Depot", end="")
            else:
                print(f"C{n}", end="")
            if i < len(route_nodes) - 1:
                print(" → ", end="")
        print(f"\n    行驶距离: {route_distance}")
        print(f"    载重: {route_load}/{data['vehicle_capacity']}")

    print(f"\n{'='*60}")
    print(f"  总行驶距离: {total_distance}")
    print(f"  总载重: {total_load}")
    print(f"{'='*60}")

    return routes


def plot_solution(data, routes, total_distance, save_path=None):
    """可视化车辆路径"""
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
              '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

    plt.figure(figsize=(10, 8))

    # 绘制仓库
    depot = data['locations'][0]
    plt.scatter(*depot, c='black', s=200, marker='s', zorder=5)
    plt.annotate('Depot', depot, xytext=(5, 5),
                 textcoords='offset points', fontsize=12, fontweight='bold')

    # 绘制客户点
    xs = [loc[0] for loc in data['locations'][1:]]
    ys = [loc[1] for loc in data['locations'][1:]]
    plt.scatter(xs, ys, c='#2c3e50', s=80, zorder=4)

    # 标注客户编号和需求量
    for i in range(1, len(data['locations'])):
        x, y = data['locations'][i]
        d = data['demands'][i]
        plt.annotate(f'C{i}({d})', (x, y), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)

    # 绘制每条路径
    for v, route in enumerate(routes):
        color = colors[v % len(colors)]
        route_xs = [data['locations'][n][0] for n in route]
        route_ys = [data['locations'][n][1] for n in route]
        plt.plot(route_xs, route_ys, color=color, linewidth=2,
                 marker='o', markersize=6, label=f'Vehicle {v + 1}')

    plt.title(f'VRP Solution (OR-Tools) — Total Distance: {total_distance}',
              fontsize=14, fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图片已保存至: {save_path}")

    plt.show()


def main():
    # ========== 1. 创建数据 ==========
    print("正在生成 VRP 测试数据...")
    data = create_data_model(num_customers=20, num_vehicles=4, capacity=100, seed=42)

    # ========== 2. 创建路由模型 ==========
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),
        data['num_vehicles'],
        data['depot']
    )
    routing = pywrapcp.RoutingModel(manager)

    # ========== 3. 注册距离回调 ==========
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # ========== 4. 添加容量约束 ==========
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # 无容量 slack
        [data['vehicle_capacity']] * data['num_vehicles'],  # 每辆车容量
        True,  # 从 0 开始累计
        'Capacity'
    )

    # ========== 5. 设置搜索策略 ==========
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # 使用引导局部搜索（Guided Local Search）以获得更优解
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 5  # 搜索时间限制 5 秒

    # ========== 6. 求解 ==========
    print("正在使用 OR-Tools 求解 VRP...")
    start_time = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    runtime = time.time() - start_time

    # ========== 7. 输出结果 ==========
    print(f"\n运行时: {runtime:.4f} 秒")
    print(f"状态: {'FEASIBLE' if solution else 'INFEASIBLE'}")

    if solution:
        routes = print_solution(data, manager, routing, solution)
        total_distance = solution.ObjectiveValue()
        plot_solution(data, routes, total_distance, save_path='vrp_ortools_route.png')
    else:
        print("未找到可行解。")

    # ========== 8. 输出 Week 1 报告摘要 ==========
    print(f"\n{'='*60}")
    print(f"  Week 1 Lab — 实验报告摘要")
    print(f"{'='*60}")
    print(f"  OS          : Windows 11")
    print(f"  Python      : {__import__('sys').version.split()[0]}")
    print(f"  OR-Tools    : {__import__('ortools').__version__}")
    print(f"  车辆数      : {data['num_vehicles']}")
    print(f"  客户数      : {data['num_customers']}")
    print(f"  容量        : {data['vehicle_capacity']}")
    print(f"  可行解      : {'是' if solution else '否'}")
    print(f"  目标值      : {solution.ObjectiveValue() if solution else 'N/A'}")
    print(f"  搜索策略    : PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH")
    print(f"  运行时间    : {runtime:.4f} 秒")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
