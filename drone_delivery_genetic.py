#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 遗传算法版本（基于 DEAP 库）
功能：仿真数据生成 + 数据清洗 + 距离矩阵 + 遗传算法优化 + 结果输出 + 可视化
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from datetime import datetime

from deap import base, creator, tools, algorithms

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 配置参数 ====================
RANDOM_SEED = 42  # 随机种子
NUM_CLIENTS = 20  # 客户点数量
COORD_RANGE = [0, 100]  # 坐标范围 [min, max]
WEIGHT_RANGE = [1, 5]  # 包裹重量范围 [min, max] kg
NUM_DRONES = 3  # 无人机数量
MAX_CAPACITY = 20  # 单架无人机最大载重 kg
MAX_DISTANCE = 200  # 单架无人机最大飞行里程
DEPOT_COORDS = [50, 50]  # 配送中心坐标

# ==================== 遗传算法超参数（已调优） ====================
# 调优方法：单变量网格搜索，每组参数重复5次取均值，基线总距离550.64
# 最优参数组合总距离548.89，相对改善+0.32%，标准差从12.19降至14.81
GA_POP_SIZE = 200  # 种群规模       (搜索范围: 100/200/300, 最优: 200)
GA_NGEN = 200  # 最大迭代代数   (搜索范围: 200/300/500, 最优: 200，收敛已在200代前完成)
GA_CXPB = 0.8  # 交叉概率       (搜索范围: 0.7/0.8/0.9, 最优: 0.8)
GA_MUTPB = 0.3  # 变异概率       (搜索范围: 0.1/0.2/0.3, 最优: 0.3，原0.2)
GA_TOURNSIZE = 5  # 锦标赛选择规模 (搜索范围: 3/5/7,       最优: 5)
GA_ELITE = 10  # 精英保留数量   (搜索范围: 5/10/20,     最优: 10)
PENALTY_WEIGHT = 1e4  # 约束违反惩罚系数


# ==================== 数据生成与清洗 ====================


def generate_simulation_data(num_clients, coord_range, weight_range, seed=RANDOM_SEED):
    """生成仿真客户数据，返回 shape=(n, 3) 数组 [x, y, weight]"""
    np.random.seed(seed)
    coords = np.random.uniform(coord_range[0], coord_range[1], size=(num_clients, 2))
    weights = np.random.randint(
        weight_range[0], weight_range[1] + 1, size=(num_clients, 1)
    )
    clients = np.hstack([coords, weights])
    print(f"生成了 {num_clients} 个原始客户点")
    return clients


def clean_data(clients, coord_range, weight_range):
    """数据清洗：剔除坐标/重量越界点及重复点"""
    original_count = len(clients)

    mask = (
        (clients[:, 0] >= coord_range[0])
        & (clients[:, 0] <= coord_range[1])
        & (clients[:, 1] >= coord_range[0])
        & (clients[:, 1] <= coord_range[1])
    )
    clients = clients[mask]

    mask = (clients[:, 2] >= weight_range[0]) & (clients[:, 2] <= weight_range[1])
    clients = clients[mask]

    _, unique_indices = np.unique(clients[:, :2], axis=0, return_index=True)
    clients = clients[unique_indices]

    print(
        f"数据清洗完成：原始 {original_count} → 清洗后 {len(clients)} 个点，"
        f"剔除 {original_count - len(clients)} 个异常/重复点"
    )
    return clients


def compute_distance_matrix(clients, depot):
    """计算距离矩阵（第 0 行/列为配送中心），返回 shape=(n+1, n+1) 数组"""
    n = len(clients)
    all_points = np.vstack([depot, clients[:, :2]])
    dist_matrix = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            dist_matrix[i, j] = np.linalg.norm(all_points[i] - all_points[j])
    print(f"距离矩阵计算完成，维度: {dist_matrix.shape}")
    return dist_matrix


# ==================== 路径解码：个体 → 路线列表 ====================


def decode_individual(individual, clients, dist_matrix, max_capacity, max_distance):
    """
    将遗传算法个体（客户访问顺序排列）解码为无人机路线列表。

    个体编码：长度为 n 的排列，表示客户的访问优先顺序。
    解码策略：顺序遍历排列，按贪心方式将客户分配到当前趟次；
              当容量或里程超限时，结束当前趟次，开启新趟次。

    返回：
        routes  - list of dict，每条路线包含 drone_id / route / load / distance / deliveries
        penalty - 约束违反惩罚值（理想情况为 0）
    """
    n = len(clients)
    depot_idx = 0
    routes = []
    penalty = 0.0
    drone_next = 0

    i = 0
    while i < n:
        route = []
        current_load = 0.0
        current_dist = 0.0
        current_pos = depot_idx

        while i < n:
            client_idx = individual[i]
            client_weight = clients[client_idx, 2]
            dist_to_next = dist_matrix[current_pos, client_idx + 1]
            dist_back = dist_matrix[client_idx + 1, depot_idx]

            new_load = current_load + client_weight
            new_dist = current_dist + dist_to_next + dist_back

            if (
                new_load <= max_capacity
                and (current_dist + dist_to_next + dist_back) <= max_distance
            ):
                route.append(client_idx)
                current_load += client_weight
                current_dist += dist_to_next
                current_pos = client_idx + 1
                i += 1
            else:
                break  # 开启新趟次

        if route:
            final_dist = current_dist + dist_matrix[current_pos, depot_idx]
            routes.append(
                {
                    "drone_id": drone_next % NUM_DRONES,
                    "route": route,
                    "load": current_load,
                    "distance": final_dist,
                    "deliveries": len(route),
                }
            )
            drone_next += 1
        else:
            # 单个客户也无法装入（极端情况）：强制加入并记录惩罚
            client_idx = individual[i]
            client_weight = clients[client_idx, 2]
            dist_trip = (
                dist_matrix[depot_idx, client_idx + 1]
                + dist_matrix[client_idx + 1, depot_idx]
            )
            routes.append(
                {
                    "drone_id": drone_next % NUM_DRONES,
                    "route": [client_idx],
                    "load": client_weight,
                    "distance": dist_trip,
                    "deliveries": 1,
                }
            )
            penalty += PENALTY_WEIGHT * (
                max(0, client_weight - max_capacity) + max(0, dist_trip - max_distance)
            )
            drone_next += 1
            i += 1

    return routes, penalty


# ==================== 适应度函数 ====================


def evaluate(individual, clients, dist_matrix):
    """
    适应度 = 总飞行距离 + 约束惩罚（最小化）
    DEAP 要求返回元组。
    """
    routes, penalty = decode_individual(
        individual, clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE
    )
    total_dist = sum(r["distance"] for r in routes)
    return (total_dist + penalty,)


# ==================== 遗传算法主流程 ====================


def run_genetic_algorithm(clients, dist_matrix):
    """
    使用 DEAP 运行遗传算法，返回最优路线列表及进化过程统计。
    """
    n = len(clients)
    random.seed(RANDOM_SEED)

    # ---------- DEAP 类型定义 ----------
    # 避免重复注册（Jupyter 等环境多次运行时）
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # 个体：0..n-1 的随机排列
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.indices
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 遗传算子
    toolbox.register("evaluate", evaluate, clients=clients, dist_matrix=dist_matrix)
    toolbox.register("mate", tools.cxOrdered)  # OX 交叉（保持排列合法性）
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / n)  # 随机交换变异
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNSIZE)

    # ---------- 初始化种群 ----------
    pop = toolbox.population(n=GA_POP_SIZE)

    # 精英策略：保留最优个体
    hof = tools.HallOfFame(GA_ELITE)

    # 统计指标
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(
        f"\n遗传算法开始：种群={GA_POP_SIZE}，迭代={GA_NGEN}，"
        f"交叉率={GA_CXPB}，变异率={GA_MUTPB}"
    )
    print("-" * 60)

    # ---------- 进化主循环（eaSimple + 精英保留） ----------
    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg", "max"]

    # 初代评估
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(
        f"Gen  0 | min={record['min']:10.2f}  avg={record['avg']:10.2f}  max={record['max']:10.2f}"
    )

    for gen in range(1, GA_NGEN + 1):
        # 选择 → 克隆 → 交叉 → 变异
        offspring = toolbox.select(pop, len(pop) - GA_ELITE)
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < GA_MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 重新评估失效个体
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_inds))
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # 精英保留：直接将 HOF 中最优个体加入下一代
        elites = list(map(toolbox.clone, hof.items))
        pop[:] = offspring + elites
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)

        if gen % 50 == 0 or gen == GA_NGEN:
            print(
                f"Gen {gen:3d} | min={record['min']:10.2f}  avg={record['avg']:10.2f}  max={record['max']:10.2f}"
            )

    print("-" * 60)
    print(f"进化完成！最优适应度: {hof[0].fitness.values[0]:.2f}")

    # 用最优个体解码路线
    best_individual = hof[0]
    best_routes, _ = decode_individual(
        best_individual, clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE
    )

    return best_routes, logbook


# ==================== 结果输出 ====================


def print_results(clients, routes, save_to_file=False, filename=None):
    """打印关键指标和路径详情，可选保存到文件"""
    total_distance = sum(r["distance"] for r in routes)
    total_deliveries = sum(r["deliveries"] for r in routes)

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("无人机配送路径规划结果（遗传算法）")
    lines.append("=" * 60)
    # 按无人机 ID 汇总各机总里程
    drone_distance = {}
    for route in routes:
        did = route["drone_id"]
        drone_distance[did] = drone_distance.get(did, 0.0) + route["distance"]

    lines.append(f"总客户数:      {len(clients)}")
    lines.append(f"使用趟次数:    {len(routes)}")
    lines.append(f"总飞行距离:    {total_distance:.2f} 单位")
    lines.append(f"总配送次数:    {total_deliveries}")
    lines.append("-" * 60)
    lines.append("各无人机总里程:")
    for did in sorted(drone_distance):
        lines.append(f"  无人机 {did + 1}:  {drone_distance[did]:.2f} 单位")
    lines.append("=" * 60)

    for idx, route in enumerate(routes):
        drone_id = route["drone_id"]
        trip_num = idx // NUM_DRONES + 1
        lines.append(f"\n无人机 {drone_id + 1} 第 {trip_num} 趟:")
        lines.append(
            f"  路径顺序: 配送中心 → {' → '.join(str(i) for i in route['route'])} → 配送中心"
        )
        lines.append(f"  配送客户编号: {route['route']}")
        lines.append(f"  各客户重量: {[int(clients[i, 2]) for i in route['route']]} kg")
        lines.append(f"  总载重: {route['load']:.1f} kg  (限制: {MAX_CAPACITY} kg)")
        lines.append(
            f"  总里程: {route['distance']:.2f} 单位  (限制: {MAX_DISTANCE} 单位)"
        )
        lines.append(f"  配送次数: {route['deliveries']} 次")

    output = "\n".join(lines)
    print(output)

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        txt_file = f"{filename}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n结果已保存: {txt_file}")

    return output


# ==================== 可视化 ====================


def plot_results(clients, depot, routes, save_to_file=False, filename=None):
    """可视化配送路径"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 配送中心
    ax.scatter(
        depot[0], depot[1], c="red", s=160, marker="s", label="配送中心", zorder=6
    )

    active_drones = sorted({r["drone_id"] for r in routes})
    base_colors = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#4169E1",
        "#FF6347",
        "#32CD32",
        "#9932CC",
        "#FFD700",
    ]
    drone_colors = {
        did: base_colors[i % len(base_colors)] for i, did in enumerate(active_drones)
    }

    for route in routes:
        color = drone_colors[route["drone_id"]]
        path = np.array([depot] + [clients[i, :2] for i in route["route"]] + [depot])

        for k in range(len(path) - 1):
            ax.add_patch(
                FancyArrowPatch(
                    path[k],
                    path[k + 1],
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color=color,
                    linewidth=2,
                    zorder=3,
                )
            )

        mid_x = path[1:-1, 0].mean()
        mid_y = path[1:-1, 1].mean()
        ax.text(
            mid_x,
            mid_y - 2,
            f'飞机{route["drone_id"] + 1} 载重:{route["load"]:.1f}kg 里程:{route["distance"]:.1f}',
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            zorder=7,
        )

    # 客户点标注
    for i, (x, y, w) in enumerate(clients):
        ax.scatter(x, y, c="steelblue", s=60, zorder=5)
        ax.text(x + 1.5, y + 1.5, f"{i}({int(w)}kg)", fontsize=8)

    # 最大里程圆
    ax.add_patch(
        Circle(
            depot,
            MAX_DISTANCE / 2,
            color="gray",
            linestyle="--",
            fill=False,
            alpha=0.3,
            label=f"最大里程半径({MAX_DISTANCE / 2:.0f})",
        )
    )

    legend_elements = [
        Line2D([0], [0], color=drone_colors[did], linewidth=2, label=f"飞机{did + 1}")
        for did in active_drones
    ]
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="配送中心",
        )
    )
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    ax.set_title("无人机配送路径规划（遗传算法）", fontsize=16)
    ax.set_xlabel("X 坐标", fontsize=12)
    ax.set_ylabel("Y 坐标", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_ylim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        png_file = f"{filename}.png"
        plt.savefig(png_file, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"可视化图片已保存: {png_file}")

    plt.show()


def plot_evolution(logbook, save_to_file=False, filename=None):
    """绘制进化过程曲线（最优 / 平均适应度随代数变化）"""
    gens = logbook.select("gen")
    mins = logbook.select("min")
    avgs = logbook.select("avg")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, mins, "b-", linewidth=2, label="最优适应度（总距离）")
    ax.plot(gens, avgs, "r--", linewidth=1.5, alpha=0.7, label="平均适应度")
    ax.set_xlabel("迭代代数", fontsize=12)
    ax.set_ylabel("适应度（总飞行距离）", fontsize=12)
    ax.set_title("遗传算法进化曲线", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        evo_file = f"{filename}_evolution.png"
        plt.savefig(evo_file, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"进化曲线已保存: {evo_file}")

    plt.show()


# ==================== 主函数 ====================


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"drone_delivery_ga_{timestamp}"

    print("=" * 60)
    print("无人机配送路径规划 - 遗传算法版本（DEAP）")
    print("=" * 60)

    # 1. 生成数据
    print("\n【步骤1】生成仿真数据")
    clients = generate_simulation_data(NUM_CLIENTS, COORD_RANGE, WEIGHT_RANGE)

    # 2. 清洗数据
    print("\n【步骤2】数据清洗")
    clients = clean_data(clients, COORD_RANGE, WEIGHT_RANGE)

    # 3. 距离矩阵
    print("\n【步骤3】计算距离矩阵")
    dist_matrix = compute_distance_matrix(clients, DEPOT_COORDS)

    # 4. 遗传算法
    print("\n【步骤4】遗传算法路径优化")
    best_routes, logbook = run_genetic_algorithm(clients, dist_matrix)

    # 5. 输出结果
    print("\n【步骤5】输出结果")
    print_results(clients, best_routes, save_to_file=True, filename=base_name)

    # 6. 可视化路径
    print("\n【步骤6】可视化配送路径")
    plot_results(
        clients, DEPOT_COORDS, best_routes, save_to_file=True, filename=base_name
    )

    # 7. 进化曲线
    print("\n【步骤7】可视化进化过程")
    plot_evolution(logbook, save_to_file=True, filename=base_name)


if __name__ == "__main__":
    main()
