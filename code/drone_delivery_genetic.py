#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 遗传算法版本（基于 DEAP 库）
功能：仿真数据生成 + 数据清洗 + 距离矩阵 + 遗传算法优化 + 结果输出 + 可视化
"""

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from datetime import datetime

from deap import base, creator, tools

from common import (
    generate_simulation_data, clean_data, compute_distance_matrix,
    RANDOM_SEED, NUM_CLIENTS, COORD_RANGE, WEIGHT_RANGE,
    MAX_CAPACITY, MAX_DISTANCE, DRONE_SPEED, DEPOT_COORDS,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 遗传算法超参数 ====================
# 调优方法：单变量网格搜索，每组参数重复5次取均值，基线总距离550.64
# 最优参数组合总距离548.89，相对改善+0.32%，标准差从12.19降至14.81
GA_POP_SIZE = 200  # 种群规模       (搜索范围: 100/200/300, 最优: 200)
GA_NGEN = 200  # 最大迭代代数   (搜索范围: 200/300/500, 最优: 200，收敛已在200代前完成)
GA_CXPB = 0.8  # 交叉概率       (搜索范围: 0.7/0.8/0.9, 最优: 0.8)
GA_MUTPB = 0.3  # 变异概率       (搜索范围: 0.1/0.2/0.3, 最优: 0.3，原0.2)
GA_TOURNSIZE = 5  # 锦标赛选择规模 (搜索范围: 3/5/7,       最优: 5)
GA_ELITE = 10  # 精英保留数量   (搜索范围: 5/10/20,     最优: 10)
PENALTY_WEIGHT = 1e4  # 约束违反惩罚系数


class FleetScheduler:
    """
    机队时间轴调度器。
    追踪每架无人机"最早可出发时刻"，按贪心策略分配任务给最早空闲的飞机。
    """

    def __init__(self, num_drones: int, speed: float):
        self.num_drones = num_drones
        self.speed = speed
        self.available_at = [0.0] * num_drones   # 每架无人机的最早就绪时刻，初始值均为0.0，表示所有无人机初始时刻均空闲可用。

    def assign(self, flight_distance: float):
        """
        为一趟飞行分配飞机。
        选择最早空闲的无人机 → 计算飞行时间 → 更新其就绪时刻。
        返回 (drone_id, depart_time, arrive_time)
        """
        drone_id = int(np.argmin(self.available_at))  # 最早空闲的飞机编号
        depart = self.available_at[drone_id]
        flight_time = flight_distance / self.speed      # 飞行时间 = 距离 / 速度
        arrive = depart + flight_time
        self.available_at[drone_id] = arrive            # 更新该飞机的下次可用时刻
        return drone_id, depart, arrive

    def reset(self):
        """重置所有飞机时间轴"""
        self.available_at = [0.0] * self.num_drones


def decode_individual(individual, clients, dist_matrix, num_drones,
                      max_capacity, max_distance, drone_speed, penalty_weight):
    """
    将遗传算法个体（客户访问顺序排列）解码为具体飞行路线。

    过程：按个体顺序逐一尝试将客户装入当前趟次，
    满足载重&里程约束则加入，否则结束当前趟次、开启新趟次。
    若某客户单独也无法满足约束，则强制分配并施加惩罚项。
    """
    n = len(clients)
    depot_idx = 0                                    # 配送中心在距离矩阵中的索引
    scheduler = FleetScheduler(num_drones, drone_speed)
    routes = []
    penalty = 0.0                                    # 累积约束违反惩罚

    i = 0
    while i < n:
        # 开始一趟新飞行
        route = []
        current_load = 0.0
        current_dist = 0.0
        current_pos = depot_idx                       # 从配送中心出发

        while i < n:
            client_idx = individual[i]                 # 按个体编码顺序取下一客户
            client_weight = clients[client_idx, 2]
            dist_to_next = dist_matrix[current_pos, client_idx + 1]   # 当前位置→该客户
            dist_back = dist_matrix[client_idx + 1, depot_idx]        # 该客户→配送中心

            new_load = current_load + client_weight
            # 预估总里程 = 已走路程 + 去该客户 + 返回配送中心
            new_dist = current_dist + dist_to_next + dist_back

            if new_load <= max_capacity and new_dist <= max_distance:
                # 满足约束：装入当前趟次
                route.append(client_idx)
                current_load += client_weight
                current_dist += dist_to_next            # 只累加去程
                current_pos = client_idx + 1
                i += 1
            else:
                if not route:
                    # 当前趟次为空（第一个客户就超限）→ 强制单独分配并增加惩罚
                    dist_trip = dist_to_next + dist_back
                    drone_id, depart, arrive = scheduler.assign(dist_trip)
                    routes.append({
                        "drone_id": drone_id,
                        "depart_time": depart,
                        "arrive_time": arrive,
                        "route": [client_idx],
                        "load": client_weight,
                        "distance": dist_trip,
                        "deliveries": 1,
                    })
                    # 惩罚 = 权重 × (超重部分 + 超距部分)
                    penalty += penalty_weight * (
                        max(0, client_weight - max_capacity)
                        + max(0, dist_trip - max_distance)
                    )
                    i += 1
                break  # 该客户装不下 → 结束当前趟次，下一轮开启新趟次

        if route:
            # 加上从最后一个客户返回配送中心的距离
            final_dist = current_dist + dist_matrix[current_pos, depot_idx]
            drone_id, depart, arrive = scheduler.assign(final_dist)
            routes.append({
                "drone_id": drone_id,
                "depart_time": depart,
                "arrive_time": arrive,
                "route": route,
                "load": current_load,
                "distance": final_dist,
                "deliveries": len(route),
            })

    return routes, penalty


def evaluate(individual, clients, dist_matrix, num_drones):
    """适应度 = 总飞行距离 + 约束惩罚（最小化）"""
    routes, penalty = decode_individual(
        individual, clients, dist_matrix, num_drones,
        MAX_CAPACITY, MAX_DISTANCE, DRONE_SPEED, PENALTY_WEIGHT,
    )
    total_dist = sum(r["distance"] for r in routes)
    return (total_dist + penalty,)


def run_genetic_algorithm(clients, dist_matrix, num_drones):
    """
    使用 DEAP 框架运行遗传算法。

    编码方式：有序排列编码 —— 个体是客户索引 0..n-1 的一个排列，
    解码时按顺序打包成满足约束的路线。

    流程：初始化种群 → 评估适应度 → 循环（选择/交叉/变异 + 精英保留）→ 返回最优解。
    """
    n = len(clients)
    random.seed(RANDOM_SEED)

    # 注册 DEAP 类型（幂等，避免重复创建报错）
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 权重 -1.0 = 最小化
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # 个体：从 [0, n-1] 中不放回抽取的排列
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, clients=clients,
                     dist_matrix=dist_matrix, num_drones=num_drones)
    toolbox.register("mate", tools.cxOrdered)                     # 有序交叉（保留排列结构）
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / n)  # 索引洗牌变异
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNSIZE)  # 锦标赛选择

    # 初始化种群和名人堂
    pop = toolbox.population(n=GA_POP_SIZE)
    hof = tools.HallOfFame(GA_ELITE)

    # 统计指标
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"\n遗传算法开始：无人机={num_drones} 架，种群={GA_POP_SIZE}，"
          f"迭代={GA_NGEN}，交叉率={GA_CXPB}，变异率={GA_MUTPB}")
    print("-" * 60)

    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg", "max"]

    # 初始种群评估
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(f"Gen  0 | min={record['min']:10.2f}  avg={record['avg']:10.2f}  max={record['max']:10.2f}")

    # 主进化循环
    for gen in range(1, GA_NGEN + 1):
        # 选择（保留精英位置给上代精英）
        offspring = toolbox.select(pop, len(pop) - GA_ELITE)
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values   # 标记适应度已失效
                del child2.fitness.values

        # 变异
        for mutant in offspring:
            if random.random() < GA_MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values   # 标记适应度已失效

        # 重新评估因交叉/变异而失效的个体
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_inds))
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # 精英保留：上代最优个体替换到种群末尾
        elites = list(map(toolbox.clone, hof.items))
        pop[:] = offspring + elites
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)

        if gen % 50 == 0 or gen == GA_NGEN:
            print(f"Gen {gen:3d} | min={record['min']:10.2f}  "
                  f"avg={record['avg']:10.2f}  max={record['max']:10.2f}")

    print("-" * 60)
    print(f"进化完成！最优适应度: {hof[0].fitness.values[0]:.2f}")

    # 解码最优个体为具体路线
    best_individual = hof[0]
    best_routes, _ = decode_individual(
        best_individual, clients, dist_matrix, num_drones,
        MAX_CAPACITY, MAX_DISTANCE, DRONE_SPEED, PENALTY_WEIGHT,
    )
    return best_routes, logbook


# ==================== 结果输出 ====================

def print_results(clients, routes, num_drones, save_to_file=False,
                  filename=None, output_dir=None):
    """按无人机分组打印路径详情。"""
    total_distance = sum(r["distance"] for r in routes)
    total_deliveries = sum(r["deliveries"] for r in routes)

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("无人机配送路径规划结果（遗传算法）")
    lines.append("=" * 60)

    drone_stats = {i: {"distance": 0.0, "trips": 0} for i in range(num_drones)}
    for r in routes:
        drone_stats[r["drone_id"]]["distance"] += r["distance"]
        drone_stats[r["drone_id"]]["trips"] += 1

    active_drones = sorted({r["drone_id"] for r in routes})

    lines.append(f"总客户数:        {len(clients)}")
    lines.append(f"无人机机队规模:  {num_drones} 架")
    lines.append(f"实际参与调度:    {len(active_drones)} 架"
                 f"（飞机编号：{[d + 1 for d in active_drones]}）")
    lines.append(f"总趟次数:        {len(routes)}")
    lines.append(f"总飞行距离:      {total_distance:.2f} 单位")
    lines.append(f"总配送次数:      {total_deliveries}")
    lines.append("-" * 60)
    lines.append("各无人机汇总:")
    for did in sorted(drone_stats):
        s = drone_stats[did]
        if s["trips"] > 0:
            lines.append(f"  无人机 {did + 1:2d}:  {s['trips']} 趟，总里程 {s['distance']:.2f} 单位")
        else:
            lines.append(f"  无人机 {did + 1:2d}:  未参与调度（机队富余）")
    lines.append("=" * 60)

    for did in sorted(active_drones):
        drone_routes = [r for r in routes if r["drone_id"] == did]
        for trip_num, route in enumerate(drone_routes, 1):
            lines.append(f"\n无人机 {did + 1} 第 {trip_num} 趟:")
            lines.append(f"  路径顺序: 配送中心 → "
                         f"{' → '.join(str(i) for i in route['route'])} → 配送中心")
            lines.append(f"  配送客户编号: {route['route']}")
            lines.append(f"  各客户重量: {[int(clients[i, 2]) for i in route['route']]} kg")
            lines.append(f"  总载重: {route['load']:.1f} kg  (限制: {MAX_CAPACITY} kg)")
            lines.append(f"  总里程: {route['distance']:.2f} 单位  (限制: {MAX_DISTANCE} 单位)")
            lines.append(f"  出发时刻: {route['depart_time']:.1f}  "
                         f"返航时刻: {route['arrive_time']:.1f}")
            lines.append(f"  配送次数: {route['deliveries']} 次")

    output = "\n".join(lines)
    print(output)

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}.txt")
        else:
            filepath = f"{filename}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n结果已保存: {filepath}")

    return output


# ==================== 可视化 ====================

# ==================== 暗色主题配色 ====================
_BG = "#0d1117"         # 图表背景（GitHub 暗色风）
_PANEL = "#161b22"       # 子图/面板背景
_GRID = "#21262d"        # 网格线和边框
_TEXT_PRI = "#e6edf3"    # 主文字颜色
_TEXT_SEC = "#8b949e"    # 次要文字颜色
_DEPOT_COL = "#ff6b6b"   # 配送中心标记色

_PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
    "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb347",
    "#63e6be", "#f8a5c2", "#a9dc76", "#fc9867", "#ab9df2", "#78dce8",
]


def _make_drone_colors(active_drones: list) -> dict:
    """为每架实际参与调度的无人机分配唯一颜色"""
    return {did: _PALETTE[i % len(_PALETTE)] for i, did in enumerate(active_drones)}


def _style_ax(ax, xlim=None, ylim=None, coord_range=None):
    """统一暗色主题样式：背景、网格、坐标轴颜色"""
    r = coord_range or COORD_RANGE
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT_SEC, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, linestyle="-", alpha=0.8)
    xl = xlim if xlim else (r[0] - 5, r[1] + 5)
    yl = ylim if ylim else (r[0] - 5, r[1] + 5)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")


def _draw_depot(ax, depot, size=120):
    """绘制配送中心：先画大光晕再画五角星标记"""
    ax.scatter(*depot, s=size * 3, color=_DEPOT_COL, alpha=0.15, zorder=7, edgecolors="none")
    ax.scatter(*depot, s=size, color=_DEPOT_COL, marker="*", zorder=8,
               edgecolors=_BG, linewidths=0.8)


def _draw_all_clients_dim(ax, clients):
    """绘制所有客户位置（半透明暗色，作为背景层）"""
    ax.scatter(clients[:, 0], clients[:, 1], s=22, color=_TEXT_SEC, alpha=0.4,
               zorder=2, edgecolors="none")


def _plot_overview(ax, clients, depot, routes, drone_colors, coord_range):
    """绘制总览图：所有飞机路线叠加在同一张图上"""
    _style_ax(ax, coord_range=coord_range)
    _draw_all_clients_dim(ax, clients)
    # 最大里程半径虚线圆
    ax.add_patch(Circle(depot, MAX_DISTANCE / 2, color=_TEXT_SEC,
                        linestyle=(0, (4, 4)), fill=False, alpha=0.25,
                        linewidth=0.8, zorder=1))

    for route in routes:
        color = drone_colors[route["drone_id"]]
        # 完整路径：配送中心 → 各客户 → 配送中心
        path = np.array([depot] + [clients[i, :2] for i in route["route"]] + [depot])
        # 先画半透明连线，再画箭头，最后高亮客户点
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, alpha=0.18,
                solid_capstyle="round", zorder=2)
        for k in range(len(path) - 1):
            ax.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle="-|>", mutation_scale=9,
                color=color, linewidth=1.2, alpha=0.65, zorder=3, capstyle="round"))
        for idx in route["route"]:
            x, y = clients[idx, :2]
            ax.scatter(x, y, s=38, color=color, zorder=5, alpha=0.65,
                       edgecolors=_BG, linewidths=0.8)

    _draw_depot(ax, depot)
    ax.set_title("总览", color=_TEXT_PRI, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("X", color=_TEXT_SEC, fontsize=7)
    ax.set_ylabel("Y", color=_TEXT_SEC, fontsize=7)


def _plot_single_drone(ax, clients, depot, drone_routes, color, drone_id, coord_range):
    """绘制单架无人机的所有趟次路线（用于详情图子图）"""
    _style_ax(ax, coord_range=coord_range)
    _draw_all_clients_dim(ax, clients)

    # 不同趟次用不同线型区分
    linestyles = ["-", "--", "-.", ":"]
    for t_idx, route in enumerate(drone_routes):
        ls = linestyles[t_idx % len(linestyles)]
        path = np.array([depot] + [clients[i, :2] for i in route["route"]] + [depot])
        # 底层粗线（光晕效果）+ 上层虚线/实线
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=4, alpha=0.12,
                solid_capstyle="round", zorder=2)
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.6, linestyle=ls,
                alpha=0.9, solid_capstyle="round", zorder=3)
        for k in range(len(path) - 1):
            ax.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle="-|>", mutation_scale=10,
                color=color, linewidth=1.4, alpha=0.85, zorder=4, capstyle="round"))
        # 标注客户编号
        for seq, idx in enumerate(route["route"]):
            x, y = clients[idx, :2]
            ax.scatter(x, y, s=45, color=color, zorder=6, edgecolors=_BG, linewidths=1.0)
            ax.text(x + 1.8, y + 1.8, f"{idx}", fontsize=6, color=_TEXT_PRI,
                    zorder=7, fontweight="bold")
        # 路线中心标注载重和里程
        cx = path[1:-1, 0].mean()
        cy = path[1:-1, 1].mean()
        ax.text(cx, cy, f"趟{t_idx + 1}  {route['load']:.0f}kg/{route['distance']:.0f}",
                fontsize=5.5, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=_PANEL,
                          edgecolor=color, alpha=0.85, linewidth=0.8), zorder=8)

    _draw_depot(ax, depot, size=80)
    total_dist = sum(r["distance"] for r in drone_routes)
    n_trips = len(drone_routes)
    ax.set_title(f"飞机 {drone_id + 1}   {n_trips}趟  {total_dist:.0f}单位",
                 color=color, fontsize=8, fontweight="bold", pad=5)
    ax.set_xlabel("X", color=_TEXT_SEC, fontsize=6)
    ax.set_ylabel("Y", color=_TEXT_SEC, fontsize=6)


def _draw_stats_panel(ax, clients, routes, num_drones, drone_colors):
    """绘制右侧统计面板：汇总指标 + 各飞机里程水平柱状图"""
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.set_xticks([])
    ax.set_yticks([])

    total_dist = sum(r["distance"] for r in routes)
    total_trips = len(routes)
    active = sorted({r["drone_id"] for r in routes})

    ax.text(0.5, 0.97, "任务统计", transform=ax.transAxes, color=_TEXT_PRI,
            fontsize=9, fontweight="bold", ha="center", va="top")

    summaries = [
        ("客户点", f"{len(clients)}"), ("机队规模", f"{num_drones} 架"),
        ("参与调度", f"{len(active)} 架"), ("总趟次", f"{total_trips}"),
        ("总里程", f"{total_dist:.1f}"),
    ]
    y = 0.88
    for label, val in summaries:
        ax.text(0.08, y, label, transform=ax.transAxes, color=_TEXT_SEC, fontsize=7.5, va="top")
        ax.text(0.92, y, val, transform=ax.transAxes, color=_TEXT_PRI, fontsize=7.5,
                va="top", ha="right", fontweight="bold")
        y -= 0.072

    ax.plot([0.05, 0.95], [y + 0.03, y + 0.03], color=_GRID, linewidth=0.8,
            transform=ax.transAxes)
    y -= 0.03

    ax.text(0.5, y, "各飞机里程", transform=ax.transAxes, color=_TEXT_SEC,
            fontsize=7, ha="center", va="top")
    y -= 0.065

    drone_stats = {}
    for r in routes:
        did = r["drone_id"]
        drone_stats.setdefault(did, {"dist": 0.0, "trips": 0})
        drone_stats[did]["dist"] += r["distance"]
        drone_stats[did]["trips"] += 1

    max_dist = max(s["dist"] for s in drone_stats.values()) if drone_stats else 1

    for did in active:
        if y < 0.04:
            break
        s = drone_stats[did]
        color = drone_colors[did]
        bar_w = s["dist"] / max_dist * 0.55
        bar_rect = plt.Rectangle((0.08, y - 0.028), bar_w, 0.022, transform=ax.transAxes,
                                 color=color, alpha=0.75, zorder=3)
        ax.add_patch(bar_rect)
        bg_rect = plt.Rectangle((0.08, y - 0.028), 0.55, 0.022, transform=ax.transAxes,
                                color=_GRID, alpha=0.4, zorder=2)
        ax.add_patch(bg_rect)
        ax.text(0.08, y, f"#{did + 1}", transform=ax.transAxes, color=color,
                fontsize=6.5, va="bottom", fontweight="bold")
        ax.text(0.92, y, f"{s['dist']:.0f}  {s['trips']}趟", transform=ax.transAxes,
                color=_TEXT_SEC, fontsize=6, va="bottom", ha="right")
        y -= 0.065


def plot_results(clients, depot, routes, num_drones, save_to_file=False,
                 filename=None, output_dir=None):
    """
    分两张独立 figure 输出：
    1. 总览图（左侧全景 + 右侧统计面板）
    2. 单机详情图（每架无人机独占子图，多趟次叠加）
    """
    import matplotlib.gridspec as gridspec

    active_drones = sorted({r["drone_id"] for r in routes})
    drone_colors = _make_drone_colors(active_drones)
    n_active = len(active_drones)
    depot_arr = np.array(depot)

    # Figure 1：总览图 + 统计侧边栏
    fig1 = plt.figure(figsize=(16, 10), facecolor=_BG)
    fig1.text(0.5, 0.97,
              f"无人机配送路径规划  ·  遗传算法  ·  机队 {num_drones} 架  ·  总览",
              ha="center", va="top", color=_TEXT_PRI, fontsize=14, fontweight="bold")
    gs1 = gridspec.GridSpec(1, 2, figure=fig1, width_ratios=[4, 1],
                            left=0.04, right=0.97, top=0.92, bottom=0.06, wspace=0.06)
    ax_overview = fig1.add_subplot(gs1[0, 0])
    ax_stats = fig1.add_subplot(gs1[0, 1])
    _plot_overview(ax_overview, clients, depot_arr, routes, drone_colors, COORD_RANGE)
    _draw_stats_panel(ax_stats, clients, routes, num_drones, drone_colors)

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}_overview.png")
        else:
            filepath = f"{filename}_overview.png"
        fig1.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"总览图已保存: {filepath}")
    plt.show()

    # Figure 2：每架无人机独立子图
    COLS = 4
    n_drone_rows = max(1, int(np.ceil(n_active / COLS)))

    fig2 = plt.figure(figsize=(COLS * 3.8, n_drone_rows * 3.8 + 0.8), facecolor=_BG)
    fig2.text(0.5, 0.98,
              f"无人机配送路径规划  ·  各飞机详情  ·  机队 {num_drones} 架",
              ha="center", va="top", color=_TEXT_PRI, fontsize=13, fontweight="bold")
    gs2 = gridspec.GridSpec(n_drone_rows, COLS, figure=fig2, hspace=0.38,
                            wspace=0.22, top=0.93, bottom=0.04, left=0.03, right=0.98)
    for plot_idx, did in enumerate(active_drones):
        row = plot_idx // COLS
        col = plot_idx % COLS
        ax = fig2.add_subplot(gs2[row, col])
        drone_routes = [r for r in routes if r["drone_id"] == did]
        _plot_single_drone(ax, clients, depot_arr, drone_routes,
                           drone_colors[did], did, COORD_RANGE)

    for blank_idx in range(n_active, n_drone_rows * COLS):
        ax = fig2.add_subplot(gs2[blank_idx // COLS, blank_idx % COLS])
        ax.set_visible(False)

    if save_to_file:
        if output_dir:
            filepath = os.path.join(output_dir, f"{filename}_details.png")
        else:
            filepath = f"{filename}_details.png"
        fig2.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"单机详情图已保存: {filepath}")
    plt.show()


def plot_evolution(logbook, save_to_file=False, filename=None, output_dir=None):
    """进化曲线：暗色主题，带渐变填充与收敛标注。"""
    gens = np.array(logbook.select("gen"))
    mins = np.array(logbook.select("min"))
    avgs = np.array(logbook.select("avg"))

    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=_BG)
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT_SEC, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.8)

    ax.fill_between(gens, avgs, mins, alpha=0.08, color="#58a6ff")
    ax.fill_between(gens, mins, mins.min(), alpha=0.12, color="#3fb950")
    ax.plot(gens, avgs, color="#58a6ff", linewidth=1.5, alpha=0.7,
            label="平均适应度", linestyle="--")
    ax.plot(gens, mins, color="#3fb950", linewidth=2.2, label="最优适应度",
            solid_capstyle="round")

    best_gen = int(gens[np.argmin(mins)])
    best_val = mins.min()
    ax.scatter([best_gen], [best_val], color="#f78166", s=60, zorder=6)
    ax.annotate(
        f"最优 {best_val:.1f}  (第{best_gen}代)",
        xy=(best_gen, best_val),
        xytext=(best_gen + len(gens) * 0.05,
                best_val + (avgs.max() - mins.min()) * 0.08),
        color=_TEXT_PRI, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=_TEXT_SEC, lw=1.0),
    )

    ax.set_xlabel("迭代代数", color=_TEXT_SEC, fontsize=9)
    ax.set_ylabel("适应度（总飞行距离）", color=_TEXT_SEC, fontsize=9)
    ax.set_title("遗传算法进化曲线", color=_TEXT_PRI, fontsize=12, fontweight="bold")
    legend = ax.legend(fontsize=9, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT_PRI)
    plt.tight_layout()

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_ga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}_evolution.png")
        else:
            filepath = f"{filename}_evolution.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"进化曲线已保存: {filepath}")

    plt.show()


def run_genetic(num_drones=10, clients=None, output_dir=None):
    """
    运行遗传算法管线（可由外部模块调用的统一入口）。
    与 run_greedy 接口一致，便于在 main.py 中互换调用。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"drone_delivery_ga_{timestamp}"

    # 若无外部数据，自动生成并清洗
    if clients is None:
        clients = generate_simulation_data()
        clients = clean_data(clients)

    dist_matrix = compute_distance_matrix(clients)
    best_routes, logbook = run_genetic_algorithm(clients, dist_matrix, num_drones)

    # 输出三种可视化：文本结果、路线图（总览+详情）、进化曲线
    print_results(clients, best_routes, num_drones, save_to_file=True,
                  filename=base_name, output_dir=output_dir)
    plot_results(clients, DEPOT_COORDS, best_routes, num_drones, save_to_file=True,
                 filename=base_name, output_dir=output_dir)
    plot_evolution(logbook, save_to_file=True, filename=base_name, output_dir=output_dir)

    total_distance = sum(r['distance'] for r in best_routes)
    return {
        'clients': clients,
        'routes': best_routes,
        'dist_matrix': dist_matrix,
        'total_distance': total_distance,
        'total_trips': len(best_routes),
        'total_deliveries': sum(r['deliveries'] for r in best_routes),
        'logbook': logbook,
    }


def main():
    import argparse

    print("=" * 60)
    print("无人机配送路径规划 - 遗传算法版本（DEAP）")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='遗传算法无人机配送路径规划')
    parser.add_argument('--num-drones', type=int, default=10, help='无人机数量')
    parser.add_argument('--output-dir', default='../outputs', help='输出目录')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    print(f"机队规模: {args.num_drones} 架无人机")
    print("\n【步骤1】生成仿真数据")
    clients = generate_simulation_data()
    print("\n【步骤2】数据清洗")
    clients = clean_data(clients)
    print("\n【步骤3】计算距离矩阵")
    dist_matrix = compute_distance_matrix(clients)
    print("\n【步骤4】遗传算法路径优化")
    best_routes, logbook = run_genetic_algorithm(clients, dist_matrix, args.num_drones)
    print("\n【步骤5】输出结果")
    print_results(clients, best_routes, args.num_drones, save_to_file=True, output_dir=output_dir)
    print("\n【步骤6】可视化配送路径")
    plot_results(clients, DEPOT_COORDS, best_routes, args.num_drones,
                 save_to_file=True, output_dir=output_dir)
    print("\n【步骤7】可视化进化过程")
    plot_evolution(logbook, save_to_file=True, output_dir=output_dir)


if __name__ == "__main__":
    main()
