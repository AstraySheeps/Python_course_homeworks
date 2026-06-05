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
GA_POP_SIZE = 200
GA_NGEN = 200
GA_CXPB = 0.8
GA_MUTPB = 0.3
GA_TOURNSIZE = 5
GA_ELITE = 10
PENALTY_WEIGHT = 1e4


class FleetScheduler:
    """追踪 num_drones 架无人机各自的「最早可出发时刻」。"""

    def __init__(self, num_drones: int, speed: float):
        self.num_drones = num_drones
        self.speed = speed
        self.available_at = [0.0] * num_drones

    def assign(self, flight_distance: float):
        """为一趟飞行分配飞机，返回 (drone_id, depart_time, arrive_time)"""
        drone_id = int(np.argmin(self.available_at))
        depart = self.available_at[drone_id]
        flight_time = flight_distance / self.speed
        arrive = depart + flight_time
        self.available_at[drone_id] = arrive
        return drone_id, depart, arrive

    def reset(self):
        self.available_at = [0.0] * self.num_drones


def decode_individual(individual, clients, dist_matrix, num_drones,
                      max_capacity, max_distance, drone_speed, penalty_weight):
    """将遗传算法个体解码为无人机路线列表，通过 FleetScheduler 分配真实机队。"""
    n = len(clients)
    depot_idx = 0
    scheduler = FleetScheduler(num_drones, drone_speed)
    routes = []
    penalty = 0.0

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

            if new_load <= max_capacity and new_dist <= max_distance:
                route.append(client_idx)
                current_load += client_weight
                current_dist += dist_to_next
                current_pos = client_idx + 1
                i += 1
            else:
                if not route:
                    # 第一个客户就超限，强制装入并记罚
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
                    penalty += penalty_weight * (
                        max(0, client_weight - max_capacity)
                        + max(0, dist_trip - max_distance)
                    )
                    i += 1
                break  # 开启新趟次

        if route:
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
    """使用 DEAP 运行遗传算法，返回最优路线列表及进化过程统计。"""
    n = len(clients)
    random.seed(RANDOM_SEED)

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, clients=clients,
                     dist_matrix=dist_matrix, num_drones=num_drones)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / n)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNSIZE)

    pop = toolbox.population(n=GA_POP_SIZE)
    hof = tools.HallOfFame(GA_ELITE)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"\n遗传算法开始：无人机={num_drones} 架，种群={GA_POP_SIZE}，"
          f"迭代={GA_NGEN}，交叉率={GA_CXPB}，变异率={GA_MUTPB}")
    print("-" * 60)

    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg", "max"]

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(f"Gen  0 | min={record['min']:10.2f}  avg={record['avg']:10.2f}  max={record['max']:10.2f}")

    for gen in range(1, GA_NGEN + 1):
        offspring = toolbox.select(pop, len(pop) - GA_ELITE)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < GA_MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_inds))
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

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

_BG = "#0d1117"
_PANEL = "#161b22"
_GRID = "#21262d"
_TEXT_PRI = "#e6edf3"
_TEXT_SEC = "#8b949e"
_DEPOT_COL = "#ff6b6b"

_PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
    "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb347",
    "#63e6be", "#f8a5c2", "#a9dc76", "#fc9867", "#ab9df2", "#78dce8",
]


def _make_drone_colors(active_drones: list) -> dict:
    return {did: _PALETTE[i % len(_PALETTE)] for i, did in enumerate(active_drones)}


def _style_ax(ax, xlim=None, ylim=None, coord_range=None):
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
    ax.scatter(*depot, s=size * 3, color=_DEPOT_COL, alpha=0.15, zorder=7, edgecolors="none")
    ax.scatter(*depot, s=size, color=_DEPOT_COL, marker="*", zorder=8,
               edgecolors=_BG, linewidths=0.8)


def _draw_all_clients_dim(ax, clients):
    ax.scatter(clients[:, 0], clients[:, 1], s=22, color=_TEXT_SEC, alpha=0.4,
               zorder=2, edgecolors="none")


def _plot_overview(ax, clients, depot, routes, drone_colors, coord_range):
    _style_ax(ax, coord_range=coord_range)
    _draw_all_clients_dim(ax, clients)
    ax.add_patch(Circle(depot, MAX_DISTANCE / 2, color=_TEXT_SEC,
                        linestyle=(0, (4, 4)), fill=False, alpha=0.25,
                        linewidth=0.8, zorder=1))

    for route in routes:
        color = drone_colors[route["drone_id"]]
        path = np.array([depot] + [clients[i, :2] for i in route["route"]] + [depot])
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
    _style_ax(ax, coord_range=coord_range)
    _draw_all_clients_dim(ax, clients)

    linestyles = ["-", "--", "-.", ":"]
    for t_idx, route in enumerate(drone_routes):
        ls = linestyles[t_idx % len(linestyles)]
        path = np.array([depot] + [clients[i, :2] for i in route["route"]] + [depot])
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=4, alpha=0.12,
                solid_capstyle="round", zorder=2)
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.6, linestyle=ls,
                alpha=0.9, solid_capstyle="round", zorder=3)
        for k in range(len(path) - 1):
            ax.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle="-|>", mutation_scale=10,
                color=color, linewidth=1.4, alpha=0.85, zorder=4, capstyle="round"))
        for seq, idx in enumerate(route["route"]):
            x, y = clients[idx, :2]
            ax.scatter(x, y, s=45, color=color, zorder=6, edgecolors=_BG, linewidths=1.0)
            ax.text(x + 1.8, y + 1.8, f"{idx}", fontsize=6, color=_TEXT_PRI,
                    zorder=7, fontweight="bold")
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
    """分两张独立 figure 输出：总览图 + 单机详情图"""
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
    """运行遗传算法并返回结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"drone_delivery_ga_{timestamp}"

    if clients is None:
        clients = generate_simulation_data()
        clients = clean_data(clients)

    dist_matrix = compute_distance_matrix(clients)
    best_routes, logbook = run_genetic_algorithm(clients, dist_matrix, num_drones)

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
