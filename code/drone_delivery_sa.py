#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 模拟退火算法版本
功能：仿真数据生成 + 数据清洗 + 距离矩阵 + 模拟退火优化 + 结果输出 + 可视化
"""

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from datetime import datetime
from copy import deepcopy

from drone_delivery_genetic import FleetScheduler, decode_individual

from common import (
    generate_simulation_data, clean_data, compute_distance_matrix,
    RANDOM_SEED, NUM_CLIENTS, COORD_RANGE, WEIGHT_RANGE,
    MAX_CAPACITY, MAX_DISTANCE, DRONE_SPEED, DEPOT_COORDS,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 模拟退火超参数 ====================
SA_T0 = 500.0           # 初始温度 —— 高温保证搜索广度
SA_T_MIN = 0.01         # 终止温度 —— 低温收敛到局部最优
SA_ALPHA = 0.98         # 降温速率 —— 越接近1降温越慢、搜索越充分
SA_ITER_PER_T = 50      # 每个温度下的迭代次数
SA_MAX_RESTARTS = 3     # 重启动次数 —— 多次独立搜索取最优
PENALTY_WEIGHT = 1e4    # 约束违反惩罚系数（与遗传算法一致）

# ==================== 暗色主题配色 ====================
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


def evaluate_solution(individual, clients, dist_matrix, num_drones):
    """适应度 = 总飞行距离 + 约束惩罚（最小化），与遗传算法评估函数一致"""
    routes, penalty = decode_individual(
        individual, clients, dist_matrix, num_drones,
        MAX_CAPACITY, MAX_DISTANCE, DRONE_SPEED, PENALTY_WEIGHT,
    )
    total_dist = sum(r["distance"] for r in routes)
    return total_dist + penalty, routes


def generate_neighbor(individual):
    """
    生成邻域解，随机选择以下三种算子之一：
    1. 交换：随机交换两个位置
    2. 反转：反转一段随机子序列（2-opt风格）
    3. 插入：将一个元素移到新位置
    """
    n = len(individual)
    neighbor = individual[:]
    r = random.random()

    if r < 0.4:
        # 交换两个位置
        i, j = random.sample(range(n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    elif r < 0.7:
        # 反转一段子序列
        i, j = sorted(random.sample(range(n), 2))
        neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
    else:
        # 插入：从位置i移到位置j
        i = random.randrange(n)
        elem = neighbor.pop(i)
        j = random.randrange(n - 1)
        neighbor.insert(j, elem)

    return neighbor


def simulated_annealing(clients, dist_matrix, num_drones,
                        t0=SA_T0, t_min=SA_T_MIN, alpha=SA_ALPHA,
                        iter_per_t=SA_ITER_PER_T):
    """
    模拟退火主算法。

    编码：客户索引 0..n-1 的排列（与遗传算法一致）
    解码：按排列顺序贪心打包成满足约束的路线

    流程：从随机解出发，逐步降温，每个温度下迭代 ITER_PER_T 次，
    按 Metropolis 准则接受新解，记录搜索轨迹。
    """
    n = len(clients)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    best_overall = None
    best_routes_overall = None
    best_cost_overall = float("inf")
    all_cost_history = []
    all_temp_history = []

    for restart in range(SA_MAX_RESTARTS):
        # 随机初始解
        current = list(range(n))
        random.shuffle(current)
        current_cost, current_routes = evaluate_solution(
            current, clients, dist_matrix, num_drones
        )

        best = current[:]
        best_routes = current_routes
        best_cost = current_cost

        cost_history = []
        temp_history = []
        T = t0
        total_iter = 0

        while T > t_min:
            for _ in range(iter_per_t):
                neighbor = generate_neighbor(current)
                neighbor_cost, neighbor_routes = evaluate_solution(
                    neighbor, clients, dist_matrix, num_drones
                )
                delta = neighbor_cost - current_cost

                if delta < 0 or random.random() < np.exp(-delta / T):
                    current = neighbor
                    current_cost = neighbor_cost
                    current_routes = neighbor_routes

                    if current_cost < best_cost:
                        best = current[:]
                        best_routes = current_routes
                        best_cost = current_cost

                cost_history.append(best_cost)
                temp_history.append(T)
                total_iter += 1

            T *= alpha

        print(f"  重启 {restart + 1}/{SA_MAX_RESTARTS}: "
              f"最优距离 = {best_cost:.2f}, 总迭代 = {total_iter}")

        if best_cost < best_cost_overall:
            best_overall = best[:]
            best_routes_overall = best_routes
            best_cost_overall = best_cost

        all_cost_history.append((cost_history, temp_history))

    print(f"\n模拟退火完成！全局最优距离: {best_cost_overall:.2f}")
    return best_overall, best_routes_overall, best_cost_overall, all_cost_history


# ==================== 结果输出 ====================

def print_results(clients, routes, num_drones, save_to_file=False,
                  filename=None, output_dir=None):
    """按无人机分组打印路径详情"""
    total_distance = sum(r["distance"] for r in routes)
    total_deliveries = sum(r["deliveries"] for r in routes)

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("无人机配送路径规划结果（模拟退火算法）")
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
            filename = f"drone_delivery_sa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

def _make_drone_colors(active_drones):
    return {did: _PALETTE[i % len(_PALETTE)] for i, did in enumerate(active_drones)}


def _style_ax(ax, coord_range=None):
    r = coord_range or COORD_RANGE
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT_SEC, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, linestyle="-", alpha=0.8)
    ax.set_xlim(r[0] - 5, r[1] + 5)
    ax.set_ylim(r[0] - 5, r[1] + 5)
    ax.set_aspect("equal")


def _draw_depot(ax, depot, size=120):
    ax.scatter(*depot, s=size * 3, color=_DEPOT_COL, alpha=0.15, zorder=7, edgecolors="none")
    ax.scatter(*depot, s=size, color=_DEPOT_COL, marker="*", zorder=8,
               edgecolors=_BG, linewidths=0.8)


def _draw_all_clients_dim(ax, clients):
    ax.scatter(clients[:, 0], clients[:, 1], s=22, color=_TEXT_SEC, alpha=0.4,
               zorder=2, edgecolors="none")


def plot_routes(clients, depot, routes, num_drones, save_to_file=False,
                filename=None, output_dir=None):
    """总览图：所有飞机路线叠加 + 右侧统计面板"""
    import matplotlib.gridspec as gridspec

    active_drones = sorted({r["drone_id"] for r in routes})
    drone_colors = _make_drone_colors(active_drones)
    depot_arr = np.array(depot)

    fig = plt.figure(figsize=(16, 10), facecolor=_BG)
    fig.text(0.5, 0.97,
             f"无人机配送路径规划  ·  模拟退火算法  ·  机队 {num_drones} 架  ·  总览",
             ha="center", va="top", color=_TEXT_PRI, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4, 1],
                           left=0.04, right=0.97, top=0.92, bottom=0.06, wspace=0.06)

    # 左侧：总览路径
    ax_main = fig.add_subplot(gs[0, 0])
    _style_ax(ax_main)
    _draw_all_clients_dim(ax_main, clients)
    ax_main.add_patch(Circle(depot, MAX_DISTANCE / 2, color=_TEXT_SEC,
                             linestyle=(0, (4, 4)), fill=False, alpha=0.25,
                             linewidth=0.8, zorder=1))

    for route in routes:
        color = drone_colors[route["drone_id"]]
        path = np.array([depot] + [clients[i, :2] for i in route["route"]] + [depot])
        ax_main.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, alpha=0.18,
                     solid_capstyle="round", zorder=2)
        for k in range(len(path) - 1):
            ax_main.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle="-|>", mutation_scale=9,
                color=color, linewidth=1.2, alpha=0.65, zorder=3, capstyle="round"))
        for idx in route["route"]:
            x, y = clients[idx, :2]
            ax_main.scatter(x, y, s=38, color=color, zorder=5, alpha=0.65,
                            edgecolors=_BG, linewidths=0.8)

    _draw_depot(ax_main, depot)
    ax_main.set_title("总览", color=_TEXT_PRI, fontsize=9, fontweight="bold", pad=6)
    ax_main.set_xlabel("X", color=_TEXT_SEC, fontsize=7)
    ax_main.set_ylabel("Y", color=_TEXT_SEC, fontsize=7)

    # 右侧：统计面板
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.set_facecolor(_PANEL)
    for sp in ax_stats.spines.values():
        sp.set_edgecolor(_GRID)
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])

    total_dist = sum(r["distance"] for r in routes)
    total_trips = len(routes)
    active = sorted({r["drone_id"] for r in routes})

    ax_stats.text(0.5, 0.97, "任务统计", transform=ax_stats.transAxes, color=_TEXT_PRI,
                  fontsize=9, fontweight="bold", ha="center", va="top")

    summaries = [
        ("客户点", f"{len(clients)}"), ("机队规模", f"{num_drones} 架"),
        ("参与调度", f"{len(active)} 架"), ("总趟次", f"{total_trips}"),
        ("总里程", f"{total_dist:.1f}"),
    ]
    y = 0.88
    for label, val in summaries:
        ax_stats.text(0.08, y, label, transform=ax_stats.transAxes, color=_TEXT_SEC, fontsize=7.5, va="top")
        ax_stats.text(0.92, y, val, transform=ax_stats.transAxes, color=_TEXT_PRI, fontsize=7.5,
                      va="top", ha="right", fontweight="bold")
        y -= 0.072

    ax_stats.plot([0.05, 0.95], [y + 0.03, y + 0.03], color=_GRID, linewidth=0.8,
                  transform=ax_stats.transAxes)
    y -= 0.03
    ax_stats.text(0.5, y, "各飞机里程", transform=ax_stats.transAxes, color=_TEXT_SEC,
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
        ax_stats.add_patch(plt.Rectangle((0.08, y - 0.028), bar_w, 0.022, transform=ax_stats.transAxes,
                                         color=color, alpha=0.75, zorder=3))
        ax_stats.add_patch(plt.Rectangle((0.08, y - 0.028), 0.55, 0.022, transform=ax_stats.transAxes,
                                         color=_GRID, alpha=0.4, zorder=2))
        ax_stats.text(0.08, y, f"#{did + 1}", transform=ax_stats.transAxes, color=color,
                      fontsize=6.5, va="bottom", fontweight="bold")
        ax_stats.text(0.92, y, f"{s['dist']:.0f}  {s['trips']}趟", transform=ax_stats.transAxes,
                      color=_TEXT_SEC, fontsize=6, va="bottom", ha="right")
        y -= 0.065

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_sa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}_overview.png")
        else:
            filepath = f"{filename}_overview.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"总览图已保存: {filepath}")
    plt.show()


def plot_annealing_curve(all_cost_history, save_to_file=False,
                         filename=None, output_dir=None):
    """
    退火曲线：每次重启的搜索轨迹（最优成本 vs 迭代次数），
    并在双Y轴上叠加温度下降曲线。
    """
    fig, axes = plt.subplots(SA_MAX_RESTARTS, 1, figsize=(14, 3.2 * SA_MAX_RESTARTS),
                             facecolor=_BG, squeeze=False)
    axes = axes.flatten()

    restart_colors = ["#58a6ff", "#3fb950", "#f78166"]

    for idx, (cost_hist, temp_hist) in enumerate(all_cost_history):
        ax = axes[idx]
        ax.set_facecolor(_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
        ax.tick_params(colors=_TEXT_SEC, labelsize=7.5)
        ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)

        color = restart_colors[idx]
        iterations = np.arange(len(cost_hist))

        ax2 = ax.twinx()
        ax2.set_facecolor(_PANEL)
        ax2.tick_params(colors=_TEXT_SEC, labelsize=7)

        # 温度曲线（对数坐标更直观）
        ax2.plot(iterations, temp_hist, color="#f78166", linewidth=0.8, alpha=0.5,
                 linestyle="--", label="温度")
        ax2.set_ylabel("温度", color=_TEXT_SEC, fontsize=7.5)
        ax2.set_yscale("log")

        # 最优成本曲线
        ax.plot(iterations, cost_hist, color=color, linewidth=1.2, alpha=0.9)
        ax.set_ylabel("最优距离", color=_TEXT_SEC, fontsize=7.5)
        ax.set_xlabel("迭代次数", color=_TEXT_SEC, fontsize=7.5)

        final_cost = cost_hist[-1]
        ax.set_title(f"重启 {idx + 1}  |  最终最优距离: {final_cost:.2f}  |  "
                     f"初始温度: {SA_T0:.0f} → 终止: {SA_T_MIN:.3f}",
                     color=_TEXT_PRI, fontsize=9, fontweight="bold")

        # 标注初始值和最终值
        ax.scatter([0], [cost_hist[0]], color=color, s=30, zorder=6, alpha=0.6)
        ax.scatter([len(cost_hist) - 1], [final_cost], color=color, s=50, zorder=6)

    fig.suptitle("模拟退火搜索轨迹", color=_TEXT_PRI, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_sa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}_annealing.png")
        else:
            filepath = f"{filename}_annealing.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"退火曲线已保存: {filepath}")
    plt.show()


def plot_convergence_summary(all_cost_history, save_to_file=False,
                             filename=None, output_dir=None):
    """汇总图：所有重启的最优成本曲线叠加对比"""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=_BG)
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT_SEC, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)

    restart_colors = ["#58a6ff", "#3fb950", "#f78166"]
    best_overall = float("inf")

    for idx, (cost_hist, _) in enumerate(all_cost_history):
        color = restart_colors[idx]
        iterations = np.arange(len(cost_hist))
        ax.plot(iterations, cost_hist, color=color, linewidth=1.5, alpha=0.85,
                label=f"重启 {idx + 1}（最终: {cost_hist[-1]:.1f}）")
        best_overall = min(best_overall, cost_hist[-1])

    ax.axhline(y=best_overall, color="#d2a8ff", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("迭代次数", color=_TEXT_SEC, fontsize=9)
    ax.set_ylabel("最优距离", color=_TEXT_SEC, fontsize=9)
    ax.set_title(f"模拟退火收敛对比  ·  全局最优: {best_overall:.2f}",
                 color=_TEXT_PRI, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT_PRI)
    plt.tight_layout()

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_sa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}_convergence.png")
        else:
            filepath = f"{filename}_convergence.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"收敛汇总图已保存: {filepath}")
    plt.show()


# ==================== 统一入口 ====================

def run_sa(num_drones=10, clients=None, output_dir=None):
    """
    运行模拟退火算法管线（可由外部模块调用的统一入口）。
    与 run_genetic 接口一致，便于在 main.py 中互换调用。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"drone_delivery_sa_{timestamp}"

    if clients is None:
        clients = generate_simulation_data()
        clients = clean_data(clients)

    dist_matrix = compute_distance_matrix(clients)
    best_individual, best_routes, best_cost, all_cost_history = simulated_annealing(
        clients, dist_matrix, num_drones
    )

    print_results(clients, best_routes, num_drones, save_to_file=True,
                  filename=base_name, output_dir=output_dir)
    plot_routes(clients, DEPOT_COORDS, best_routes, num_drones, save_to_file=True,
                filename=base_name, output_dir=output_dir)
    plot_annealing_curve(all_cost_history, save_to_file=True,
                         filename=base_name, output_dir=output_dir)
    plot_convergence_summary(all_cost_history, save_to_file=True,
                             filename=base_name, output_dir=output_dir)

    total_distance = sum(r['distance'] for r in best_routes)
    return {
        'clients': clients,
        'routes': best_routes,
        'dist_matrix': dist_matrix,
        'total_distance': total_distance,
        'total_trips': len(best_routes),
        'total_deliveries': sum(r['deliveries'] for r in best_routes),
        'all_cost_history': all_cost_history,
    }


def main():
    import argparse

    print("=" * 60)
    print("无人机配送路径规划 - 模拟退火算法版本")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='模拟退火无人机配送路径规划')
    parser.add_argument('--num-drones', type=int, default=10, help='无人机数量')
    parser.add_argument('--output-dir', default='../outputs', help='输出目录')
    parser.add_argument('--t0', type=float, default=SA_T0, help='初始温度')
    parser.add_argument('--alpha', type=float, default=SA_ALPHA, help='降温速率')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    print(f"机队规模: {args.num_drones} 架无人机")
    print(f"初始温度: {args.t0}, 降温速率: {args.alpha}")

    print("\n【步骤1】生成仿真数据")
    clients = generate_simulation_data()
    print("\n【步骤2】数据清洗")
    clients = clean_data(clients)
    print("\n【步骤3】计算距离矩阵")
    dist_matrix = compute_distance_matrix(clients)
    print("\n【步骤4】模拟退火路径优化")
    best_individual, best_routes, best_cost, all_cost_history = simulated_annealing(
        clients, dist_matrix, args.num_drones,
        t0=args.t0, alpha=args.alpha
    )
    print("\n【步骤5】输出结果")
    print_results(clients, best_routes, args.num_drones, save_to_file=True, output_dir=output_dir)
    print("\n【步骤6】可视化配送路径")
    plot_routes(clients, DEPOT_COORDS, best_routes, args.num_drones,
                save_to_file=True, output_dir=output_dir)
    print("\n【步骤7】可视化退火过程")
    plot_annealing_curve(all_cost_history, save_to_file=True, output_dir=output_dir)
    plot_convergence_summary(all_cost_history, save_to_file=True, output_dir=output_dir)


if __name__ == "__main__":
    main()
