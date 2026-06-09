#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 贪心算法基线版本
功能：仿真数据生成 + 数据清洗 + 距离矩阵 + 贪心算法 + 结果输出 + 可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from common import (
    generate_simulation_data, clean_data, compute_distance_matrix,
    RANDOM_SEED, NUM_CLIENTS, COORD_RANGE, WEIGHT_RANGE,
    NUM_DRONES, MAX_CAPACITY, MAX_DISTANCE, DEPOT_COORDS,
    BG, PANEL, GRID, TEXT_PRI, TEXT_SEC, DEPOT_COL, PALETTE,
    DRONE_SPEED, SERVICE_TIME, PENALTY_WEIGHT,
)

from drone_delivery_genetic import FleetScheduler

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def greedy_assignment(clients, dist_matrix, max_capacity, max_distance,
                      time_windows=None):
    """
    改进贪心算法分配路径。
    每趟路线中，若最近客户装不下，继续尝试次近客户，而非直接放弃。
    当所有剩余客户都无法装入当前路线时才开启新路线。

    若提供 time_windows=(ready_times, due_times)，则选择客户时综合考虑
    距离和时间窗：迟到按惩罚系数影响排序优先级，早到允许等待。
    """
    n = len(clients)
    unassigned = set(range(n))
    drone_routes = []
    depot_idx = 0
    max_iterations = n * 10
    has_tw = time_windows is not None
    if has_tw:
        ready_times, due_times = time_windows

    iterations = 0
    while unassigned and iterations < max_iterations:
        iterations += 1
        current_route = []
        current_load = 0.0
        current_distance = 0.0
        current_pos = depot_idx
        current_time = 0.0                     # 累积时间（含等待与服务）

        while unassigned:
            candidates = []
            for client_idx in unassigned:
                dist = dist_matrix[current_pos, client_idx + 1]
                if has_tw:
                    travel_time = dist / DRONE_SPEED
                    arrival = current_time + travel_time
                    ready, due = ready_times[client_idx], due_times[client_idx]
                    # 迟到惩罚（换算为等效距离）远大于等待惩罚
                    late = max(0.0, arrival - due)
                    wait = max(0.0, ready - arrival)
                    key = dist + (late * 3.0 + wait * 0.1) * DRONE_SPEED
                else:
                    key = dist
                candidates.append((key, dist, client_idx))
            candidates.sort(key=lambda x: x[0])

            found = False
            for _, min_dist, client_idx in candidates:
                client_weight = clients[client_idx, 2]
                new_load = current_load + client_weight
                new_dist = current_distance + min_dist + dist_matrix[client_idx + 1, depot_idx]

                if new_load <= max_capacity and new_dist <= max_distance:
                    current_route.append(client_idx)
                    current_load = new_load
                    current_distance += min_dist
                    if has_tw:
                        travel_time = min_dist / DRONE_SPEED
                        arrival = current_time + travel_time
                        ready = ready_times[client_idx]
                        current_time = max(arrival, ready) + SERVICE_TIME
                    current_pos = client_idx + 1
                    unassigned.remove(client_idx)
                    found = True
                    break

            if not found:
                break

        if current_route:
            current_distance += dist_matrix[current_pos, depot_idx]
            route_info = {
                'route': current_route,
                'load': current_load,
                'distance': current_distance,
                'deliveries': len(current_route)
            }
            if has_tw:
                # 存储含等待+服务的实际耗时（供 FleetScheduler 使用）
                route_info['_tw_duration'] = (
                    current_time + dist_matrix[current_pos, depot_idx] / DRONE_SPEED
                )
            drone_routes.append(route_info)
        else:
            remaining = list(unassigned)
            for client_idx in remaining:
                client_weight = clients[client_idx, 2]
                dist_trip = (dist_matrix[depot_idx, client_idx + 1]
                             + dist_matrix[client_idx + 1, depot_idx])
                drone_routes.append({
                    'route': [client_idx],
                    'load': client_weight,
                    'distance': dist_trip,
                    'deliveries': 1
                })
                unassigned.discard(client_idx)
                if client_weight > max_capacity:
                    print(f"  [警告] 客户{client_idx}重量{client_weight:.0f}超限({max_capacity})，已强制分配")
                if dist_trip > max_distance:
                    print(f"  [警告] 客户{client_idx}距离{dist_trip:.1f}超限({max_distance})，已强制分配")
            break

    if iterations >= max_iterations:
        print("  [警告] 达到最大迭代次数，强制终止")

    print(f"贪心分配完成，共使用 {len(drone_routes)} 架次")
    return drone_routes


def plot_results(clients, depot, routes, save_to_file=False, filename=None, output_dir=None):
    """贪心算法路径可视化（暗色主题，与遗传/退火风格统一）。

    将所有架次的路线叠加在同一张图上，不同趟次用不同颜色区分。
    """
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.lines import Line2D
    from datetime import datetime

    fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

    # 按无人机分组，每架飞机一个颜色
    drone_routes = {}
    for route in routes:
        did = route.get('drone_id', 0)
        drone_routes.setdefault(did, []).append(route)

    active_drones = sorted(drone_routes.keys())
    drone_colors = {did: PALETTE[i % len(PALETTE)] for i, did in enumerate(active_drones)}

    # 绘制所有客户点（半透明暗色背景层）
    ax.scatter(clients[:, 0], clients[:, 1], s=22, color=TEXT_SEC, alpha=0.4,
               zorder=2, edgecolors='none')
    # 标注客户编号和重量
    for i, (x, y, w) in enumerate(clients):
        ax.text(x + 1.5, y + 1.5, f'{i}({int(w)}kg)', fontsize=7, color=TEXT_SEC)

    # 为每条路线绘制箭头路径
    for route in routes:
        did = route.get('drone_id', 0)
        color = drone_colors[did]
        trip_idx = drone_routes[did].index(route) + 1

        path = np.array([depot] + [clients[i, :2] for i in route['route']] + [depot])
        # 半透明连线
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, alpha=0.18,
                solid_capstyle='round', zorder=2)
        # 方向箭头
        for k in range(len(path) - 1):
            ax.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle='-|>', mutation_scale=10,
                color=color, linewidth=1.3, alpha=0.65, zorder=3, capstyle='round'))

        # 客户点着色
        for ci in route['route']:
            ax.scatter(*clients[ci, :2], s=45, color=color, zorder=5, alpha=0.7,
                      edgecolors=BG, linewidths=0.8)

        # 路径中心标注
        mid_x = np.mean(path[:, 0])
        mid_y = np.mean(path[:, 1])
        ax.text(mid_x, mid_y - 2.5,
                f'飞机{did + 1}·趟{trip_idx} | {route["load"]:.0f}kg/{route["distance"]:.0f}',
                fontsize=6.5, color=color, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=PANEL,
                          edgecolor=color, alpha=0.85, linewidth=0.8), zorder=7)

    # 最大里程半径虚线圆
    ax.add_patch(Circle(depot, MAX_DISTANCE / 2, color=TEXT_SEC,
                        linestyle=(0, (4, 4)), fill=False, alpha=0.25,
                        linewidth=0.8, zorder=1))

    # 配送中心 —— 光晕+五角星
    ax.scatter(*depot, s=360, color=DEPOT_COL, alpha=0.15, zorder=7, edgecolors='none')
    ax.scatter(*depot, s=120, color=DEPOT_COL, marker='*', zorder=8,
               edgecolors=BG, linewidths=0.8)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=DEPOT_COL,
               markersize=12, label='配送中心'),
    ]
    for did in active_drones:
        color = drone_colors[did]
        n_trips = len(drone_routes[did])
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=8, label=f'无人机 {did + 1} ({n_trips}趟)')
        )
    ax.legend(handles=legend_elements, fontsize=9, facecolor=PANEL,
              edgecolor=GRID, labelcolor=TEXT_PRI, loc='upper right')

    total_dist = sum(r['distance'] for r in routes)
    makespan = max(r['arrive_time'] for r in routes) if routes else 0
    ax.set_title(f'无人机配送路径规划（贪心算法）\n'
                 f'总距离: {total_dist:.1f}  |  makespan: {makespan:.1f}  |  总趟次: {len(routes)}',
                 color=TEXT_PRI, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('X坐标', color=TEXT_SEC, fontsize=10)
    ax.set_ylabel('Y坐标', color=TEXT_SEC, fontsize=10)
    ax.set_xlim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_ylim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_to_file:
        if filename is None:
            filename = f"drone_delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}.png")
        else:
            filepath = f"{filename}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"可视化图片已保存: {filepath}")

    plt.show()


def print_results(clients, routes, num_drones=NUM_DRONES, save_to_file=False,
                  filename=None, output_dir=None):
    """打印关键指标和路径详情"""
    total_distance = sum(route['distance'] for route in routes)
    total_deliveries = sum(route['deliveries'] for route in routes)
    makespan = max(r['arrive_time'] for r in routes) if routes else 0

    result_lines = []
    result_lines.append("\n" + "=" * 60)
    result_lines.append("无人机配送路径规划结果（贪心算法）")
    result_lines.append("=" * 60)
    result_lines.append(f"总客户数: {len(clients)}")
    result_lines.append(f"总趟次: {len(routes)}")
    result_lines.append(f"总飞行距离: {total_distance:.2f} 单位")
    result_lines.append(f"总完成时间(makespan): {makespan:.2f} 时间单位")
    result_lines.append(f"总配送次数: {total_deliveries}")
    result_lines.append("=" * 60)

    # 按无人机分组统计趟次
    route_by_drone = {}
    for route in routes:
        did = route.get('drone_id', 0)
        route_by_drone.setdefault(did, []).append(route)

    for did in sorted(route_by_drone):
        for t_idx, route in enumerate(route_by_drone[did]):
            result_lines.append(f"\n无人机 {did + 1} 第 {t_idx + 1} 趟:")
            result_lines.append(f"  路径顺序: 配送中心 -> {' -> '.join(str(i) for i in route['route'])} -> 配送中心")
            result_lines.append(f"  配送客户编号: {route['route']}")
            result_lines.append(f"  各客户重量: {[int(clients[i, 2]) for i in route['route']]} kg")
            result_lines.append(f"  总载重: {route['load']:.1f} kg (限制: {MAX_CAPACITY} kg)")
            result_lines.append(f"  总里程: {route['distance']:.2f} 单位 (限制: {MAX_DISTANCE} 单位)")
            result_lines.append(f"  出发时刻: {route['depart_time']:.1f}  "
                             f"返航时刻: {route['arrive_time']:.1f}")
            result_lines.append(f"  配送次数: {route['deliveries']} 次")

    print("\n".join(result_lines))

    if save_to_file:
        if filename is None:
            from datetime import datetime
            filename = f"drone_delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}.txt")
        else:
            filepath = f"{filename}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(result_lines))
        print(f"\n结果已保存: {filepath}")

    return "\n".join(result_lines)


def _assign_routes_to_drones(routes, num_drones):
    """将路线分配给无人机机队，计算各趟次的出发/返航时刻。

    贪心算法本身不建模并行调度，此函数作为后处理步骤，
    通过 FleetScheduler 将路线分配给最早空闲的无人机，得到 makespan。
    若路线包含 _tw_duration（时间窗模式），使用含等待+服务的实际耗时。
    """
    scheduler = FleetScheduler(num_drones, DRONE_SPEED)
    for route in routes:
        dist = route['distance']
        duration = route.get('_tw_duration')  # None 表示无时间窗，用 distance/speed
        drone_id, depart, arrive = scheduler.assign(dist, duration=duration)
        route['drone_id'] = drone_id
        route['depart_time'] = depart
        route['arrive_time'] = arrive
    return routes


def run_greedy(clients=None, output_dir=None, time_windows=None, num_drones=None):
    """运行贪心算法并返回结果（可由外部模块调用的统一入口）"""
    from datetime import datetime

    if num_drones is None:
        num_drones = NUM_DRONES

    if clients is None:
        clients = generate_simulation_data()
        clients = clean_data(clients)

    dist_matrix = compute_distance_matrix(clients)
    routes = greedy_assignment(clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE,
                                time_windows=time_windows)

    # 后处理：通过 FleetScheduler 分配无人机，计算时序信息
    routes = _assign_routes_to_drones(routes, num_drones)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"drone_delivery_{timestamp}"

    # 输出文本结果和可视化到文件
    print_results(clients, routes, num_drones, save_to_file=True,
                  filename=result_filename, output_dir=output_dir)
    plot_results(clients, DEPOT_COORDS, routes, save_to_file=True,
                 filename=result_filename, output_dir=output_dir)

    # 聚合统计指标并返回
    total_distance = sum(r['distance'] for r in routes)
    makespan = max(r['arrive_time'] for r in routes) if routes else 0
    return {
        'clients': clients,
        'routes': routes,
        'dist_matrix': dist_matrix,
        'total_distance': total_distance,
        'makespan': makespan,
        'total_trips': len(routes),
        'total_deliveries': sum(r['deliveries'] for r in routes),
    }


def main():
    print("=" * 60)
    print("无人机配送路径规划 - 贪心算法基线版本")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description='贪心算法无人机配送路径规划')
    parser.add_argument('--output-dir', default='../outputs', help='输出目录')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    print("\n【步骤1】生成仿真数据")
    clients = generate_simulation_data()
    print("\n【步骤2】数据清洗")
    clients = clean_data(clients)
    print("\n【步骤3】计算距离矩阵")
    dist_matrix = compute_distance_matrix(clients)
    print("\n【步骤4】贪心算法路径分配")
    routes = greedy_assignment(clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE)
    print("\n【步骤5】输出结果")
    print_results(clients, routes, save_to_file=True, output_dir=output_dir)
    print("\n【步骤6】可视化")
    plot_results(clients, DEPOT_COORDS, routes, save_to_file=True, output_dir=output_dir)


if __name__ == "__main__":
    main()
