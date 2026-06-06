#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 贪心算法基线版本
功能：仿真数据生成 + 数据清洗 + 距离矩阵 + 贪心算法 + 结果输出 + 可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from common import (
    generate_simulation_data, clean_data, compute_distance_matrix,
    RANDOM_SEED, NUM_CLIENTS, COORD_RANGE, WEIGHT_RANGE,
    NUM_DRONES, MAX_CAPACITY, MAX_DISTANCE, DEPOT_COORDS,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def greedy_assignment(clients, dist_matrix, max_capacity, max_distance):
    """
    改进贪心算法分配路径。
    每趟路线中，若最近客户装不下，继续尝试次近客户，而非直接放弃。
    当所有剩余客户都无法装入当前路线时才开启新路线。
    """
    n = len(clients)
    unassigned = set(range(n))          # 未分配客户集合
    drone_routes = []
    depot_idx = 0                        # 配送中心在距离矩阵中的索引
    max_iterations = n * 10              # 安全上限，防止死循环

    iterations = 0
    while unassigned and iterations < max_iterations:
        iterations += 1
        # 开始一趟新飞行
        current_route = []
        current_load = 0
        current_distance = 0
        current_pos = depot_idx            # 从配送中心出发

        while unassigned:
            # 计算所有未分配客户到"当前位置"的距离，并排序
            candidates = []
            for client_idx in unassigned:
                dist = dist_matrix[current_pos, client_idx + 1]
                candidates.append((dist, client_idx))
            candidates.sort(key=lambda x: x[0])  # 按距离从小到大

            found = False
            for min_dist, client_idx in candidates:
                client_weight = clients[client_idx, 2]
                new_load = current_load + client_weight
                # 新总里程 = 已走 + 去该客户 + 从该客户回配送中心
                new_dist = current_distance + min_dist + dist_matrix[client_idx + 1, depot_idx]

                # 两个约束：载重上限 + 里程上限
                if new_load <= max_capacity and new_dist <= max_distance:
                    current_route.append(client_idx)
                    current_load = new_load
                    current_distance += min_dist   # 只累加去程，回程在下面统一加
                    current_pos = client_idx + 1    # +1 因为距离矩阵第0行是配送中心
                    unassigned.remove(client_idx)
                    found = True
                    break

            if not found:
                break  # 无人可装，结束当前路线

        if current_route:
            # 加上从最后一个客户返回配送中心的距离
            current_distance += dist_matrix[current_pos, depot_idx]
            drone_routes.append({
                'route': current_route,
                'load': current_load,
                'distance': current_distance,
                'deliveries': len(current_route)
            })
        else:
            # 死胡同：所有剩余客户单独都无法满足约束
            # 强制逐客户分配（每人一趟往返），并记录警告
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
    """可视化配送路径"""
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 10))

    # 配送中心 —— 红色方块
    ax.scatter(depot[0], depot[1], c='red', s=120, marker='s', label='配送中心', zorder=5)

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 三架无人机的专属颜色
    num_drones = NUM_DRONES

    # 将架次按"趟"分组（每 num_drones 架次算一次并发趟）
    trip_clients = {}
    for idx, route in enumerate(routes):
        trip_num = idx // num_drones + 1
        if trip_num not in trip_clients:
            trip_clients[trip_num] = []
        trip_clients[trip_num].extend(route['route'])

    trip_colors = ['#4169E1', '#FF6347', '#32CD32', '#9932CC', '#FFD700']

    for trip_num, client_indices in trip_clients.items():
        color = trip_colors[(trip_num - 1) % len(trip_colors)]
        unique_clients = list(set(client_indices))
        client_coords = clients[unique_clients, :2]
        ax.scatter(client_coords[:, 0], client_coords[:, 1],
                   c=color, s=100, marker='o',
                   label=f'客户点（第{trip_num}趟）', zorder=4)

    for i, (x, y, w) in enumerate(clients):
        ax.text(x + 1.5, y + 1.5, f'{i}({int(w)}kg)', fontsize=9)

    # 为每趟路线绘制从配送中心出发→服务客户→返回的箭头路径
    for idx, route in enumerate(routes):
        drone_id = idx % num_drones
        color = colors[drone_id]

        # 完整路径：配送中心 → 各客户 → 配送中心
        path = [depot] + [clients[i, :2] for i in route['route']] + [depot]
        path = np.array(path)

        # 逐段绘制箭头
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            arrow = FancyArrowPatch(
                start, end, arrowstyle='-|>', mutation_scale=15,
                color=color, linewidth=2, zorder=3
            )
            ax.add_patch(arrow)

        # 在路径中心显示该趟次的关键信息
        mid_x = np.mean(path[:, 0])
        mid_y = np.mean(path[:, 1])
        ax.text(mid_x, mid_y - 2,
                f'飞机{drone_id + 1}第{idx // num_drones + 1}趟\n'
                f'载重:{route["load"]:.1f}kg\n里程:{route["distance"]:.1f}',
                fontsize=8, color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                zorder=6)

    circle = Circle(depot, MAX_DISTANCE / 2, color='gray', linestyle='--',
                    fill=False, alpha=0.3, label=f'最大里程半径({MAX_DISTANCE/2})')
    ax.add_patch(circle)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[0], linewidth=2, label='飞机1'),
        Line2D([0], [0], color=colors[1], linewidth=2, label='飞机2'),
        Line2D([0], [0], color=colors[2], linewidth=2, label='飞机3'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='配送中心'),
    ]
    for trip_num in sorted(trip_clients.keys()):
        color = trip_colors[(trip_num - 1) % len(trip_colors)]
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=10, label=f'客户点（第{trip_num}趟）')
        )

    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    ax.set_title('无人机配送路径规划（贪心算法）', fontsize=16)
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_ylim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_to_file:
        if filename is None:
            from datetime import datetime
            filename = f"drone_delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{filename}.png")
        else:
            filepath = f"{filename}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"可视化图片已保存: {filepath}")

    plt.show()


def print_results(clients, routes, save_to_file=False, filename=None, output_dir=None):
    """打印关键指标和路径详情"""
    total_distance = sum(route['distance'] for route in routes)
    total_deliveries = sum(route['deliveries'] for route in routes)

    result_lines = []
    result_lines.append("\n" + "=" * 60)
    result_lines.append("无人机配送路径规划结果（贪心算法）")
    result_lines.append("=" * 60)
    result_lines.append(f"总客户数: {len(clients)}")
    result_lines.append(f"总架次: {len(routes)}")
    result_lines.append(f"总飞行距离: {total_distance:.2f} 单位")
    result_lines.append(f"总配送次数: {total_deliveries}")
    result_lines.append("=" * 60)

    for idx, route in enumerate(routes):
        drone_id = idx % NUM_DRONES
        trip_num = idx // NUM_DRONES + 1
        result_lines.append(f"\n无人机 {drone_id + 1} 第 {trip_num} 趟:")
        result_lines.append(f"  路径顺序: 配送中心 -> {' -> '.join(str(i) for i in route['route'])} -> 配送中心")
        result_lines.append(f"  配送客户编号: {route['route']}")
        result_lines.append(f"  各客户重量: {[int(clients[i, 2]) for i in route['route']]} kg")
        result_lines.append(f"  总载重: {route['load']:.1f} kg (限制: {MAX_CAPACITY} kg)")
        result_lines.append(f"  总里程: {route['distance']:.2f} 单位 (限制: {MAX_DISTANCE} 单位)")
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


def run_greedy(clients=None, output_dir=None):
    """运行贪心算法并返回结果（可由外部模块调用的统一入口）"""
    from datetime import datetime

    # 若无外部数据，自动生成并清洗
    if clients is None:
        clients = generate_simulation_data()
        clients = clean_data(clients)

    dist_matrix = compute_distance_matrix(clients)
    routes = greedy_assignment(clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"drone_delivery_{timestamp}"

    # 输出文本结果和可视化到文件
    print_results(clients, routes, save_to_file=True, filename=result_filename, output_dir=output_dir)
    plot_results(clients, DEPOT_COORDS, routes, save_to_file=True, filename=result_filename, output_dir=output_dir)

    # 聚合统计指标并返回
    total_distance = sum(r['distance'] for r in routes)
    return {
        'clients': clients,
        'routes': routes,
        'dist_matrix': dist_matrix,
        'total_distance': total_distance,
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
