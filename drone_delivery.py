#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 贪心算法基线版本
功能：仿真数据生成 + 数据清洗 + 距离矩阵 + 贪心算法 + 结果输出 + 可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置参数 ====================
RANDOM_SEED = 42              # 随机种子，确保结果可复现
NUM_CLIENTS = 20              # 客户点数量
COORD_RANGE = [0, 100]        # 坐标范围 [min, max]
WEIGHT_RANGE = [1, 5]         # 包裹重量范围 [min, max] kg
NUM_DRONES = 3                # 无人机数量
MAX_CAPACITY = 20             # 单架无人机最大载重 kg
MAX_DISTANCE = 200            # 单架无人机最大飞行里程
DEPOT_COORDS = [50, 50]       # 配送中心坐标


def generate_simulation_data(num_clients, coord_range, weight_range, seed=RANDOM_SEED):
    """
    生成仿真客户数据
    :param num_clients: 客户数量
    :param coord_range: 坐标范围 [min, max]
    :param weight_range: 重量范围 [min, max]
    :param seed: 随机种子
    :return: 客户数据数组，shape=(num_clients, 3)，每行为 [x, y, weight]
    """
    np.random.seed(seed)
    
    # 生成坐标
    coords = np.random.uniform(coord_range[0], coord_range[1], size=(num_clients, 2))
    
    # 生成包裹重量（整数）
    weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=(num_clients, 1))
    
    # 组合数据
    clients = np.hstack([coords, weights])
    
    print(f"生成了 {num_clients} 个原始客户点")
    return clients


def clean_data(clients, coord_range, weight_range):
    """
    数据清洗：剔除异常值和重复点
    :param clients: 原始客户数据
    :param coord_range: 坐标范围
    :param weight_range: 重量范围
    :return: 清洗后的客户数据
    """
    original_count = len(clients)
    
    # 1. 剔除坐标异常点（超出范围）
    mask = (clients[:, 0] >= coord_range[0]) & (clients[:, 0] <= coord_range[1]) & \
           (clients[:, 1] >= coord_range[0]) & (clients[:, 1] <= coord_range[1])
    clients = clients[mask]
    
    # 2. 剔除重量异常点
    mask = (clients[:, 2] >= weight_range[0]) & (clients[:, 2] <= weight_range[1])
    clients = clients[mask]
    
    # 3. 剔除重复点（坐标相同视为重复）
    unique_coords, unique_indices = np.unique(clients[:, :2], axis=0, return_index=True)
    clients = clients[unique_indices]
    
    cleaned_count = len(clients)
    removed_count = original_count - cleaned_count
    
    print(f"数据清洗完成：原始 {original_count} 个点，清洗后 {cleaned_count} 个点，剔除 {removed_count} 个异常/重复点")
    return clients


def compute_distance_matrix(clients, depot):
    """
    计算距离矩阵
    :param clients: 客户数据，shape=(n, 3)
    :param depot: 配送中心坐标 [x, y]
    :return: 距离矩阵，shape=(n+1, n+1)，第0行为配送中心到各点距离
    """
    n = len(clients)
    # 将配送中心作为第0个点
    all_points = np.vstack([depot, clients[:, :2]])
    
    # 计算欧氏距离矩阵
    dist_matrix = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            dist_matrix[i, j] = np.linalg.norm(all_points[i] - all_points[j])
    
    print(f"距离矩阵计算完成，维度: {dist_matrix.shape}")
    return dist_matrix


def greedy_assignment(clients, dist_matrix, max_capacity, max_distance):
    """
    贪心算法分配路径
    :param clients: 客户数据，shape=(n, 3)
    :param dist_matrix: 距离矩阵
    :param max_capacity: 最大载重
    :param max_distance: 最大里程
    :return: 无人机路径列表，每个路径包含: (客户索引列表, 总载重, 总里程)
    """
    n = len(clients)
    unassigned = set(range(n))  # 未分配的客户索引
    drone_routes = []           # 存储每架无人机的路径
    
    depot_idx = 0  # 配送中心在距离矩阵中的索引
    
    while unassigned:
        current_route = []
        current_load = 0
        current_distance = 0
        current_pos = depot_idx  # 当前位置，从配送中心出发
        
        while unassigned:
            # 找到最近的未分配客户
            min_dist = float('inf')
            next_client = None
            
            for client_idx in unassigned:
                # 客户在距离矩阵中的索引为 client_idx + 1
                dist = dist_matrix[current_pos, client_idx + 1]
                if dist < min_dist:
                    min_dist = dist
                    next_client = client_idx
            
            if next_client is None:
                break
            
            # 检查约束
            client_weight = clients[next_client, 2]
            # 往返距离：当前位置到客户 + 客户返回配送中心
            required_distance = current_distance + min_dist + dist_matrix[next_client + 1, depot_idx]
            required_load = current_load + client_weight
            
            if required_load <= max_capacity and required_distance <= max_distance:
                current_route.append(next_client)
                current_load += client_weight
                current_distance += min_dist
                current_pos = next_client + 1  # 更新当前位置
                unassigned.remove(next_client)
            else:
                # 无法继续分配，当前无人机返回配送中心
                break
        
        # 计算返回配送中心的距离
        if current_route:
            current_distance += dist_matrix[current_pos, depot_idx]
            drone_routes.append({
                'route': current_route,
                'load': current_load,
                'distance': current_distance,
                'deliveries': len(current_route)
            })
    
    print(f"贪心分配完成，共使用 {len(drone_routes)} 架无人机")
    return drone_routes


def plot_results(clients, depot, routes, save_to_file=False, filename=None):
    """
    可视化配送路径，可选保存到图片文件
    :param clients: 客户数据
    :param depot: 配送中心坐标
    :param routes: 无人机路径列表
    :param save_to_file: 是否保存到图片文件
    :param filename: 保存文件名（不含扩展名）
    """
    from matplotlib.patches import FancyArrowPatch
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制配送中心（缩小尺寸）
    ax.scatter(depot[0], depot[1], c='red', s=120, marker='s', label='配送中心', zorder=5)
    
    # 3架飞机对应3种颜色，超过3趟则循环使用
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    num_drones = 3
    
    # 统计每趟经过的客户点索引
    trip_clients = {}
    for idx, route in enumerate(routes):
        trip_num = idx // num_drones + 1
        if trip_num not in trip_clients:
            trip_clients[trip_num] = []
        trip_clients[trip_num].extend(route['route'])
    
    # 不同趟次使用不同颜色（统一圆形）
    trip_colors = ['#4169E1', '#FF6347', '#32CD32', '#9932CC', '#FFD700']
    
    # 绘制各趟次的客户点（统一使用圆形）
    for trip_num, client_indices in trip_clients.items():
        color = trip_colors[(trip_num - 1) % len(trip_colors)]
        unique_clients = list(set(client_indices))
        client_coords = clients[unique_clients, :2]
        ax.scatter(client_coords[:, 0], client_coords[:, 1], 
                   c=color, s=100, marker='o', 
                   label=f'客户点（第{trip_num}趟）', zorder=4)
    
    # 标注客户点编号和重量
    for i, (x, y, w) in enumerate(clients):
        ax.text(x + 1.5, y + 1.5, f'{i}({int(w)}kg)', fontsize=9)
    
    # 绘制各无人机路径
    for idx, route in enumerate(routes):
        drone_id = idx % num_drones
        trip_num = idx // num_drones + 1
        color = colors[drone_id]
        
        path = [depot] + [clients[i, :2] for i in route['route']] + [depot]
        path = np.array(path)
        
        # 绘制带箭头的路径
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='-|>',
                mutation_scale=15,
                color=color,
                linewidth=2,
                zorder=3
            )
            ax.add_patch(arrow)
        
        # 绘制载重和里程信息（在路径中点）
        mid_x = np.mean(path[:, 0])
        mid_y = np.mean(path[:, 1])
        ax.text(mid_x, mid_y - 2, 
                f'飞机{drone_id + 1}第{trip_num}趟\n载重:{route["load"]:.1f}kg\n里程:{route["distance"]:.1f}',
                fontsize=8, color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                zorder=6)
    
    # 绘制无人机约束圆（以配送中心为中心）
    circle = Circle(depot, MAX_DISTANCE/2, color='gray', linestyle='--', 
                    fill=False, alpha=0.3, label=f'最大里程半径({MAX_DISTANCE/2})')
    ax.add_patch(circle)
    
    # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[0], linewidth=2, label='飞机1'),
        Line2D([0], [0], color=colors[1], linewidth=2, label='飞机2'),
        Line2D([0], [0], color=colors[2], linewidth=2, label='飞机3'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='配送中心'),
    ]
    # 添加各趟次的客户点图例（统一圆形）
    for trip_num in sorted(trip_clients.keys()):
        color = trip_colors[(trip_num - 1) % len(trip_colors)]
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=10, label=f'客户点（第{trip_num}趟）')
        )
    
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    
    ax.set_title('无人机配送路径规划', fontsize=16)
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_ylim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # 保存图片
    if save_to_file:
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drone_delivery_plot_{timestamp}"
        
        png_filename = f"{filename}.png"
        plt.savefig(png_filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"可视化图片已保存到文件: {png_filename}")
    
    plt.show()


def print_results(clients, routes, save_to_file=False, filename=None):
    """
    打印关键指标和路径详情，可选保存到文件
    :param clients: 客户数据
    :param routes: 无人机路径列表
    :param save_to_file: 是否保存到文件
    :param filename: 保存文件名（不含扩展名）
    :return: 结果字符串
    """
    total_distance = sum(route['distance'] for route in routes)
    total_deliveries = sum(route['deliveries'] for route in routes)
    
    result_lines = []
    result_lines.append("\n" + "="*60)
    result_lines.append("无人机配送路径规划结果")
    result_lines.append("="*60)
    result_lines.append(f"总客户数: {len(clients)}")
    result_lines.append(f"使用无人机数: {len(routes)}")
    result_lines.append(f"总飞行距离: {total_distance:.2f} 单位")
    result_lines.append(f"总配送次数: {total_deliveries}")
    result_lines.append("="*60)
    
    for idx, route in enumerate(routes):
        drone_id = idx % 3
        trip_num = idx // 3 + 1
        result_lines.append(f"\n无人机 {drone_id + 1} 第 {trip_num} 趟:")
        result_lines.append(f"  路径顺序: 配送中心 -> {' -> '.join([str(i) for i in route['route']])} -> 配送中心")
        result_lines.append(f"  配送客户编号: {route['route']}")
        result_lines.append(f"  各客户重量: {[int(clients[i, 2]) for i in route['route']]} kg")
        result_lines.append(f"  总载重: {route['load']:.1f} kg (限制: {MAX_CAPACITY} kg)")
        result_lines.append(f"  总里程: {route['distance']:.2f} 单位 (限制: {MAX_DISTANCE} 单位)")
        result_lines.append(f"  配送次数: {route['deliveries']} 次")
    
    # 打印到控制台
    print("\n".join(result_lines))
    
    # 保存到文件
    if save_to_file:
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drone_delivery_result_{timestamp}"
        
        txt_filename = f"{filename}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(result_lines))
        print(f"\n结果已保存到文件: {txt_filename}")
    
    return "\n".join(result_lines)


def main():
    """
    主函数：完整执行流程
    """
    from datetime import datetime
    
    print("="*60)
    print("无人机配送路径规划 - 贪心算法基线版本")
    print("="*60)
    
    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"drone_delivery_{timestamp}"
    
    # 1. 生成仿真数据
    print("\n【步骤1】生成仿真数据")
    clients = generate_simulation_data(NUM_CLIENTS, COORD_RANGE, WEIGHT_RANGE)
    
    # 2. 数据清洗
    print("\n【步骤2】数据清洗")
    clients = clean_data(clients, COORD_RANGE, WEIGHT_RANGE)
    
    # 3. 计算距离矩阵
    print("\n【步骤3】计算距离矩阵")
    dist_matrix = compute_distance_matrix(clients, DEPOT_COORDS)
    
    # 4. 贪心算法分配
    print("\n【步骤4】贪心算法路径分配")
    routes = greedy_assignment(clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE)
    
    # 5. 输出结果（保存到txt文件）
    print("\n【步骤5】输出结果")
    print_results(clients, routes, save_to_file=True, filename=result_filename)
    
    # 6. 可视化（保存到png文件）
    print("\n【步骤6】可视化")
    plot_results(clients, DEPOT_COORDS, routes, save_to_file=True, filename=result_filename)


if __name__ == "__main__":
    main()
