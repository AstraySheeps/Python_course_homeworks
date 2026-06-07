#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 共享模块
提供：配置常量、仿真数据生成、数据清洗、距离矩阵计算、暗色主题配色
"""

import numpy as np

# ==================== 模拟场景配置 ====================
RANDOM_SEED = 42          # 默认随机种子（确保结果可复现）
NUM_CLIENTS = 20          # 客户点数量
COORD_RANGE = [0, 100]    # 坐标范围（正方形区域）
WEIGHT_RANGE = [1, 5]     # 包裹重量范围（kg，均匀整数）
NUM_DRONES = 3            # 无人机数量（贪心算法默认值）
MAX_CAPACITY = 20         # 单架最大载重（kg）
MAX_DISTANCE = 200        # 单趟最大飞行里程（欧氏距离单位）
DRONE_SPEED = 10.0        # 飞行速度（距离单位/时间单位）
DEPOT_COORDS = [50, 50]   # 配送中心坐标
PENALTY_WEIGHT = 1e4      # 约束违反惩罚系数（超重/超距的放大倍数）

# ==================== 暗色主题配色（所有算法共用） ====================
BG = "#0d1117"            # 图表背景
PANEL = "#161b22"          # 子图/面板背景
GRID = "#21262d"           # 网格线和边框色
TEXT_PRI = "#e6edf3"       # 主文字颜色
TEXT_SEC = "#8b949e"       # 次要文字颜色
DEPOT_COL = "#ff6b6b"      # 配送中心标记色
PALETTE = [                # 无人机路线色板（最多16架不同颜色）
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
    "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb347",
    "#63e6be", "#f8a5c2", "#a9dc76", "#fc9867", "#ab9df2", "#78dce8",
]


def generate_simulation_data(num_clients=NUM_CLIENTS, coord_range=None,
                             weight_range=None, seed=RANDOM_SEED):
    """生成仿真客户数据。

    Args:
        num_clients: 客户点数量
        coord_range: [min, max] 坐标范围
        weight_range: [min, max] 重量范围（整数）
        seed: 随机种子（保证可复现）

    Returns:
        shape=(n, 3) 的 ndarray，每行为 [x, y, weight]
    """
    if coord_range is None:
        coord_range = COORD_RANGE
    if weight_range is None:
        weight_range = WEIGHT_RANGE

    np.random.seed(seed)
    coords = np.random.uniform(coord_range[0], coord_range[1], size=(num_clients, 2))
    weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=(num_clients, 1))
    clients = np.hstack([coords, weights])
    print(f"生成了 {num_clients} 个原始客户点")
    return clients


def clean_data(clients, coord_range=None, weight_range=None):
    """数据清洗：依次完成坐标越界剔除 → 重量异常过滤 → 坐标去重。

    Args:
        clients: shape=(n, 3) 原始客户数据
        coord_range: 合法坐标范围
        weight_range: 合法重量范围

    Returns:
        清洗后的客户数据数组
    """
    if coord_range is None:
        coord_range = COORD_RANGE
    if weight_range is None:
        weight_range = WEIGHT_RANGE

    original_count = len(clients)
    mask = (
        (clients[:, 0] >= coord_range[0]) & (clients[:, 0] <= coord_range[1])
        & (clients[:, 1] >= coord_range[0]) & (clients[:, 1] <= coord_range[1])
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


def compute_distance_matrix(clients, depot=None):
    """计算欧氏距离矩阵。

    第 0 行/列为配送中心，第 1..n 行/列为各客户点。
    矩阵对称，d[i][j] = d[j][i]。

    Args:
        clients: shape=(n, 3) 客户数据
        depot: 配送中心坐标，默认使用 DEPOT_COORDS

    Returns:
        shape=(n+1, n+1) 的对称距离矩阵
    """
    if depot is None:
        depot = DEPOT_COORDS

    n = len(clients)
    all_points = np.vstack([depot, clients[:, :2]])
    dist_matrix = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            dist_matrix[i, j] = np.linalg.norm(all_points[i] - all_points[j])
    print(f"距离矩阵计算完成，维度: {dist_matrix.shape}")
    return dist_matrix
