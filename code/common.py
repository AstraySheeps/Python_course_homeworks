#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 共享模块
提供：配置常量、仿真数据生成、数据清洗、距离矩阵计算
"""

import numpy as np

# ==================== 配置参数 ====================
RANDOM_SEED = 42
NUM_CLIENTS = 20
COORD_RANGE = [0, 100]
WEIGHT_RANGE = [1, 5]
NUM_DRONES = 3
MAX_CAPACITY = 20
MAX_DISTANCE = 200
DRONE_SPEED = 10.0
DEPOT_COORDS = [50, 50]


def generate_simulation_data(num_clients=NUM_CLIENTS, coord_range=None,
                             weight_range=None, seed=RANDOM_SEED):
    """生成仿真客户数据，返回 shape=(n, 3) 数组 [x, y, weight]"""
    if coord_range is None:
        coord_range = COORD_RANGE
    if weight_range is None:
        weight_range = WEIGHT_RANGE

    np.random.seed(seed)
    # 在坐标范围内均匀生成 (x, y)，重量为范围内的随机整数
    coords = np.random.uniform(coord_range[0], coord_range[1], size=(num_clients, 2))
    weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=(num_clients, 1))
    # 横向拼接：每行 [x, y, weight]
    clients = np.hstack([coords, weights])
    print(f"生成了 {num_clients} 个原始客户点")
    return clients


def clean_data(clients, coord_range=None, weight_range=None):
    """数据清洗：剔除坐标/重量越界点及重复点"""
    if coord_range is None:
        coord_range = COORD_RANGE
    if weight_range is None:
        weight_range = WEIGHT_RANGE

    original_count = len(clients)
    # 第一轮筛选：剔除坐标越界的点
    mask = (
        (clients[:, 0] >= coord_range[0]) & (clients[:, 0] <= coord_range[1])
        & (clients[:, 1] >= coord_range[0]) & (clients[:, 1] <= coord_range[1])
    )
    clients = clients[mask]
    # 第二轮筛选：剔除重量越界的点
    mask = (clients[:, 2] >= weight_range[0]) & (clients[:, 2] <= weight_range[1])
    clients = clients[mask]
    # 第三轮去重：基于 (x, y) 坐标去除重复点
    _, unique_indices = np.unique(clients[:, :2], axis=0, return_index=True)
    clients = clients[unique_indices]
    print(
        f"数据清洗完成：原始 {original_count} → 清洗后 {len(clients)} 个点，"
        f"剔除 {original_count - len(clients)} 个异常/重复点"
    )
    return clients


def compute_distance_matrix(clients, depot=None):
    """计算距离矩阵（第0行/列为配送中心），返回 shape=(n+1, n+1) 数组"""
    if depot is None:
        depot = DEPOT_COORDS

    n = len(clients)
    # 第0行为配送中心，第1..n行为各客户点
    all_points = np.vstack([depot, clients[:, :2]])
    dist_matrix = np.zeros((n + 1, n + 1))
    # 计算欧氏距离对称矩阵
    for i in range(n + 1):
        for j in range(n + 1):
            dist_matrix[i, j] = np.linalg.norm(all_points[i] - all_points[j])
    print(f"距离矩阵计算完成，维度: {dist_matrix.shape}")
    return dist_matrix
