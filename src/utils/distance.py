#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""距离计算工具"""

import numpy as np
from config import ROAD_FACTOR


def euclidean_distance(p1, p2):
    """两点欧氏距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def road_distance(p1, p2):
    """实际道路距离 = 欧氏距离 × 曲折系数"""
    return ROAD_FACTOR * euclidean_distance(p1, p2)


def compute_distance_matrix(customers, depot):
    """计算距离矩阵（含仓库为索引0）

    Returns:
        shape=(n+1, n+1) 的对称矩阵，索引0为仓库
    """
    n = len(customers)
    points = [(depot[0], depot[1])] + [(c.x, c.y) for c in customers]
    mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            d = road_distance(points[i], points[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


def compute_flight_time(dist_km, speed):
    """飞行时间（小时）"""
    return dist_km / speed
