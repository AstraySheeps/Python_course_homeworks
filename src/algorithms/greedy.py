#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贪心算法 — 基线求解器
策略：最近客户优先 + 约束检查
变体：prioritize_urgent 优先紧急客户版本
"""

import numpy as np
from .base import BaseAlgorithm


def _decode_greedy(problem, order, check_range=True):
    """按给定顺序贪心解码为客户路线

    使用最佳适应(Best-Fit)策略：对每个客户，选择剩余容量最充足且能装入的无人机。
    确保所有客户都被分配且不超载。

    Args:
        problem: Problem 实例
        order: 客户访问顺序（索引列表）
        check_range: 是否检查航程约束

    Returns:
        list[list[int]] 路线列表
    """
    n = problem.n
    m = problem.m
    routes = [[] for _ in range(m)]

    for ci in order:
        customer = problem.customers[ci]
        demand = customer.demand

        # 寻找能装入且剩余容量最充足的无人机
        best_drone = -1
        best_slack = -1
        for k in range(m):
            drone = problem.drones[k]
            current_load = sum(problem.customers[i].demand for i in routes[k])
            if current_load + demand <= drone.capacity:
                slack = drone.capacity - (current_load + demand)
                if slack > best_slack:
                    best_slack = slack
                    best_drone = k

        if best_drone >= 0:
            routes[best_drone].append(ci)
        else:
            # 所有无人机都无法装入 → 放入负载最轻的无人机（超载由罚金处理）
            loads = [sum(problem.customers[i].demand for i in routes[k]) for k in range(m)]
            min_load_k = min(range(m), key=lambda k: loads[k])
            routes[min_load_k].append(ci)

    # 趟内 EDD 排序（按截止时间升序），减少延迟
    for k in range(len(routes)):
        if routes[k]:
            routes[k].sort(key=lambda ci: problem.customers[ci].time_window[1])

    # 过滤空路线
    return [r for r in routes if r]


class GreedyAlgorithm(BaseAlgorithm):
    """最近客户优先贪心算法"""

    def __init__(self, problem, prioritize_urgent=False, config=None):
        super().__init__(problem, config)
        self.prioritize_urgent = prioritize_urgent

    def solve(self):
        problem = self.problem
        n = problem.n
        m = problem.m
        dist = problem.dist_matrix

        unvisited = set(range(n))

        if self.prioritize_urgent:
            # 按类型排序: urgent → normal → relaxed
            type_order = {'urgent': 0, 'normal': 1, 'relaxed': 2}
            order = sorted(range(n), key=lambda i: (
                type_order.get(problem.customers[i].customer_type, 1),
                dist[0, i + 1]
            ))
        else:
            # 按距离排序（最近优先）
            order = sorted(range(n), key=lambda i: dist[0, i + 1])

        routes = _decode_greedy(problem, order)

        eval_result = problem.evaluate_solution(routes)
        self.best_solution = routes
        self.best_cost = eval_result['total_cost']
        self.history = [(0, self.best_cost)]

        return routes, self.best_cost, self.history


def solve_greedy(problem, seed=None):
    """贪心算法求解入口"""
    algo = GreedyAlgorithm(problem)
    return algo.solve()


def solve_greedy_urgent(problem, seed=None):
    """优先紧急客户贪心算法求解入口"""
    algo = GreedyAlgorithm(problem, prioritize_urgent=True)
    return algo.solve()
