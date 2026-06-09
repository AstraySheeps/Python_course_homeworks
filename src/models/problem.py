#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题定义模块 — 统一人民币成本函数 + 约束检查
"""

import numpy as np
from config import (
    COST_PER_KM, COST_PER_MIN, PENALTY_DELAY_PER_MIN,
    PENALTY_OVERLOAD_PER_KG, PENALTY_EXCESS_RANGE_PER_KM,
    DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE,
)


class Problem:
    """无人机配送路径规划问题实例"""

    def __init__(self, customers, drones, dist_matrix):
        self.customers = customers        # list[Customer]
        self.drones = drones              # list[Drone]
        self.dist_matrix = dist_matrix    # (n+1)×(n+1) ndarray
        self.n = len(customers)
        self.m = len(drones)

    # ==================== 路线时间链计算 ====================

    def compute_route_timeline(self, route_indices, drone_idx=0):
        """计算单条路线的时间链

        路线: 仓库 → c1 → c2 → ... → ck → 仓库
        返回详细的时间链信息和成本分解。

        Args:
            route_indices: 客户索引列表 (0-based, 对应 customers 列表)
            drone_idx: 无人机索引

        Returns:
            dict 含 flight_time, waiting_time, delay_time, service_time,
                 operation_time, total_distance, route_load, 及各成本项
        """
        drone = self.drones[drone_idx]
        speed = drone.speed
        depot_idx = 0

        result = {
            'flight_time': 0.0,      # 纯飞行时间（小时）
            'waiting_time': 0.0,     # 提前到达等待时间（小时）
            'delay_time': 0.0,       # 延迟时间（小时）
            'service_time': 0.0,     # 服务时间（小时）
            'total_distance': 0.0,   # 总飞行距离（km）
            'route_load': 0.0,       # 总载重（kg）
            'timeline': [],          # 每站详细时间
        }

        if not route_indices:
            return result

        current_pos = depot_idx
        current_time = 0.0

        for ci in route_indices:
            customer = self.customers[ci]
            dist = self.dist_matrix[current_pos, ci + 1]
            travel_time = dist / speed
            arrival = current_time + travel_time

            e, l = customer.time_window
            wait = max(0.0, e - arrival)
            delay = max(0.0, arrival - l)
            start = max(arrival, e)
            depart = start + customer.service_time

            result['flight_time'] += travel_time
            result['waiting_time'] += wait
            result['delay_time'] += delay
            result['service_time'] += customer.service_time
            result['total_distance'] += dist
            result['route_load'] += customer.demand
            result['timeline'].append({
                'customer_id': customer.id,
                'arrival': arrival,
                'wait': wait,
                'delay': delay,
                'start': start,
                'depart': depart,
            })

            current_pos = ci + 1
            current_time = depart

        # 返回仓库
        dist_back = self.dist_matrix[current_pos, depot_idx]
        result['flight_time'] += dist_back / speed
        result['total_distance'] += dist_back
        result['operation_time'] = (
            result['flight_time'] + result['waiting_time'] + result['service_time']
        )

        return result

    # ==================== 完整解评估 ====================

    def evaluate_solution(self, routes):
        """评估完整解，返回统一人民币总成本及各项分解

        Args:
            routes: list[list[int]], 每架无人机的客户访问顺序
                    e.g. [[2,5,3], [1,4], [7,8], [], ...]

        Returns:
            dict 含 total_cost 及各成本项分解
        """
        n_drones = len(routes)
        total_distance = 0.0
        total_flight_time = 0.0
        total_waiting_time = 0.0
        total_delay_time = 0.0
        total_service_time = 0.0
        total_overload = 0.0
        total_excess_range = 0.0
        route_details = []
        makespan = 0.0

        for k, route_indices in enumerate(routes):
            if not route_indices:
                route_details.append(None)
                continue

            drone = self.drones[k] if k < self.m else self.drones[0]
            timeline = self.compute_route_timeline(route_indices, k)

            total_distance += timeline['total_distance']
            total_flight_time += timeline['flight_time']
            total_waiting_time += timeline['waiting_time']
            total_delay_time += timeline['delay_time']
            total_service_time += timeline['service_time']

            # 超重罚金
            overload = max(0.0, timeline['route_load'] - drone.capacity)
            total_overload += overload

            # 超航程罚金
            excess = max(0.0, timeline['total_distance'] - drone.max_range)
            total_excess_range += excess

            # makespan (最大完成时间)
            route_end = (timeline['flight_time'] + timeline['waiting_time']
                         + timeline['service_time'])
            makespan = max(makespan, route_end)

            route_details.append(timeline)

        total_operation_time = (
            total_flight_time + total_waiting_time + total_service_time
        )

        # 统一人民币成本
        flight_cost = COST_PER_KM * total_distance
        operation_cost = COST_PER_MIN * (total_operation_time * 60)   # 转换为分钟
        delay_penalty = PENALTY_DELAY_PER_MIN * (total_delay_time * 60)
        overload_penalty = PENALTY_OVERLOAD_PER_KG * total_overload
        excess_range_penalty = PENALTY_EXCESS_RANGE_PER_KM * total_excess_range

        total_cost = (flight_cost + operation_cost + delay_penalty
                      + overload_penalty + excess_range_penalty)

        # 可行性判断（仅硬约束：载重、航程；时间窗为软约束不计入）
        is_feasible = (total_overload < 1e-6 and total_excess_range < 1e-6)

        # 负载均衡度（基尼系数）
        loads = [d['route_load'] for d in route_details if d is not None]
        gini = self._gini(loads) if loads else 0.0

        return {
            'total_cost': total_cost,
            'flight_cost': flight_cost,
            'operation_cost': operation_cost,
            'delay_penalty': delay_penalty,
            'overload_penalty': overload_penalty,
            'excess_range_penalty': excess_range_penalty,
            'total_distance': total_distance,
            'total_flight_time': total_flight_time,
            'total_waiting_time': total_waiting_time,
            'total_delay_time': total_delay_time,
            'total_service_time': total_service_time,
            'total_operation_time': total_operation_time,
            'total_overload': total_overload,
            'total_excess_range': total_excess_range,
            'makespan': makespan * 60,      # 转换为分钟
            'makespan_hours': makespan,
            'route_details': route_details,
            'is_feasible': is_feasible,
            'load_gini': gini,
            'num_active_drones': sum(1 for r in routes if r),
        }

    # ==================== 约束检查 ====================

    def check_constraints(self, routes):
        """检查解是否满足所有硬约束

        Returns:
            dict 含各约束的违反情况
        """
        violations = {
            'capacity': [],
            'range': [],
            'visit_once': False,
            'flow_conservation': True,
            'depot_depart_return': [],
        }

        visited = set()
        for k, route in enumerate(routes):
            if not route:
                continue
            drone = self.drones[k] if k < self.m else self.drones[0]

            # 容量约束
            load = sum(self.customers[i].demand for i in route)
            if load > drone.capacity + 1e-6:
                violations['capacity'].append((k, load, drone.capacity))

            # 航程约束
            timeline = self.compute_route_timeline(route, k)
            if timeline['total_distance'] > drone.max_range + 1e-6:
                violations['range'].append(
                    (k, timeline['total_distance'], drone.max_range)
                )

            # 访问唯一性检查
            for ci in route:
                if ci in visited:
                    violations['visit_once'] = True
                visited.add(ci)

        return violations

    # ==================== 辅助方法 ====================

    @staticmethod
    def _gini(values):
        """计算基尼系数"""
        if not values or sum(values) < 1e-10:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_vals)
                / (n * np.sum(sorted_vals)) - (n + 1) / n)

    def get_customer(self, idx):
        return self.customers[idx]

    @property
    def num_customers(self):
        return self.n

    @property
    def num_drones(self):
        return self.m
