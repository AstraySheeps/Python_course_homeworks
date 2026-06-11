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

    # ==================== 约束违反分析报告 ====================

    def get_violation_report(self, routes):
        """生成约束违反分析报告

        区分硬约束（载重、航程）和软约束（时间窗延迟），
        统计违反的无人机/客户数量、违反幅度和占比。

        Returns:
            dict 含 hard_violations, soft_violations, summary
        """
        report = {
            'hard_violations': {
                'capacity': [],          # [(drone_idx, load, capacity, excess_kg), ...]
                'range': [],             # [(drone_idx, dist, max_range, excess_km), ...]
                'visit_once': False,     # 是否存在重复访问
                'depot_depart_return': [],  # 未从仓库出发或未返回的无人机索引
            },
            'soft_violations': {
                'delay': [],             # [(customer_id, arrival, due, delay_min), ...]
                'wait': [],              # [(customer_id, ready, arrival, wait_min), ...]
            },
            'summary': {},
        }

        visited = set()
        for k, route in enumerate(routes):
            if not route:
                # 空路线：无人机未出发也未返回
                report['hard_violations']['depot_depart_return'].append(k)
                continue

            drone = self.drones[k] if k < self.m else self.drones[0]
            timeline = self.compute_route_timeline(route, k)

            # --- 硬约束：载重 ---
            load = timeline['route_load']
            if load > drone.capacity + 1e-6:
                excess = load - drone.capacity
                report['hard_violations']['capacity'].append(
                    (k, load, drone.capacity, excess))

            # --- 硬约束：航程 ---
            dist = timeline['total_distance']
            if dist > drone.max_range + 1e-6:
                excess = dist - drone.max_range
                report['hard_violations']['range'].append(
                    (k, dist, drone.max_range, excess))

            # --- 硬约束：仓库出发与返回 ---
            if len(route) == 0:
                report['hard_violations']['depot_depart_return'].append(k)

            # --- 硬约束：访问唯一性 ---
            for ci in route:
                if ci in visited:
                    report['hard_violations']['visit_once'] = True
                visited.add(ci)

            # --- 软约束：时间窗延迟和等待 ---
            for t in timeline['timeline']:
                ci = t['customer_id']
                if t['delay'] > 1e-6:
                    report['soft_violations']['delay'].append(
                        (ci, t['arrival'], self.customers[ci].time_window[1],
                         t['delay'] * 60))   # 转为分钟
                if t['wait'] > 1e-6:
                    report['soft_violations']['wait'].append(
                        (ci, self.customers[ci].time_window[0], t['arrival'],
                         t['wait'] * 60))

        # --- 汇总 ---
        n_capacity = len(report['hard_violations']['capacity'])
        n_range = len(report['hard_violations']['range'])
        n_delay_customers = len(report['soft_violations']['delay'])
        n_wait_customers = len(report['soft_violations']['wait'])

        total_overload = sum(v[3] for v in report['hard_violations']['capacity'])
        total_excess_range = sum(v[3] for v in report['hard_violations']['range'])
        total_delay_min = sum(v[3] for v in report['soft_violations']['delay'])
        total_wait_min = sum(v[3] for v in report['soft_violations']['wait'])

        n_hard = n_capacity + n_range + int(report['hard_violations']['visit_once'])
        n_soft = n_delay_customers

        report['summary'] = {
            'hard_constraint_satisfied': (n_hard == 0 and len(report['hard_violations']['depot_depart_return']) == 0),
            'hard_violation_count': n_hard,
            'capacity_violations': n_capacity,
            'total_overload_kg': total_overload,
            'range_violations': n_range,
            'total_excess_range_km': total_excess_range,
            'soft_violation_count': n_soft,
            'delay_customer_count': n_delay_customers,
            'total_delay_min': total_delay_min,
            'max_delay_min': max([v[3] for v in report['soft_violations']['delay']]) if report['soft_violations']['delay'] else 0.0,
            'wait_customer_count': n_wait_customers,
            'total_wait_min': total_wait_min,
            'delayed_customer_pct': n_delay_customers / self.n * 100 if self.n > 0 else 0,
        }

        return report

    def violation_report_markdown(self, routes, algo_name=''):
        """生成约束违反分析的 Markdown 文本"""
        report = self.get_violation_report(routes)
        s = report['summary']
        lines = []

        lines.append(f"### 约束违反分析{f' — {algo_name}' if algo_name else ''}")
        lines.append("")

        # 硬约束
        if s['hard_constraint_satisfied']:
            lines.append(f"- **硬约束**: ✅ PASS（载重≤{DRONE_CAPACITY}kg, 航程≤{DRONE_MAX_RANGE}km）")
        else:
            lines.append(f"- **硬约束**: ❌ FAIL（{s['hard_violation_count']} 项违反）")
            if s['capacity_violations'] > 0:
                for v in report['hard_violations']['capacity']:
                    drone_idx, load, cap, excess = v
                    lines.append(f"  - 无人机#{drone_idx} 超载: {load:.1f}/{cap:.0f}kg（+{excess:.1f}kg）")
            if s['range_violations'] > 0:
                for v in report['hard_violations']['range']:
                    drone_idx, dist, max_r, excess = v
                    lines.append(f"  - 无人机#{drone_idx} 超航程: {dist:.1f}/{max_r:.0f}km（+{excess:.1f}km）")
            if report['hard_violations']['visit_once']:
                lines.append(f"  - 存在客户被重复访问")
            if report['hard_violations']['depot_depart_return']:
                lines.append(f"  - 无人机#{report['hard_violations']['depot_depart_return']} 未出发")

        # 软约束
        lines.append("")
        lines.append("- **软约束（时间窗）**:")
        if s['soft_violation_count'] == 0:
            lines.append("  - 延迟: 0 个客户，全部按时送达 ✅")
        else:
            lines.append(f"  - 延迟客户: {s['delay_customer_count']}/{self.n}（{s['delayed_customer_pct']:.1f}%）")
            lines.append(f"  - 总延迟: {s['total_delay_min']:.1f} min | 平均: {s['total_delay_min']/max(s['delay_customer_count'],1):.1f} min/客户 | 最大: {s['max_delay_min']:.1f} min")
            sorted_delays = sorted(report['soft_violations']['delay'], key=lambda x: -x[3])[:3]
            for d in sorted_delays:
                cid, arrival, due, delay_min = d
                ctype = self.customers[cid].customer_type
                lines.append(f"    - 客户#{cid}（{ctype}）: 到达 {arrival:.2f}h > 截止 {due:.2f}h，延迟 {delay_min:.0f}min")
        if s['wait_customer_count'] > 0:
            lines.append(f"  - 提前到达: {s['wait_customer_count']} 个客户，总等待 {s['total_wait_min']:.1f} min")
        else:
            lines.append("  - 提前到达: 0 个客户")

        lines.append("")
        return '\n'.join(lines), report

    def print_violation_report(self, routes, algo_name=''):
        """打印格式化的约束违反分析报告（终端输出，保留兼容）"""
        md_text, report = self.violation_report_markdown(routes, algo_name)
        print(md_text)
        return report

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

