#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解验证工具"""

import numpy as np


def validate_solution(problem, routes):
    """验证解的合法性，返回违反报告"""
    report = {
        'valid': True,
        'capacity_violations': [],
        'range_violations': [],
        'duplicate_visits': False,
        'missing_customers': [],
        'total_cost': None,
    }

    visited = set()
    for k, route in enumerate(routes):
        if not route:
            continue
        drone = problem.drones[k] if k < problem.m else problem.drones[0]

        # 容量
        load = sum(problem.customers[i].demand for i in route)
        if load > drone.capacity + 1e-6:
            report['capacity_violations'].append((k, load, drone.capacity))
            report['valid'] = False

        # 航程
        timeline = problem.compute_route_timeline(route, k)
        if timeline['total_distance'] > drone.max_range + 1e-6:
            report['range_violations'].append(
                (k, timeline['total_distance'], drone.max_range)
            )
            report['valid'] = False

        # 重复访问
        for ci in route:
            if ci in visited:
                report['duplicate_visits'] = True
                report['valid'] = False
            visited.add(ci)

    # 遗漏客户
    all_customers = set(range(problem.n))
    missing = all_customers - visited
    if missing:
        report['missing_customers'] = list(missing)
        report['valid'] = False

    return report
