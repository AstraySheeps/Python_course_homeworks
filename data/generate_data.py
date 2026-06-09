#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据生成模块 — 支持多场景、客户类型、时间窗、可行性校验
"""

import os
import numpy as np
import pandas as pd
from config import (
    SCENARIOS, AREA_SIZE, DEPOT_COORDS, SEED,
    DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE,
    DEMAND_RANGE, SERVICE_TIME_BASE, MAX_TOTAL_DEMAND_RATIO,
    CUSTOMER_TYPE_DIST, TIME_WINDOW_SLOTS, TIME_WINDOW_WIDTHS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'data', 'output')


def generate_customers(num_customers, seed=SEED):
    """生成客户数据

    客户坐标:
        - 70% 以仓库为中心的高斯分布 (σ=7.5 km)
        - 30% 区域内均匀分布
    客户类型与时间窗按配置比例生成。

    Returns:
        list[dict] 每个元素含 id, x, y, demand, customer_type,
                   time_window_start, time_window_end, service_time
    """
    rng = np.random.RandomState(seed)
    depot_x, depot_y = DEPOT_COORDS

    # 生成坐标
    n_gaussian = int(num_customers * 0.7)
    n_uniform = num_customers - n_gaussian

    coords = []
    # 高斯分布客户（热区）
    for _ in range(n_gaussian):
        x = rng.normal(depot_x, 7.5)
        y = rng.normal(depot_y, 7.5)
        x = np.clip(x, AREA_SIZE[0] + 0.5, AREA_SIZE[1] - 0.5)
        y = np.clip(y, AREA_SIZE[0] + 0.5, AREA_SIZE[1] - 0.5)
        coords.append((x, y))

    # 均匀分布客户（边缘区域）
    for _ in range(n_uniform):
        x = rng.uniform(AREA_SIZE[0] + 0.5, AREA_SIZE[1] - 0.5)
        y = rng.uniform(AREA_SIZE[0] + 0.5, AREA_SIZE[1] - 0.5)
        coords.append((x, y))

    rng.shuffle(coords)

    # 生成需求量
    demands = rng.uniform(DEMAND_RANGE[0], DEMAND_RANGE[1], size=num_customers)

    # 生成客户类型（按比例）
    type_choices = []
    for t, ratio in CUSTOMER_TYPE_DIST.items():
        count = int(num_customers * ratio)
        type_choices.extend([t] * count)
    # 填补舍入误差
    while len(type_choices) < num_customers:
        type_choices.append('normal')
    rng.shuffle(type_choices)

    # 生成时间窗和服务时间
    customers = []
    for i in range(num_customers):
        ctype = type_choices[i]
        slot = TIME_WINDOW_SLOTS[ctype]
        width_range = TIME_WINDOW_WIDTHS[ctype]

        tw_start = round(rng.uniform(slot[0], slot[1] - width_range[0]), 2)
        tw_width = round(rng.uniform(width_range[0], width_range[1]), 2)
        tw_end = round(min(8.0, tw_start + tw_width), 2)

        # 服务时间与需求量弱正相关
        base_s = SERVICE_TIME_BASE[0] / 60.0      # 转换为小时
        svc = base_s + 0.005 * demands[i] + rng.uniform(0, 0.005) * 10
        svc = np.clip(svc, 5 / 60.0, 15 / 60.0)

        customers.append({
            'id': i,
            'x': round(coords[i][0], 2),
            'y': round(coords[i][1], 2),
            'demand': round(demands[i], 1),
            'customer_type': ctype,
            'time_window_start': tw_start,
            'time_window_end': tw_end,
            'service_time': round(svc, 4),
        })

    return customers


def feasibility_check(customers, num_drones, capacity=DRONE_CAPACITY,
                      max_ratio=MAX_TOTAL_DEMAND_RATIO):
    """校验单趟可行性：总需求 ≤ 0.85 × M × C"""
    total_demand = sum(c['demand'] for c in customers)
    total_capacity = num_drones * capacity
    limit = max_ratio * total_capacity
    ok = total_demand <= limit
    return ok, total_demand, total_capacity, limit


def generate_scenario(scenario_name, seed=SEED, output_dir=None):
    """生成指定场景的完整数据

    根据场景规模自动调整需求范围，确保满足单趟可行性：
    sum(demand) ≤ 0.85 × M × C

    Args:
        scenario_name: "small" | "standard" | "large"
        seed: 随机种子
        output_dir: 输出目录

    Returns:
        list[dict] 客户数据
    """
    cfg = SCENARIOS[scenario_name]
    num_customers = cfg['num_customers']
    num_drones = cfg['num_drones']

    total_capacity = num_drones * DRONE_CAPACITY
    limit = MAX_TOTAL_DEMAND_RATIO * total_capacity
    # 每客户最大平均需求 = limit / num_customers
    max_avg = limit / num_customers
    # 设置需求上限为 min(8, max_avg * 1.8) 确保可行性
    demand_hi = min(8.0, max_avg * 1.6)

    print(f"  [{scenario_name}] 总运力={total_capacity:.0f}kg, "
          f"85%上限={limit:.1f}kg, 每客户平均≤{max_avg:.2f}kg, "
          f"需求范围=[1, {demand_hi:.1f}]kg")

    # 重试直到满足可行性条件
    max_attempts = 50
    for attempt in range(max_attempts):
        customers = generate_customers(num_customers, seed=seed + attempt)
        # 按场景调整需求范围
        for c in customers:
            c['demand'] = round(1.0 + (c['demand'] / 8.0) * (demand_hi - 1.0), 1)
        ok, total_d, total_c, limit_val = feasibility_check(
            customers, num_drones
        )
        if ok:
            break
    else:
        # 仍失败则进一步缩小
        print(f"  [{scenario_name}] 缩小需求范围重试...")
        demand_hi *= 0.75
        for attempt in range(max_attempts):
            customers = generate_customers(num_customers, seed=seed + 100 + attempt)
            for c in customers:
                c['demand'] = round(1.0 + (c['demand'] / 8.0) * (demand_hi - 1.0), 1)
            ok, total_d, total_c, limit_val = feasibility_check(
                customers, num_drones
            )
            if ok:
                break

    # 保存 CSV
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(customers)
    csv_path = os.path.join(output_dir, f'scenario_{scenario_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"场景 [{scenario_name}]: {num_customers}客户/{num_drones}架无人机 → {csv_path}")

    return customers


def generate_all_scenarios(seed=SEED):
    """生成全部三个场景的数据"""
    for name in ['small', 'standard', 'large']:
        generate_scenario(name, seed=seed)


def load_customers_from_csv(csv_path):
    """从 CSV 加载客户数据"""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')


if __name__ == '__main__':
    generate_all_scenarios()
    print("全部场景数据生成完成！")
