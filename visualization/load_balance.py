#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
负载分布可视化 — 图8: 负载分布堆叠柱状图 + 基尼系数
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import BG, PANEL, GRID, TEXT_PRI, TEXT_SEC, PALETTE, DRONE_CAPACITY

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')


def plot_load_distribution(problem, routes_dict, algo_labels,
                           save_to_file=True, output_dir=None):
    """图8: 负载分布 — 堆叠柱状图 + 基尼系数"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    algo_keys = list(routes_dict.keys())
    n_algos = len(algo_keys)

    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 6), facecolor=BG)
    if n_algos == 1:
        axes = [axes]

    for ax, algo_key in zip(axes, algo_keys):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_SEC, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4, axis='y')

        routes = routes_dict[algo_key]
        eval_result = problem.evaluate_solution(routes)

        # 收集各无人机载重
        loads = []
        drone_labels = []
        active_count = 0
        for k, route in enumerate(routes):
            if route:
                load = sum(problem.customers[i].demand for i in route)
                loads.append(load)
                drone_labels.append(f'#{k + 1}')
                active_count += 1

        if loads:
            colors = [PALETTE[k % len(PALETTE)] for k in range(len(loads))]
            bars = ax.bar(drone_labels, loads, color=colors, alpha=0.8,
                          edgecolor=BG, linewidth=1)

            # 载重上限线
            ax.axhline(y=DRONE_CAPACITY, color='#f78166', linewidth=1.5,
                       linestyle='--', alpha=0.7, label=f'载重上限 ({DRONE_CAPACITY}kg)')

            for bar, load in zip(bars, loads):
                utilization = load / DRONE_CAPACITY * 100
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{load:.1f}kg\n({utilization:.0f}%)',
                        ha='center', va='bottom', color=TEXT_PRI, fontsize=7)

        name = algo_labels.get(algo_key, algo_key)
        ax.set_title(f'{name}\n基尼系数: {eval_result["load_gini"]:.3f}',
                     color=TEXT_PRI, fontsize=10, fontweight='bold')
        ax.set_ylabel('载重 (kg)', color=TEXT_SEC, fontsize=9)
        ax.set_ylim(0, DRONE_CAPACITY * 1.3)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT_PRI)

    fig.suptitle('各无人机载重分布', color=TEXT_PRI, fontsize=14, fontweight='bold', y=1.02)

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig8_load_distribution.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图8已保存: {path}")
    plt.close(fig)
    return fig
