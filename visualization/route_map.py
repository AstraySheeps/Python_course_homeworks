#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配送路线图可视化 — 图1/2/3: 客户分布图、最优路线图、算法对比路线图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from datetime import datetime

from config import BG, PANEL, GRID, TEXT_PRI, TEXT_SEC, DEPOT_COL, PALETTE, DEPOT_COORDS

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')


def _style_ax(ax, xlim=(0, 50), ylim=(0, 50)):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


def _draw_depot(ax, depot=DEPOT_COORDS):
    ax.scatter(*depot, s=300, color=DEPOT_COL, alpha=0.12, zorder=7, edgecolors='none')
    ax.scatter(*depot, s=100, color=DEPOT_COL, marker='*', zorder=8,
               edgecolors=BG, linewidths=0.8)


def plot_customer_distribution(customers, depot=DEPOT_COORDS,
                               save_to_file=True, output_dir=None):
    """图1: 客户分布图 — 按类型着色"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    _style_ax(ax)

    type_colors = {'urgent': '#ff6b6b', 'normal': '#58a6ff', 'relaxed': '#3fb950'}
    type_labels = {'urgent': '紧急订单', 'normal': '普通订单', 'relaxed': '宽松订单'}

    for ctype in ['urgent', 'normal', 'relaxed']:
        pts = [(c.x, c.y) for c in customers if c.customer_type == ctype]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=type_colors[ctype], label=type_labels[ctype],
                       s=40, alpha=0.8, edgecolors=BG, linewidths=0.5)

    _draw_depot(ax, depot)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT_PRI)
    ax.set_title('客户分布图', color=TEXT_PRI, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (km)', color=TEXT_SEC)
    ax.set_ylabel('Y (km)', color=TEXT_SEC)

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig1_customer_distribution.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图1已保存: {path}")
    plt.close(fig)
    return fig


def plot_optimal_routes(problem, routes, depot=DEPOT_COORDS,
                        title='最优配送路线图', save_to_file=True, output_dir=None):
    """图2: 最优配送路线图 — 使用 GA 最优解"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    _style_ax(ax)

    customers = problem.customers

    # 背景客户点
    for c in customers:
        ax.scatter(c.x, c.y, s=18, color=TEXT_SEC, alpha=0.35, zorder=2)

    _draw_depot(ax, depot)

    active_drones = [i for i, r in enumerate(routes) if r]
    for drone_idx in active_drones:
        color = PALETTE[drone_idx % len(PALETTE)]
        route = routes[drone_idx]
        path = np.array([depot] + [(customers[i].x, customers[i].y) for i in route] + [depot])

        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, alpha=0.2)
        for k in range(len(path) - 1):
            ax.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle='-|>', mutation_scale=10,
                color=color, linewidth=1.3, alpha=0.7, zorder=3))
        for ci in route:
            c = customers[ci]
            ax.scatter(c.x, c.y, s=35, color=color, zorder=5, alpha=0.7,
                       edgecolors=BG, linewidths=0.6)

    eval_result = problem.evaluate_solution(routes)
    ax.set_title(f'{title}\n总成本: {eval_result["total_cost"]:.1f}元 | '
                 f'距离: {eval_result["total_distance"]:.1f}km | '
                 f'Makespan: {eval_result["makespan"]:.1f}min',
                 color=TEXT_PRI, fontsize=12, fontweight='bold')

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig2_optimal_routes.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图2已保存: {path}")
    plt.close(fig)
    return fig


def plot_algorithm_comparison(problem, routes_dict, depot=DEPOT_COORDS,
                              save_to_file=True, output_dir=None):
    """图3: 三种算法路线对比 — 多子图并列"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    algos = list(routes_dict.keys())
    n = len(algos)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), facecolor=BG)
    if n == 1:
        axes = [axes]

    customers = problem.customers
    algo_names = {'greedy': '贪心算法', 'sa': '模拟退火', 'ga': '遗传算法',
                  'random': '随机搜索', 'greedy_urgent': '贪心(紧急优先)'}

    for ax, algo_key in zip(axes, algos):
        _style_ax(ax)

        for c in customers:
            ax.scatter(c.x, c.y, s=12, color=TEXT_SEC, alpha=0.3, zorder=2)
        _draw_depot(ax, depot)

        routes = routes_dict[algo_key]
        eval_result = problem.evaluate_solution(routes)
        active = [i for i, r in enumerate(routes) if r]

        for drone_idx in active:
            color = PALETTE[drone_idx % len(PALETTE)]
            route = routes[drone_idx]
            path = np.array([depot] + [(customers[i].x, customers[i].y) for i in route] + [depot])
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, alpha=0.2)
            for k in range(len(path) - 1):
                ax.add_patch(FancyArrowPatch(
                    path[k], path[k + 1], arrowstyle='-|>', mutation_scale=8,
                    color=color, linewidth=1.0, alpha=0.65, zorder=3))
            for ci in route:
                ax.scatter(customers[ci].x, customers[ci].y, s=25, color=color,
                           zorder=5, alpha=0.6, edgecolors=BG, linewidths=0.5)

        name = algo_names.get(algo_key, algo_key)
        ax.set_title(f'{name}\n成本:{eval_result["total_cost"]:.0f}元 '
                     f'距离:{eval_result["total_distance"]:.0f}km',
                     color=TEXT_PRI, fontsize=10, fontweight='bold')

    fig.suptitle('算法路线对比', color=TEXT_PRI, fontsize=14, fontweight='bold', y=1.01)

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig3_algo_comparison.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图3已保存: {path}")
    plt.close(fig)
    return fig
