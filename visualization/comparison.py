#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
算法对比可视化 — 图4/6/7: 成本对比柱状图、多指标对比、运行时间vs成本散点图
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import BG, PANEL, GRID, TEXT_PRI, TEXT_SEC, PALETTE

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')


def plot_cost_comparison(summary_dict, algo_labels, save_to_file=True, output_dir=None):
    """图4: 算法成本对比 — 柱状图 + 误差棒"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4, axis='y')

    algo_keys = list(summary_dict.keys())
    names = [algo_labels.get(k, k) for k in algo_keys]
    means = [summary_dict[k]['total_cost']['mean'] for k in algo_keys]
    stds = [summary_dict[k]['total_cost']['std'] for k in algo_keys]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(algo_keys))]

    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.8,
                  edgecolor=BG, linewidth=1.2, capsize=8,
                  error_kw=dict(ecolor=TEXT_SEC, capthick=1))

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 2,
                f'{mean:.1f}', ha='center', va='bottom', color=TEXT_PRI, fontsize=10,
                fontweight='bold')

    ax.set_ylabel('总成本 (元)', color=TEXT_SEC, fontsize=11)
    ax.set_title('算法总成本对比 (mean ± std)', color=TEXT_PRI, fontsize=13, fontweight='bold')

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig4_cost_comparison.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图4已保存: {path}")
    plt.close(fig)
    return fig


def plot_multi_metric_comparison(summary_dict, algo_labels,
                                 save_to_file=True, output_dir=None):
    """图6: 多指标对比 — 分组柱状图"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    metrics = [
        ('total_cost', '总成本(元)'),
        ('total_distance', '总距离(km)'),
        ('makespan', 'Makespan(min)'),
        ('runtime', '运行时间(s)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    axes = axes.flatten()
    algo_keys = list(summary_dict.keys())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(algo_keys))]
    names = [algo_labels.get(k, k) for k in algo_keys]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_SEC, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4, axis='y')

        means = [summary_dict[k][metric_key]['mean'] for k in algo_keys]
        stds = [summary_dict[k][metric_key]['std'] for k in algo_keys]

        bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.8,
                      edgecolor=BG, linewidth=1, capsize=6,
                      error_kw=dict(ecolor=TEXT_SEC, capthick=1))

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{mean:.1f}', ha='center', va='bottom', color=TEXT_PRI,
                    fontsize=8, fontweight='bold')

        ax.set_title(metric_label, color=TEXT_PRI, fontsize=11, fontweight='bold')

    fig.suptitle('多指标对比', color=TEXT_PRI, fontsize=14, fontweight='bold', y=1.02)

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig6_multi_metric.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图6已保存: {path}")
    plt.close(fig)
    return fig


def plot_runtime_vs_cost(all_results, algo_labels,
                         save_to_file=True, output_dir=None):
    """图7: 运行时间 vs 成本 — 散点图"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4)

    markers = ['o', 's', '^', 'D', 'v']

    for idx, (algo_name, results) in enumerate(all_results.items()):
        color = PALETTE[idx % len(PALETTE)]
        marker = markers[idx % len(markers)]
        costs = [r['total_cost'] for r in results]
        times = [r['runtime'] for r in results]
        label = algo_labels.get(algo_name, algo_name)
        ax.scatter(times, costs, c=color, marker=marker, s=50, alpha=0.7,
                   label=label, edgecolors=BG, linewidths=0.5)

    ax.set_xlabel('运行时间 (s)', color=TEXT_SEC, fontsize=11)
    ax.set_ylabel('总成本 (元)', color=TEXT_SEC, fontsize=11)
    ax.set_title('运行时间 vs 成本', color=TEXT_PRI, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT_PRI)

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig7_runtime_vs_cost.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图7已保存: {path}")
    plt.close(fig)
    return fig
