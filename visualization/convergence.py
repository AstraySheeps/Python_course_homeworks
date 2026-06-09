#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
收敛曲线可视化 — 图5: SA收敛曲线 + GA进化曲线
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import BG, PANEL, GRID, TEXT_PRI, TEXT_SEC, PALETTE

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')


def plot_convergence_curves(sa_history=None, ga_history=None,
                            save_to_file=True, output_dir=None):
    """图5: 收敛曲线 — SA成本vs迭代 + GA最优/平均适应度vs代数"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    n_plots = sum([sa_history is not None, ga_history is not None])
    if n_plots == 0:
        return None

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), facecolor=BG)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if sa_history is not None:
        ax = axes[plot_idx]
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

        iters = [h[0] for h in sa_history]
        costs = [h[1] for h in sa_history]
        ax.plot(iters, costs, color=PALETTE[2], linewidth=1.2, alpha=0.9)
        ax.set_xlabel('迭代次数', color=TEXT_SEC, fontsize=9)
        ax.set_ylabel('最优成本 (元)', color=TEXT_SEC, fontsize=9)
        ax.set_title('模拟退火收敛曲线', color=TEXT_PRI, fontsize=11, fontweight='bold')

        # 标注最终值
        ax.scatter([iters[-1]], [costs[-1]], color='#f78166', s=40, zorder=6)
        ax.annotate(f'{costs[-1]:.1f}', xy=(iters[-1], costs[-1]),
                    xytext=(iters[-1] * 0.9, costs[0] + (costs[-1] - costs[0]) * 0.3),
                    color=TEXT_PRI, fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=TEXT_SEC, lw=0.8))
        plot_idx += 1

    if ga_history is not None:
        ax = axes[plot_idx]
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

        gens = [h[0] for h in ga_history]
        mins = [h[1] for h in ga_history]

        ax.plot(gens, mins, color=PALETTE[1], linewidth=2.0, alpha=0.9,
                label='最优适应度')

        # 标注最优
        best_idx = np.argmin(mins)
        ax.scatter([gens[best_idx]], [mins[best_idx]], color='#f78166', s=40, zorder=6)
        ax.annotate(f'{mins[best_idx]:.1f}', xy=(gens[best_idx], mins[best_idx]),
                    xytext=(gens[best_idx] + len(gens) * 0.05,
                            mins[best_idx] + (max(mins) - min(mins)) * 0.1),
                    color=TEXT_PRI, fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=TEXT_SEC, lw=0.8))

        ax.set_xlabel('代数', color=TEXT_SEC, fontsize=9)
        ax.set_ylabel('最优成本 (元)', color=TEXT_SEC, fontsize=9)
        ax.set_title('遗传算法进化曲线', color=TEXT_PRI, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT_PRI)

    fig.suptitle('算法收敛性分析', color=TEXT_PRI, fontsize=13, fontweight='bold', y=1.02)

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'fig5_convergence.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"图5已保存: {path}")
    plt.close(fig)
    return fig
