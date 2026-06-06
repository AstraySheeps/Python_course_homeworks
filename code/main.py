#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 统一入口
支持贪心算法、遗传算法、以及两者对比
用法:
    python main.py --algo greedy          # 仅贪心算法
    python main.py --algo genetic         # 仅遗传算法
    python main.py --algo both            # 两者对比（默认）
    python main.py --algo both --no-show  # 不显示图形窗口
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

from common import (
    generate_simulation_data, clean_data, compute_distance_matrix,
    RANDOM_SEED, NUM_CLIENTS, COORD_RANGE, WEIGHT_RANGE,
    MAX_CAPACITY, MAX_DISTANCE, DEPOT_COORDS,
)


def print_comparison(greedy_result, genetic_result):
    """打印贪心 vs 遗传的指标对比表格及改进幅度"""
    print("\n" + "=" * 70)
    print("算法对比结果")
    print("=" * 70)
    print(f"{'指标':<20} {'贪心算法':>20} {'遗传算法':>20}")
    print("-" * 70)
    print(f"{'总飞行距离':<20} {greedy_result['total_distance']:>20.2f} {genetic_result['total_distance']:>20.2f}")
    print(f"{'总架次/趟次':<20} {greedy_result['total_trips']:>20d} {genetic_result['total_trips']:>20d}")
    print(f"{'总配送次数':<20} {greedy_result['total_deliveries']:>20d} {genetic_result['total_deliveries']:>20d}")

    # 计算遗传算法相对于贪心算法的改进
    improvement = (greedy_result['total_distance'] - genetic_result['total_distance'])
    improvement_pct = improvement / greedy_result['total_distance'] * 100
    print("-" * 70)
    print(f"遗传算法相比贪心算法，总距离减少: {improvement:.2f} 单位 ({improvement_pct:.1f}%)")
    print("=" * 70)


def plot_comparison(greedy_result, genetic_result, save_to_file=False, output_dir=None):
    """绘制算法对比图：左右并排展示贪心路径 vs 遗传路径，使用相同数据"""
    from matplotlib.patches import FancyArrowPatch

    clients = greedy_result['clients']
    depot = DEPOT_COORDS

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#32CD32', '#9932CC',
              '#FFD700', '#FF6347', '#4169E1']

    # 左侧画贪心算法，右侧画遗传算法
    for alg_name, result, ax in [("贪心算法", greedy_result, axes[0]),
                                  ("遗传算法", genetic_result, axes[1])]:
        routes = result['routes']
        ax.scatter(depot[0], depot[1], c='red', s=100, marker='s', label='配送中心', zorder=5)

        # 所有客户（灰色）及其标注
        for i, (x, y, w) in enumerate(clients):
            ax.scatter(x, y, s=60, c='gray', alpha=0.5, zorder=3)
            ax.text(x + 1.2, y + 1.2, f'{i}({int(w)}kg)', fontsize=7, alpha=0.7)

        # 逐趟绘制路径箭头
        for idx, route in enumerate(routes):
            color = colors[idx % len(colors)]
            path = [depot] + [clients[i, :2] for i in route['route']] + [depot]
            path = np.array(path)

            for k in range(len(path) - 1):
                ax.add_patch(FancyArrowPatch(
                    path[k], path[k + 1], arrowstyle='-|>', mutation_scale=12,
                    color=color, linewidth=1.8, alpha=0.8, zorder=4))

            # 该趟路线上的客户点着对应颜色
            client_coords = clients[route['route'], :2]
            ax.scatter(client_coords[:, 0], client_coords[:, 1],
                       c=color, s=60, zorder=5)

        total_dist = result['total_distance']
        n_trips = result['total_trips']
        ax.set_title(f"{alg_name}\n总距离: {total_dist:.1f}  |  总趟次: {n_trips}",
                     fontsize=13, fontweight='bold')

        ax.set_xlim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
        ax.set_ylim(COORD_RANGE[0] - 5, COORD_RANGE[1] + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X坐标', fontsize=10)
        ax.set_ylabel('Y坐标', fontsize=10)

    plt.suptitle('无人机配送路径规划 - 算法对比', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_to_file:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir,
                f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        else:
            filepath = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"对比图已保存: {filepath}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='无人机配送路径规划',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    python main.py --algo greedy       # 仅贪心算法
    python main.py --algo genetic      # 仅遗传算法
    python main.py --algo both         # 两者对比
    python main.py --no-show           # 不显示图形窗口
        ''')
    parser.add_argument('--algo', choices=['greedy', 'genetic', 'both'],
                        default='both', help='算法选择 (默认: both)')
    parser.add_argument('--num-drones', type=int, default=10,
                        help='遗传算法无人机数量 (默认: 10)')
    parser.add_argument('--no-show', action='store_true',
                        help='不显示图形窗口 (用于批处理)')
    parser.add_argument('--output-dir', default='../outputs',
                        help='输出目录 (默认: ../outputs)')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    # 非交互后端：使用 Agg 避免 plt.show() 弹窗阻塞（需在导入 pyplot 前设置）
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from drone_delivery import run_greedy
    from drone_delivery_genetic import run_genetic

    print("=" * 60)
    print("无人机配送路径规划")
    print("=" * 60)
    print(f"随机种子: {RANDOM_SEED}")
    print(f"客户数:   {NUM_CLIENTS}")
    print(f"载重限制: {MAX_CAPACITY} kg")
    print(f"里程限制: {MAX_DISTANCE} 单位")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 生成共享数据
    print("\n【数据准备】")
    clients = generate_simulation_data()
    clients = clean_data(clients)
    print()

    greedy_result = None
    genetic_result = None

    if args.algo in ('greedy', 'both'):
        print("=" * 60)
        print(">>> 运行贪心算法")
        print("=" * 60)
        greedy_result = run_greedy(clients=clients, output_dir=output_dir)

    if args.algo in ('genetic', 'both'):
        print("\n" + "=" * 60)
        print(">>> 运行遗传算法")
        print("=" * 60)
        genetic_result = run_genetic(
            num_drones=args.num_drones, clients=clients, output_dir=output_dir
        )

    if args.algo == 'both' and greedy_result and genetic_result:
        print_comparison(greedy_result, genetic_result)
        plot_comparison(greedy_result, genetic_result, save_to_file=True, output_dir=output_dir)

    print("\n完成！")


if __name__ == "__main__":
    main()
