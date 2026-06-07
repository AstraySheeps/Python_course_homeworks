#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - 统一入口
支持贪心算法、遗传算法、模拟退火算法、以及对比
用法:
    python main.py --algo greedy          # 仅贪心算法
    python main.py --algo genetic         # 仅遗传算法
    python main.py --algo sa              # 仅模拟退火算法
    python main.py --algo both            # 贪心 vs 遗传（默认）
    python main.py --algo all             # 三种算法对比
    python main.py --algo both --no-show  # 不显示图形窗口
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

from common import (
    generate_simulation_data, clean_data, compute_distance_matrix,
    generate_time_windows,
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
    import matplotlib.pyplot as plt
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


def print_all_comparison(greedy_result, genetic_result, sa_result):
    """打印贪心 vs 遗传 vs 模拟退火的指标对比表格"""
    results = [
        ("贪心算法", greedy_result),
        ("遗传算法", genetic_result),
        ("模拟退火", sa_result),
    ]

    print("\n" + "=" * 80)
    print("三种算法对比结果")
    print("=" * 80)
    header = f"{'指标':<20}"
    for name, _ in results:
        header += f" {name:>18}"
    print(header)
    print("-" * 80)

    print(f"{'总飞行距离':<20}", end="")
    for _, r in results:
        print(f" {r['total_distance']:>18.2f}", end="")
    print()

    print(f"{'总趟次':<20}", end="")
    for _, r in results:
        print(f" {r['total_trips']:>18d}", end="")
    print()

    print(f"{'总配送次数':<20}", end="")
    for _, r in results:
        print(f" {r['total_deliveries']:>18d}", end="")
    print()

    print("-" * 80)

    baseline = greedy_result['total_distance']
    for name, r in results:
        imp = baseline - r['total_distance']
        imp_pct = imp / baseline * 100
        print(f"{name} 相比贪心基线，总距离减少: {imp:+.2f} 单位 ({imp_pct:+.1f}%)")
    print("=" * 80)


def plot_all_comparison(greedy_result, genetic_result, sa_result,
                        save_to_file=False, output_dir=None):
    """绘制三种算法对比图：三列并排展示"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    clients = greedy_result['clients']
    depot = DEPOT_COORDS

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#32CD32', '#9932CC',
              '#FFD700', '#FF6347', '#4169E1']

    configs = [
        ("贪心算法", greedy_result, axes[0]),
        ("遗传算法", genetic_result, axes[1]),
        ("模拟退火", sa_result, axes[2]),
    ]

    for alg_name, result, ax in configs:
        routes = result['routes']
        ax.scatter(depot[0], depot[1], c='red', s=100, marker='s', label='配送中心', zorder=5)

        for i, (x, y, w) in enumerate(clients):
            ax.scatter(x, y, s=60, c='gray', alpha=0.5, zorder=3)
            ax.text(x + 1.2, y + 1.2, f'{i}({int(w)}kg)', fontsize=7, alpha=0.7)

        for idx, route in enumerate(routes):
            color = colors[idx % len(colors)]
            path = [depot] + [clients[i, :2] for i in route['route']] + [depot]
            path = np.array(path)

            for k in range(len(path) - 1):
                ax.add_patch(FancyArrowPatch(
                    path[k], path[k + 1], arrowstyle='-|>', mutation_scale=12,
                    color=color, linewidth=1.8, alpha=0.8, zorder=4))

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

    plt.suptitle('无人机配送路径规划 - 三种算法对比', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_to_file:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir,
                f"comparison_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        else:
            filepath = f"comparison_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"三方对比图已保存: {filepath}")

    plt.show()


def print_multi_seed_summary(all_results, algo_names, seeds, output_dir):
    """多种子评估汇总：打印均值±标准差，并保存到文件"""
    from datetime import datetime

    print(f"\n{'='*80}")
    print("多随机种子评估汇总")
    print(f"{'='*80}")
    print(f"种子列表: {seeds}")
    print(f"样本数:   {len(seeds)}")
    print()

    active_algos = [k for k in ['greedy', 'genetic', 'sa'] if all_results.get(k)]

    # 表头
    header = f"{'指标':<20}"
    for key in active_algos:
        header += f" {algo_names[key]:>22}"
    print(header)
    print("-" * 80)

    # 总飞行距离
    print(f"{'总飞行距离':<20}", end="")
    for key in active_algos:
        dists = [r['total_distance'] for r in all_results[key]]
        mean_d = np.mean(dists)
        std_d = np.std(dists, ddof=1) if len(dists) > 1 else 0
        print(f" {mean_d:>15.2f}±{std_d:>5.2f}", end="")
    print()

    # 总趟次
    print(f"{'总趟次':<20}", end="")
    for key in active_algos:
        trips = [r['total_trips'] for r in all_results[key]]
        mean_t = np.mean(trips)
        std_t = np.std(trips, ddof=1) if len(trips) > 1 else 0
        print(f" {mean_t:>15.1f}±{std_t:>5.1f}", end="")
    print()

    # 总配送次数
    print(f"{'总配送次数':<20}", end="")
    for key in active_algos:
        deliveries = [r['total_deliveries'] for r in all_results[key]]
        mean_d = np.mean(deliveries)
        std_d = np.std(deliveries, ddof=1) if len(deliveries) > 1 else 0
        print(f" {mean_d:>15.1f}±{std_d:>5.1f}", end="")
    print()

    print("-" * 80)

    # 相比贪心基线的改进
    if all_results.get('greedy'):
        greedy_mean = np.mean([r['total_distance'] for r in all_results['greedy']])
        for key in ['genetic', 'sa']:
            if all_results.get(key):
                algo_mean = np.mean([r['total_distance'] for r in all_results[key]])
                imp = greedy_mean - algo_mean
                imp_pct = imp / greedy_mean * 100
                print(f"{algo_names[key]} 相比贪心基线（均值），总距离减少: "
                      f"{imp:+.2f} 单位 ({imp_pct:+.1f}%)")

    # 各次详细结果
    print(f"\n各次详细结果:")
    for key in active_algos:
        dists = [r['total_distance'] for r in all_results[key]]
        detail = ", ".join(f"{d:.2f}" for d in dists)
        print(f"  {algo_names[key]}: [{detail}]")

    print("=" * 80)

    # 保存汇总到文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir,
            f"multi_seed_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("多随机种子评估汇总\n")
            f.write(f"种子: {seeds}\n")
            f.write(f"样本数: {len(seeds)}\n\n")
            for key in active_algos:
                dists = [r['total_distance'] for r in all_results[key]]
                trips = [r['total_trips'] for r in all_results[key]]
                f.write(f"{algo_names[key]}:\n")
                f.write(f"  总距离: {np.mean(dists):.2f} ± {np.std(dists, ddof=1):.2f}\n")
                f.write(f"  各次: {[f'{d:.2f}' for d in dists]}\n")
                f.write(f"  趟次: {np.mean(trips):.1f} ± {np.std(trips, ddof=1):.1f}\n\n")
            if all_results.get('greedy'):
                greedy_mean = np.mean([r['total_distance'] for r in all_results['greedy']])
                for key in ['genetic', 'sa']:
                    if all_results.get(key):
                        algo_mean = np.mean([r['total_distance'] for r in all_results[key]])
                        imp_pct = (greedy_mean - algo_mean) / greedy_mean * 100
                        f.write(f"{algo_names[key]} 相比贪心提升: {imp_pct:.1f}%\n")
        print(f"\n汇总已保存: {filepath}")


def run_multi_seed_evaluation(args, output_dir):
    """多随机种子评估：在多个数据样本上运行算法，统计均值±标准差"""
    import matplotlib
    matplotlib.use('Agg')

    from drone_delivery import run_greedy, greedy_assignment
    from drone_delivery_genetic import run_genetic, run_genetic_algorithm
    from drone_delivery_sa import run_sa, simulated_annealing

    seeds = list(range(RANDOM_SEED, RANDOM_SEED + args.seeds))

    all_results = {'greedy': [], 'genetic': [], 'sa': []}
    algo_names = {'greedy': '贪心算法', 'genetic': '遗传算法', 'sa': '模拟退火'}

    print(f"\n{'='*70}")
    print(f"多随机种子评估模式：{args.seeds} 个种子 ({seeds[0]} ~ {seeds[-1]})")
    print(f"{'='*70}")

    for idx, seed in enumerate(seeds):
        print(f"\n--- 种子 {seed} ({idx+1}/{args.seeds}) ---")

        clients = generate_simulation_data(seed=seed)
        clients = clean_data(clients)
        dist_matrix = compute_distance_matrix(clients)
        is_last = (idx == args.seeds - 1)

        # 时间窗约束
        time_windows = None
        if args.tw:
            time_windows = generate_time_windows(clients, dist_matrix, seed=seed)

        if args.algo in ('greedy', 'both', 'all'):
            routes = greedy_assignment(clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE,
                                        time_windows=time_windows)
            total_dist = sum(r['distance'] for r in routes)
            all_results['greedy'].append({
                'clients': clients, 'routes': routes, 'dist_matrix': dist_matrix,
                'total_distance': total_dist, 'total_trips': len(routes),
                'total_deliveries': sum(r['deliveries'] for r in routes),
            })
            print(f"  贪心: {total_dist:.2f} ({len(routes)} 趟)")

        if args.algo in ('genetic', 'both', 'all'):
            best_routes, logbook = run_genetic_algorithm(
                clients, dist_matrix, args.num_drones, time_windows=time_windows
            )
            total_dist = sum(r['distance'] for r in best_routes)
            all_results['genetic'].append({
                'clients': clients, 'routes': best_routes, 'dist_matrix': dist_matrix,
                'total_distance': total_dist, 'total_trips': len(best_routes),
                'total_deliveries': sum(r['deliveries'] for r in best_routes),
                'logbook': logbook,
            })
            print(f"  遗传: {total_dist:.2f} ({len(best_routes)} 趟)")

        if args.algo in ('sa', 'all'):
            _, best_routes, best_cost, _ = simulated_annealing(
                clients, dist_matrix, args.num_drones, time_windows=time_windows
            )
            all_results['sa'].append({
                'clients': clients, 'routes': best_routes, 'dist_matrix': dist_matrix,
                'total_distance': best_cost, 'total_trips': len(best_routes),
                'total_deliveries': sum(r['deliveries'] for r in best_routes),
            })
            print(f"  退火: {best_cost:.2f} ({len(best_routes)} 趟)")

        # 最后一个种子：生成完整可视化输出（图表+文本文件）
        if is_last:
            print(f"\n[种子 {seed}] 生成可视化输出...")
            if args.algo in ('greedy', 'both', 'all'):
                run_greedy(clients=clients, output_dir=output_dir,
                           time_windows=time_windows)
            if args.algo in ('genetic', 'both', 'all'):
                run_genetic(num_drones=args.num_drones, clients=clients,
                            output_dir=output_dir, time_windows=time_windows)
            if args.algo in ('sa', 'all'):
                run_sa(num_drones=args.num_drones, clients=clients,
                       output_dir=output_dir, time_windows=time_windows)

    # 打印汇总统计
    print_multi_seed_summary(all_results, algo_names, seeds, output_dir)

    # 生成对比图（所有种子的数据来自最后一个种子）
    if args.algo in ('both', 'all'):
        last_results = {}
        if all_results['greedy']:
            last_results['greedy'] = all_results['greedy'][-1]
        if all_results['genetic']:
            last_results['genetic'] = all_results['genetic'][-1]
        if all_results['sa']:
            last_results['sa'] = all_results['sa'][-1]

        if len(last_results) == 3:
            plot_all_comparison(
                last_results['greedy'], last_results['genetic'],
                last_results['sa'], save_to_file=True, output_dir=output_dir
            )
        elif 'greedy' in last_results and 'genetic' in last_results:
            plot_comparison(
                last_results['greedy'], last_results['genetic'],
                save_to_file=True, output_dir=output_dir
            )


def main():
    parser = argparse.ArgumentParser(
        description='无人机配送路径规划',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    python main.py --algo greedy       # 仅贪心算法
    python main.py --algo genetic      # 仅遗传算法
    python main.py --algo both         # 两者对比
    python main.py --algo all --seeds 5  # 5个随机种子评估
    python main.py --no-show           # 不显示图形窗口
        ''')
    parser.add_argument('--algo', choices=['greedy', 'genetic', 'sa', 'both', 'all'],
                        default='both', help='算法选择 (默认: both)')
    parser.add_argument('--num-drones', type=int, default=10,
                        help='无人机数量 (默认: 10)')
    parser.add_argument('--seeds', type=int, default=1,
                        help='多种子评估的种子数量 (默认: 1，即单次运行)')
    parser.add_argument('--no-show', action='store_true',
                        help='不显示图形窗口 (用于批处理)')
    parser.add_argument('--output-dir', default='../outputs',
                        help='输出目录 (默认: ../outputs)')
    parser.add_argument('--tw', action='store_true',
                        help='启用时间窗约束（每个客户有指定配送时间窗口）')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    print("=" * 60)
    print("无人机配送路径规划")
    print("=" * 60)
    print(f"随机种子基准: {RANDOM_SEED}")
    print(f"客户数:       {NUM_CLIENTS}")
    print(f"载重限制:     {MAX_CAPACITY} kg")
    print(f"里程限制:     {MAX_DISTANCE} 单位")
    print(f"无人机数量:   {args.num_drones}")
    print(f"输出目录:     {output_dir}")
    print("=" * 60)

    # 多种子评估模式
    if args.seeds > 1:
        run_multi_seed_evaluation(args, output_dir)
        print("\n完成！")
        return

    # 非交互后端：使用 Agg 避免 plt.show() 弹窗阻塞（需在导入 pyplot 前设置）
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from drone_delivery import run_greedy
    from drone_delivery_genetic import run_genetic
    from drone_delivery_sa import run_sa

    # 生成共享数据
    print("\n【数据准备】")
    clients = generate_simulation_data()
    clients = clean_data(clients)

    # 时间窗约束（可选）
    time_windows = None
    if args.tw:
        dist_matrix = compute_distance_matrix(clients)
        time_windows = generate_time_windows(clients, dist_matrix)
        tw_info = f"就绪最早={time_windows[0].min():.1f}~{time_windows[0].max():.1f}, 截止最晚={time_windows[1].min():.1f}~{time_windows[1].max():.1f}"
        print(f"时间窗约束已启用: {tw_info}")
    print()

    greedy_result = None
    genetic_result = None
    sa_result = None

    if args.algo in ('greedy', 'both', 'all'):
        print("=" * 60)
        print(">>> 运行贪心算法")
        print("=" * 60)
        greedy_result = run_greedy(clients=clients, output_dir=output_dir,
                                   time_windows=time_windows)

    if args.algo in ('genetic', 'both', 'all'):
        print("\n" + "=" * 60)
        print(">>> 运行遗传算法")
        print("=" * 60)
        genetic_result = run_genetic(
            num_drones=args.num_drones, clients=clients, output_dir=output_dir,
            time_windows=time_windows,
        )

    if args.algo in ('sa', 'all'):
        print("\n" + "=" * 60)
        print(">>> 运行模拟退火算法")
        print("=" * 60)
        sa_result = run_sa(
            num_drones=args.num_drones, clients=clients, output_dir=output_dir,
            time_windows=time_windows,
        )

    if args.algo in ('both', 'all') and greedy_result and genetic_result and sa_result:
        print_all_comparison(greedy_result, genetic_result, sa_result)
        plot_all_comparison(greedy_result, genetic_result, sa_result,
                            save_to_file=True, output_dir=output_dir)
    elif args.algo == 'both' and greedy_result and genetic_result:
        print_comparison(greedy_result, genetic_result)
        plot_comparison(greedy_result, genetic_result, save_to_file=True, output_dir=output_dir)

    print("\n完成！")


if __name__ == "__main__":
    main()
