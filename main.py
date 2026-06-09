#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机集群配送路径规划 — 一键运行主脚本
用法:
    python main.py                          # 标准场景，4种算法对比
    python main.py --scenario small         # 小规模验证
    python main.py --scenario large         # 大规模压力测试
    python main.py --runs 30                # 30次独立运行
    python main.py --visualize-only         # 仅生成可视化图表
"""

import os
import sys
import argparse
import time
import numpy as np

from config import SCENARIOS, SEED, NUM_INDEPENDENT_RUNS, DEPOT_COORDS
from data.generate_data import generate_scenario
from src.models.customer import Customer
from src.models.drone import Drone
from src.models.problem import Problem
from src.utils.distance import compute_distance_matrix
from src.algorithms.greedy import solve_greedy, solve_greedy_urgent
from src.algorithms.sa import solve_sa
from src.algorithms.ga import solve_ga
from src.algorithms.random_search import solve_random_search
from visualization.route_map import (
    plot_customer_distribution, plot_optimal_routes, plot_algorithm_comparison
)
from visualization.comparison import (
    plot_cost_comparison, plot_multi_metric_comparison, plot_runtime_vs_cost
)
from visualization.convergence import plot_convergence_curves
from visualization.load_balance import plot_load_distribution


ALGO_NAMES = {
    'greedy': '贪心算法',
    'greedy_urgent': '贪心(紧急优先)',
    'sa': '模拟退火',
    'ga': '遗传算法',
    'random': '随机搜索',
}


def build_problem(customers_dict, num_drones):
    from config import DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE

    customers = [
        Customer(
            id=c['id'], x=c['x'], y=c['y'], demand=c['demand'],
            customer_type=c['customer_type'],
            time_window=(c['time_window_start'], c['time_window_end']),
            service_time=c['service_time'],
        )
        for c in customers_dict
    ]
    drones = [
        Drone(i, DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE)
        for i in range(num_drones)
    ]
    dist_matrix = compute_distance_matrix(customers, DEPOT_COORDS)
    return Problem(customers, drones, dist_matrix)


def run_single(scenario_name='standard', algo_names=None, seed=SEED, verbose=True):
    """单次运行所有算法"""
    if algo_names is None:
        algo_names = ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']

    cfg = SCENARIOS[scenario_name]
    customers_dict = generate_scenario(scenario_name, seed=seed)
    problem = build_problem(customers_dict, cfg['num_drones'])

    solve_fns = {
        'greedy': solve_greedy,
        'greedy_urgent': solve_greedy_urgent,
        'sa': solve_sa,
        'ga': solve_ga,
        'random': solve_random_search,
    }

    results = {}
    for algo_name in algo_names:
        t0 = time.time()
        routes, cost, history = solve_fns[algo_name](problem, seed=seed)
        elapsed = time.time() - t0

        eval_result = problem.evaluate_solution(routes)
        eval_result['routes'] = routes
        eval_result['history'] = history
        eval_result['runtime'] = elapsed
        results[algo_name] = eval_result

        if verbose:
            print(f"  {ALGO_NAMES[algo_name]:<16} "
                  f"成本={eval_result['total_cost']:>10.2f}元  "
                  f"距离={eval_result['total_distance']:>8.2f}km  "
                  f"Makespan={eval_result['makespan']:>8.1f}min  "
                  f"耗时={elapsed:>6.2f}s  "
                  f"可行={'✓' if eval_result['is_feasible'] else '✗'}")

    return problem, results


def run_and_visualize(scenario_name='standard', algo_names=None, seed=SEED):
    """运行算法并生成全部可视化"""
    print(f"\n{'='*60}")
    print(f"场景: {scenario_name}")
    print(f"{'='*60}")

    problem, results = run_single(scenario_name, algo_names, seed)

    # 收集路线
    routes_dict = {}
    for algo_name, r in results.items():
        routes_dict[algo_name] = r['routes']

    # 生成全部图表
    print(f"\n生成可视化图表...")
    plot_customer_distribution(problem.customers)

    # 使用 GA 最优解画路线图
    if 'ga' in results:
        plot_optimal_routes(problem, results['ga']['routes'],
                            title='遗传算法最优配送路线')

    # 算法路线对比
    if len(routes_dict) >= 3:
        # 取 greedy, sa, ga
        compare_routes = {}
        for k in ['greedy', 'sa', 'ga']:
            if k in routes_dict:
                compare_routes[k] = routes_dict[k]
        if len(compare_routes) >= 2:
            plot_algorithm_comparison(problem, compare_routes)

    # 负载分布
    if len(routes_dict) >= 2:
        plot_load_distribution(problem, routes_dict, ALGO_NAMES)

    # 收敛曲线
    sa_history = results.get('sa', {}).get('history')
    ga_history = results.get('ga', {}).get('history')
    if sa_history or ga_history:
        plot_convergence_curves(sa_history=sa_history, ga_history=ga_history)

    # 成本对比柱状图和散点图
    summary = {}
    for algo_name, r in results.items():
        summary[algo_name] = {
            'total_cost': {'mean': r['total_cost'], 'std': 0},
            'total_distance': {'mean': r['total_distance'], 'std': 0},
            'makespan': {'mean': r['makespan'], 'std': 0},
        }
    plot_cost_comparison(summary, ALGO_NAMES)
    plot_multi_metric_comparison(summary, ALGO_NAMES)

    all_results_for_scatter = {a: [r] for a, r in results.items()}
    plot_runtime_vs_cost(all_results_for_scatter, ALGO_NAMES)

    print(f"\n全部图表已生成到 visualization/output/")

    # 打印对比汇总表
    print(f"\n{'='*80}")
    print("算法对比汇总")
    print(f"{'='*80}")
    header = f"{'算法':<16} {'总成本(元)':>12} {'距离(km)':>10} {'Makespan(min)':>14} {'延迟(h)':>10} {'可行':>6}"
    print(header)
    print("-" * 80)
    for algo_name in ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']:
        if algo_name in results:
            r = results[algo_name]
            print(f"{ALGO_NAMES[algo_name]:<16} {r['total_cost']:>12.2f} "
                  f"{r['total_distance']:>10.2f} {r['makespan']:>14.1f} "
                  f"{r['total_delay_time']:>10.4f} "
                  f"{'✓' if r['is_feasible'] else '✗':>6}")

    baseline_cost = results.get('greedy', {}).get('total_cost', 0)
    if baseline_cost > 0:
        print("-" * 80)
        for algo_name in ['greedy_urgent', 'sa', 'ga', 'random']:
            if algo_name in results:
                imp = baseline_cost - results[algo_name]['total_cost']
                imp_pct = imp / baseline_cost * 100
                print(f"{ALGO_NAMES[algo_name]} vs 贪心: {imp:+.2f}元 ({imp_pct:+.1f}%)")

    return problem, results


def main():
    parser = argparse.ArgumentParser(description='无人机集群配送路径规划')
    parser.add_argument('--scenario', default='standard',
                        choices=['small', 'standard', 'large'])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--runs', type=int, default=1,
                        help=f'独立运行次数 (默认1次, 批量实验建议{NUM_INDEPENDENT_RUNS}次)')
    parser.add_argument('--algo', nargs='+',
                        default=['greedy', 'sa', 'ga', 'random'])
    parser.add_argument('--output-dir', default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("无人机集群配送路径规划")
    print("=" * 60)
    print(f"场景: {args.scenario} ({SCENARIOS[args.scenario]['num_customers']}客户/"
          f"{SCENARIOS[args.scenario]['num_drones']}架无人机)")
    print(f"算法: {[ALGO_NAMES[a] for a in args.algo]}")
    print(f"种子: {args.seed}")
    print("=" * 60)

    if args.runs > 1:
        # 批量实验模式
        from experiments.run_experiments import run_experiment
        run_experiment(args.scenario, args.algo, args.runs)
    else:
        # 单次运行 + 可视化
        run_and_visualize(args.scenario, args.algo, args.seed)

    print("\n完成！")


if __name__ == '__main__':
    main()
