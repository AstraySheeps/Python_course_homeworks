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

import argparse
import os
import time
from datetime import datetime

from config import SCENARIOS, SEED, NUM_INDEPENDENT_RUNS, DEPOT_COORDS, ALGO_NAMES
from data.generate_data import generate_scenario
from src.utils.factories import build_problem
from src.algorithms import SOLVERS
from visualization.route_map import (
    plot_customer_distribution, plot_optimal_routes, plot_algorithm_comparison
)
from visualization.comparison import (
    plot_cost_comparison, plot_multi_metric_comparison, plot_runtime_vs_cost
)
from visualization.convergence import plot_convergence_curves
from visualization.load_balance import plot_load_distribution

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _build_run_report(problem, results, scenario_name, seed):
    """构建单次运行的 Markdown 报告"""
    cfg = SCENARIOS[scenario_name]
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []

    lines.append(f"# 无人机配送路径规划 — 运行报告")
    lines.append("")
    lines.append(f"- **场景**: {scenario_name}（{cfg['num_customers']} 客户 / {cfg['num_drones']} 架无人机）")
    lines.append(f"- **种子**: {seed}")
    lines.append(f"- **生成时间**: {now}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 算法对比汇总表
    lines.append("## 算法对比汇总")
    lines.append("")
    lines.append("| 算法 | 总成本(元) | 距离(km) | Makespan(min) | 延迟(h) | 超重(kg) | 超航程(km) | 基尼系数 | 耗时(s) | 可行 |")
    lines.append("|------|-----------|----------|---------------|---------|----------|------------|----------|---------|------|")
    algo_order = ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']
    for algo_name in algo_order:
        if algo_name not in results:
            continue
        r = results[algo_name]
        ok = '✅' if r['is_feasible'] else '❌'
        lines.append(f"| {ALGO_NAMES.get(algo_name, algo_name)} "
                     f"| {r['total_cost']:>10.2f} "
                     f"| {r['total_distance']:>8.2f} "
                     f"| {r['makespan']:>12.1f} "
                     f"| {r['total_delay_time']:>8.4f} "
                     f"| {r['total_overload']:>8.2f} "
                     f"| {r['total_excess_range']:>10.2f} "
                     f"| {r['load_gini']:>8.3f} "
                     f"| {r['runtime']:>7.2f} "
                     f"| {ok} |")
    lines.append("")

    # 成本分解
    lines.append("## 成本分解（元）")
    lines.append("")
    lines.append("| 算法 | 飞行成本 | 运营成本 | 延迟罚金 | 超重罚金 | 超航程罚金 | 总成本 |")
    lines.append("|------|----------|----------|----------|----------|------------|--------|")
    for algo_name in algo_order:
        if algo_name not in results:
            continue
        r = results[algo_name]
        lines.append(f"| {ALGO_NAMES.get(algo_name, algo_name)} "
                     f"| {r['flight_cost']:>8.2f} "
                     f"| {r['operation_cost']:>8.2f} "
                     f"| {r['delay_penalty']:>8.2f} "
                     f"| {r['overload_penalty']:>8.2f} "
                     f"| {r['excess_range_penalty']:>10.2f} "
                     f"| {r['total_cost']:>8.2f} |")
    lines.append("")

    # 相对贪心基线的改进
    baseline_cost = results.get('greedy', {}).get('total_cost', 0)
    if baseline_cost > 0:
        lines.append("## 相对贪心基线的改进")
        lines.append("")
        lines.append("| 算法 | 改进(元) | 改进(%) |")
        lines.append("|------|---------|---------|")
        for algo_name in ['greedy_urgent', 'sa', 'ga', 'random']:
            if algo_name not in results:
                continue
            imp = baseline_cost - results[algo_name]['total_cost']
            imp_pct = imp / baseline_cost * 100
            arrow = '↓' if imp > 0 else '↑'
            lines.append(f"| {ALGO_NAMES.get(algo_name, algo_name)} "
                         f"| {arrow} {abs(imp):.2f} "
                         f"| {arrow} {abs(imp_pct):.1f}% |")
        lines.append("")

    # 约束违反分析
    lines.append("---")
    lines.append("")
    lines.append("## 约束违反分析")
    lines.append("")
    for algo_name in algo_order:
        if algo_name not in results:
            continue
        md_text, _ = problem.violation_report_markdown(
            results[algo_name]['routes'],
            ALGO_NAMES.get(algo_name, algo_name)
        )
        lines.append(md_text)
        lines.append("")

    return '\n'.join(lines)


def run_single(scenario_name='standard', algo_names=None, seed=SEED):
    """单次运行所有算法，返回 problem 和 results（不打印到终端）"""
    if algo_names is None:
        algo_names = ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']

    cfg = SCENARIOS[scenario_name]
    customers_dict = generate_scenario(scenario_name, seed=seed)
    problem = build_problem(customers_dict, cfg['num_drones'])

    results = {}
    for algo_name in algo_names:
        t0 = time.time()
        routes, cost, history = SOLVERS[algo_name](problem, seed=seed)
        elapsed = time.time() - t0

        eval_result = problem.evaluate_solution(routes)
        eval_result['routes'] = routes
        eval_result['history'] = history
        eval_result['runtime'] = elapsed
        results[algo_name] = eval_result

    return problem, results


def run_and_visualize(scenario_name='standard', algo_names=None, seed=SEED):
    """运行算法，生成全部可视化图表和 Markdown 报告"""
    print(f"运行中... 场景: {scenario_name}")

    problem, results = run_single(scenario_name, algo_names, seed)

    # 收集路线
    routes_dict = {}
    for algo_name, r in results.items():
        routes_dict[algo_name] = r['routes']

    # 生成全部图表
    print(f"  生成可视化图表...")
    plot_customer_distribution(problem.customers)

    if 'ga' in results:
        plot_optimal_routes(problem, results['ga']['routes'],
                            title='遗传算法最优配送路线')

    compare_routes = {}
    for k in ['greedy', 'sa', 'ga']:
        if k in routes_dict:
            compare_routes[k] = routes_dict[k]
    if len(compare_routes) >= 2:
        plot_algorithm_comparison(problem, compare_routes)

    if len(routes_dict) >= 2:
        plot_load_distribution(problem, routes_dict, ALGO_NAMES)

    sa_history = results.get('sa', {}).get('history')
    ga_history = results.get('ga', {}).get('history')
    if sa_history or ga_history:
        plot_convergence_curves(sa_history=sa_history, ga_history=ga_history)

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

    # 生成 Markdown 报告
    print(f"  生成文字报告...")
    md_content = _build_run_report(problem, results, scenario_name, seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(OUTPUT_DIR, f'report_{scenario_name}_{timestamp}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"  图表已保存到 outputs/")
    print(f"  报告已保存到 {report_path}")
    print(f"  Done.")

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

    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("无人机集群配送路径规划")
    print("=" * 60)
    print(f"场景: {args.scenario} ({SCENARIOS[args.scenario]['num_customers']}客户/"
          f"{SCENARIOS[args.scenario]['num_drones']}架无人机)")
    print(f"算法: {[ALGO_NAMES[a] for a in args.algo]}")
    print(f"种子: {args.seed}")

    if args.runs > 1:
        # 批量实验模式
        from experiments.run_experiments import run_experiment
        run_experiment(args.scenario, args.algo, args.runs)
    else:
        # 单次运行 + 可视化
        run_and_visualize(args.scenario, args.algo, args.seed)

    print("=" * 60)
    print("全部完成！报告和图表保存在 outputs/ 目录")
    print("=" * 60)


if __name__ == '__main__':
    main()
