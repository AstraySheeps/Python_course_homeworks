#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验脚本 — 多场景、多seed、统计显著性
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCENARIOS, SEED, NUM_INDEPENDENT_RUNS, ALGO_NAMES
from data.generate_data import generate_scenario
from src.utils.factories import build_problem
from src.utils.statistics import summary_table, paired_ttest
from src.algorithms import SOLVERS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'output')


def run_experiment(scenario_name='standard', algo_names=None,
                   n_runs=NUM_INDEPENDENT_RUNS, verbose=True):
    """运行指定场景的批量实验"""
    if algo_names is None:
        algo_names = ['greedy', 'sa', 'ga', 'random']

    cfg = SCENARIOS[scenario_name]
    num_customers = cfg['num_customers']
    num_drones = cfg['num_drones']

    if verbose:
        print(f"\n{'='*60}")
        print(f"实验: {scenario_name} ({num_customers}客户/{num_drones}架无人机)")
        print(f"算法: {[ALGO_NAMES[a] for a in algo_names]}")
        print(f"独立运行次数: {n_runs}")
        print(f"{'='*60}")

    all_results = {a: [] for a in algo_names}

    for run in range(n_runs):
        seed = SEED + run
        customers = generate_scenario(scenario_name, seed=seed)
        problem = build_problem(customers, num_drones)

        for algo_name in algo_names:
            t0 = time.time()
            solve_fn = SOLVERS[algo_name]
            routes, cost, history = solve_fn(problem, seed=seed)
            elapsed = time.time() - t0
            eval_result = problem.evaluate_solution(routes)
            eval_result['algo'] = algo_name
            eval_result['run'] = run
            eval_result['seed'] = seed
            eval_result['runtime'] = elapsed
            all_results[algo_name].append(eval_result)

        if verbose and (run + 1) % 10 == 0:
            print(f"  已完成 {run + 1}/{n_runs} 次运行")

    # 汇总统计
    summary = summary_table(all_results)

    # 统计检验 (GA vs SA)
    stats_test = None
    if 'ga' in all_results and 'sa' in all_results:
        ga_costs = [r['total_cost'] for r in all_results['ga']]
        sa_costs = [r['total_cost'] for r in all_results['sa']]
        stats_test = paired_ttest(sa_costs, ga_costs)

    if verbose:
        _print_summary(summary, stats_test, algo_names)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        OUTPUT_DIR, f'experiment_{scenario_name}_{timestamp}.json'
    )

    # 简化结果用于 JSON 序列化
    serializable = {}
    for algo_name in algo_names:
        serializable[algo_name] = []
        for r in all_results[algo_name]:
            serializable[algo_name].append({
                'run': r['run'], 'seed': r['seed'],
                'total_cost': r['total_cost'],
                'total_distance': r['total_distance'],
                'makespan': r['makespan'],
                'makespan_hours': r['makespan_hours'],
                'total_delay_time': r['total_delay_time'],
                'total_overload': r['total_overload'],
                'total_excess_range': r['total_excess_range'],
                'is_feasible': r['is_feasible'],
                'load_gini': r['load_gini'],
                'num_active_drones': r['num_active_drones'],
                'runtime': r['runtime'],
            })

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'scenario': scenario_name,
            'n_runs': n_runs,
            'algorithms': algo_names,
            'results': serializable,
            'summary': {a: {
                k: {'mean': float(v['mean']), 'std': float(v['std']),
                    'min': float(v['min']), 'max': float(v['max'])}
                if isinstance(v, dict) and 'mean' in v else v
                for k, v in s.items()
            } for a, s in summary.items()},
        }, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n结果已保存: {result_file}")

    return all_results, summary, stats_test


def _print_summary(summary, stats_test, algo_names):
    """打印汇总统计"""
    print(f"\n{'='*80}")
    print("实验结果汇总 (mean ± std)")
    print(f"{'='*80}")

    metrics = [
        ('total_cost', '总成本(元)', '.2f'),
        ('total_distance', '总距离(km)', '.2f'),
        ('makespan', 'Makespan(min)', '.2f'),
        ('total_delay_time', '总延迟(h)', '.4f'),
        ('load_gini', '负载基尼系数', '.4f'),
    ]

    for key, label, fmt in metrics:
        print(f"\n{label}:")
        for algo_name in algo_names:
            if algo_name not in summary:
                continue
            s = summary[algo_name].get(key, {})
            if isinstance(s, dict) and 'mean' in s:
                print(f"  {ALGO_NAMES[algo_name]:<16} {s['mean']:{fmt}} ± {s['std']:{fmt}}")

    # 可行率
    print(f"\n可行率:")
    for algo_name in algo_names:
        if algo_name not in summary:
            continue
        rate = summary[algo_name].get('feasible_rate', 0)
        print(f"  {ALGO_NAMES[algo_name]:<16} {rate:.1%}")

    if stats_test:
        print(f"\n统计检验 (SA vs GA):")
        print(f"  t = {stats_test['t_statistic']:.4f}, p = {stats_test['p_value']:.4f}")
        print(f"  显著性: {'✓ p < 0.05' if stats_test['significant'] else '✗ 不显著'}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='批量实验')
    parser.add_argument('--scenario', default='standard',
                        choices=['small', 'standard', 'large'])
    parser.add_argument('--runs', type=int, default=NUM_INDEPENDENT_RUNS)
    parser.add_argument('--algo', nargs='+',
                        default=['greedy', 'sa', 'ga', 'random'])
    args = parser.parse_args()

    run_experiment(args.scenario, args.algo, args.runs)
    print("\n完成！")
