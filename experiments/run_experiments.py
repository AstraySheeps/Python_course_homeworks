#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验脚本 — 多场景、多seed、统计显著性
文字报告保存为 Markdown 到 outputs/ 目录
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCENARIOS, SEED, NUM_INDEPENDENT_RUNS, ALGO_NAMES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'output')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def _build_experiment_markdown(scenario_name, algo_names, all_results,
                                summary, stats_test, n_runs, elapsed):
    """构建批量实验的 Markdown 报告"""
    cfg = SCENARIOS[scenario_name]
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []

    lines.append(f"# 批量实验报告")
    lines.append("")
    lines.append(f"- **场景**: {scenario_name}（{cfg['num_customers']} 客户 / {cfg['num_drones']} 架无人机）")
    lines.append(f"- **独立运行次数**: {n_runs}")
    lines.append(f"- **算法**: {', '.join(ALGO_NAMES[a] for a in algo_names)}")
    lines.append(f"- **总耗时**: {elapsed:.1f}s")
    lines.append(f"- **生成时间**: {now}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 汇总统计表
    lines.append("## 实验结果汇总（mean ± std）")
    lines.append("")

    metrics = [
        ('total_cost', '总成本(元)', '.2f'),
        ('flight_cost', '飞行成本(元)', '.2f'),
        ('operation_cost', '运营成本(元)', '.2f'),
        ('delay_penalty', '延迟罚金(元)', '.2f'),
        ('total_distance', '总距离(km)', '.2f'),
        ('total_flight_time', '总飞行时间(h)', '.4f'),
        ('total_delay_time', '总延迟(h)', '.4f'),
        ('makespan', 'Makespan(min)', '.2f'),
        ('load_gini', '负载基尼系数', '.4f'),
        ('num_active_drones', '活跃无人机数', '.1f'),
        ('runtime', '运行时间(s)', '.3f'),
    ]

    for key, label, fmt in metrics:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 算法 | Mean | Std | Min | Max |")
        lines.append("|------|------|-----|-----|-----|")
        for algo_name in algo_names:
            if algo_name not in summary:
                continue
            s = summary[algo_name].get(key, {})
            if isinstance(s, dict) and 'mean' in s:
                lines.append(f"| {ALGO_NAMES.get(algo_name, algo_name)} "
                             f"| {s['mean']:{fmt}} "
                             f"| {s['std']:{fmt}} "
                             f"| {s['min']:{fmt}} "
                             f"| {s['max']:{fmt}} |")
        lines.append("")

    # 可行率
    lines.append("### 可行率")
    lines.append("")
    lines.append("| 算法 | 可行率 |")
    lines.append("|------|--------|")
    for algo_name in algo_names:
        if algo_name not in summary:
            continue
        rate = summary[algo_name].get('feasible_rate', 0)
        lines.append(f"| {ALGO_NAMES.get(algo_name, algo_name)} | {rate:.1%} |")
    lines.append("")

    # 统计检验
    if stats_test:
        lines.append("---")
        lines.append("")
        lines.append("## 统计检验（配对 t 检验：SA vs GA）")
        lines.append("")
        lines.append(f"- t 统计量: {stats_test['t_statistic']:.4f}")
        lines.append(f"- p 值: {stats_test['p_value']:.4f}")
        lines.append(f"- 显著性: {'✅ p < 0.05，差异显著' if stats_test['significant'] else '❌ 差异不显著'}")
        lines.append("")

    return '\n'.join(lines)


def run_experiment(scenario_name='standard', algo_names=None,
                   n_runs=NUM_INDEPENDENT_RUNS, verbose=True):
    """运行指定场景的批量实验，结果保存为 JSON 和 Markdown"""
    if algo_names is None:
        algo_names = ['greedy', 'sa', 'ga', 'random']

    from data.generate_data import generate_scenario
    from src.utils.factories import build_problem
    from src.utils.statistics import summary_table, paired_ttest
    from src.algorithms import SOLVERS

    cfg = SCENARIOS[scenario_name]
    num_customers = cfg['num_customers']
    num_drones = cfg['num_drones']

    if verbose:
        print(f"\n批量实验: {scenario_name} ({num_customers}客户/{num_drones}架无人机)")
        print(f"算法: {[ALGO_NAMES[a] for a in algo_names]}, 运行 {n_runs} 次")

    t_start = time.time()
    all_results = {a: [] for a in algo_names}

    for run in range(n_runs):
        seed = SEED + run
        customers = generate_scenario(scenario_name, seed=seed)
        problem = build_problem(customers, num_drones)

        for algo_name in algo_names:
            t0 = time.time()
            solve_fn = SOLVERS[algo_name]
            routes, cost, history = solve_fn(problem, seed=seed)
            elapsed_algo = time.time() - t0
            eval_result = problem.evaluate_solution(routes)
            eval_result['algo'] = algo_name
            eval_result['run'] = run
            eval_result['seed'] = seed
            eval_result['runtime'] = elapsed_algo
            all_results[algo_name].append(eval_result)

        if verbose and (run + 1) % 10 == 0:
            print(f"  已完成 {run + 1}/{n_runs} 次运行")

    elapsed_total = time.time() - t_start

    # 汇总统计
    summary = summary_table(all_results)

    # 统计检验 (GA vs SA)
    stats_test = None
    if 'ga' in all_results and 'sa' in all_results:
        ga_costs = [r['total_cost'] for r in all_results['ga']]
        sa_costs = [r['total_cost'] for r in all_results['sa']]
        stats_test = paired_ttest(sa_costs, ga_costs)

    # 保存 JSON 结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        OUTPUT_DIR, f'experiment_{scenario_name}_{timestamp}.json'
    )

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
            'stats_test': stats_test,
        }, f, indent=2, ensure_ascii=False)

    # 生成并保存 Markdown 报告
    md_content = _build_experiment_markdown(
        scenario_name, algo_names, all_results, summary,
        stats_test, n_runs, elapsed_total
    )
    report_path = os.path.join(
        REPORT_DIR, f'report_experiment_{scenario_name}_{timestamp}.md'
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    if verbose:
        print(f"  JSON 结果: {result_file}")
        print(f"  Markdown 报告: {report_path}")
        print(f"  耗时: {elapsed_total:.1f}s")

    return all_results, summary, stats_test


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
    print("完成！")
