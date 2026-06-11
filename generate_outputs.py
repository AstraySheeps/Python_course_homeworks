#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键生成全部标准输出图表和 Markdown 报告（用于PPT和汇报）"""
import matplotlib
matplotlib.use('Agg')
import os
import time
from datetime import datetime

from config import SCENARIOS, SEED, ALGO_NAMES
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

ALGO_ORDER = ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _build_output_report(problem, results, scenario_name):
    """构建 Markdown 报告"""
    cfg = SCENARIOS[scenario_name]
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []

    lines.append(f"# 标准输出报告（PPT用）")
    lines.append("")
    lines.append(f"- **场景**: {scenario_name}（{cfg['num_customers']} 客户 / {cfg['num_drones']} 架无人机）")
    lines.append(f"- **种子**: {SEED}")
    lines.append(f"- **生成时间**: {now}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 算法对比汇总")
    lines.append("")
    lines.append("| 算法 | 总成本(元) | 距离(km) | Makespan(min) | 延迟(h) | 基尼系数 | 可行 |")
    lines.append("|------|-----------|----------|---------------|---------|----------|------|")
    for a in ALGO_ORDER:
        if a not in results:
            continue
        r = results[a]
        ok = '✅' if r['is_feasible'] else '❌'
        lines.append(f"| {ALGO_NAMES[a]} "
                     f"| {r['total_cost']:.2f} "
                     f"| {r['total_distance']:.2f} "
                     f"| {r['makespan']:.1f} "
                     f"| {r['total_delay_time']:.4f} "
                     f"| {r['load_gini']:.3f} "
                     f"| {ok} |")
    lines.append("")

    # 相对贪心基线的改进
    baseline = results['greedy']['total_cost']
    if baseline > 0:
        lines.append("## 相对贪心基线的改进")
        lines.append("")
        lines.append("| 算法 | 改进(元) | 改进(%) |")
        lines.append("|------|---------|---------|")
        for a in ['greedy_urgent', 'sa', 'ga', 'random']:
            if a not in results:
                continue
            imp = baseline - results[a]['total_cost']
            imp_pct = imp / baseline * 100
            arrow = '↓' if imp > 0 else '↑'
            lines.append(f"| {ALGO_NAMES[a]} | {arrow} {abs(imp):.2f} | {arrow} {abs(imp_pct):.1f}% |")
        lines.append("")

    # 约束违反分析
    lines.append("---")
    lines.append("")
    lines.append("## 约束违反分析")
    lines.append("")
    for a in ALGO_ORDER:
        if a not in results:
            continue
        md_text, _ = problem.violation_report_markdown(
            results[a]['routes'], ALGO_NAMES.get(a, a)
        )
        lines.append(md_text)
        lines.append("")

    return '\n'.join(lines)


def run_scenario(scenario_name):
    """运行场景，生成全部图表和 Markdown 报告"""
    print(f"运行中... 场景: {scenario_name}")

    cfg = SCENARIOS[scenario_name]
    customers_dict = generate_scenario(scenario_name, seed=SEED)
    problem = build_problem(customers_dict, cfg['num_drones'])

    results = {}
    for algo_name in ALGO_ORDER:
        t0 = time.time()
        routes, cost, history = SOLVERS[algo_name](problem, seed=SEED)
        elapsed = time.time() - t0
        eval_result = problem.evaluate_solution(routes)
        eval_result['routes'] = routes
        eval_result['history'] = history
        eval_result['runtime'] = elapsed
        results[algo_name] = eval_result

    # 图表生成
    print(f"  生成可视化图表...")
    plot_customer_distribution(problem.customers, save_to_file=True)

    if 'ga' in results:
        plot_optimal_routes(problem, results['ga']['routes'],
                            title='Genetic Algorithm - Optimal Delivery Routes',
                            save_to_file=True)

    compare = {k: results[k]['routes'] for k in ['greedy', 'sa', 'ga'] if k in results}
    plot_algorithm_comparison(problem, compare, save_to_file=True)

    summary = {}
    for a, r in results.items():
        summary[a] = {
            'total_cost': {'mean': r['total_cost'], 'std': 0},
            'total_distance': {'mean': r['total_distance'], 'std': 0},
            'makespan': {'mean': r['makespan'], 'std': 0},
        }
    plot_cost_comparison(summary, ALGO_NAMES, save_to_file=True)

    sa_h = results.get('sa', {}).get('history')
    ga_h = results.get('ga', {}).get('history')
    plot_convergence_curves(sa_history=sa_h, ga_history=ga_h, save_to_file=True)

    plot_multi_metric_comparison(summary, ALGO_NAMES, save_to_file=True)

    all_res = {a: [r] for a, r in results.items()}
    plot_runtime_vs_cost(all_res, ALGO_NAMES, save_to_file=True)

    routes_dict = {k: r['routes'] for k, r in results.items()}
    plot_load_distribution(problem, routes_dict, ALGO_NAMES, save_to_file=True)

    # 生成 Markdown 报告
    print(f"  生成文字报告...")
    md_content = _build_output_report(problem, results, scenario_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(OUTPUT_DIR, f'report_output_{scenario_name}_{timestamp}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"  图表已保存到 outputs/")
    print(f"  报告: {report_path}")
    print(f"  Done.")

    return problem, results


if __name__ == '__main__':
    # Standard scenario for PPT
    problem, results = run_scenario('standard')

    # Also generate small/large scenario data
    print(f"\n生成 small/large 场景数据...")
    generate_scenario('small', seed=SEED)
    generate_scenario('large', seed=SEED)

    print(f"\n{'='*60}")
    print(f"全部完成！8张图表 + 报告 保存在 outputs/")
    print(f"数据 CSV 在 data/output/")
    print(f"{'='*60}")
