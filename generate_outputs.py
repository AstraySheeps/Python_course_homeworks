#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键生成全部标准输出图表（用于PPT和汇报）"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import time, os, sys

from config import SCENARIOS, SEED, DEPOT_COORDS, DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE
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
    'greedy': 'Greedy',
    'greedy_urgent': 'Greedy(Urgent)',
    'sa': 'SA',
    'ga': 'GA',
    'random': 'Random',
}

ALGO_NAMES_CN = {
    'greedy': '贪心算法',
    'greedy_urgent': '贪心(紧急优先)',
    'sa': '模拟退火',
    'ga': '遗传算法',
    'random': '随机搜索',
}

ALGO_ORDER = ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']


def build_problem(customers_dict, num_drones):
    customers = [
        Customer(id=c['id'], x=c['x'], y=c['y'], demand=c['demand'],
                 customer_type=c['customer_type'],
                 time_window=(c['time_window_start'], c['time_window_end']),
                 service_time=c['service_time'])
        for c in customers_dict
    ]
    drones = [Drone(i, DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE) for i in range(num_drones)]
    dist_matrix = compute_distance_matrix(customers, DEPOT_COORDS)
    return Problem(customers, drones, dist_matrix)


SOLVE_FNS = {
    'greedy': solve_greedy,
    'greedy_urgent': solve_greedy_urgent,
    'sa': solve_sa,
    'ga': solve_ga,
    'random': solve_random_search,
}


def run_scenario(scenario_name):
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*70}")

    cfg = SCENARIOS[scenario_name]
    customers_dict = generate_scenario(scenario_name, seed=SEED)
    problem = build_problem(customers_dict, cfg['num_drones'])

    results = {}
    for algo_name in ALGO_ORDER:
        t0 = time.time()
        routes, cost, history = SOLVE_FNS[algo_name](problem, seed=SEED)
        elapsed = time.time() - t0
        eval_result = problem.evaluate_solution(routes)
        eval_result['routes'] = routes
        eval_result['history'] = history
        eval_result['runtime'] = elapsed
        results[algo_name] = eval_result
        ok_str = 'Y' if eval_result['is_feasible'] else 'N'
        print(f"  {ALGO_NAMES[algo_name]:<18} cost={cost:>10.2f}  dist={eval_result['total_distance']:>8.2f}km  "
              f"ms={eval_result['makespan']:>8.1f}min  t={elapsed:>6.2f}s  ok={ok_str}")

    # ---- 图表生成 ----
    print(f"\n  Generating charts...")

    # Fig1: Customer distribution
    plot_customer_distribution(problem.customers, save_to_file=True)

    # Fig2: Optimal route (GA)
    if 'ga' in results:
        plot_optimal_routes(problem, results['ga']['routes'],
                            title='Genetic Algorithm - Optimal Delivery Routes',
                            save_to_file=True)

    # Fig3: Algorithm route comparison
    compare = {k: results[k]['routes'] for k in ['greedy', 'sa', 'ga'] if k in results}
    plot_algorithm_comparison(problem, compare, save_to_file=True)

    # Fig4: Cost bar chart
    summary = {}
    for a, r in results.items():
        summary[a] = {
            'total_cost': {'mean': r['total_cost'], 'std': 0},
            'total_distance': {'mean': r['total_distance'], 'std': 0},
            'makespan': {'mean': r['makespan'], 'std': 0},
        }
    plot_cost_comparison(summary, ALGO_NAMES_CN, save_to_file=True)

    # Fig5: Convergence curves
    sa_h = results.get('sa', {}).get('history')
    ga_h = results.get('ga', {}).get('history')
    plot_convergence_curves(sa_history=sa_h, ga_history=ga_h, save_to_file=True)

    # Fig6: Multi-metric comparison
    plot_multi_metric_comparison(summary, ALGO_NAMES_CN, save_to_file=True)

    # Fig7: Runtime vs cost
    all_res = {a: [r] for a, r in results.items()}
    plot_runtime_vs_cost(all_res, ALGO_NAMES_CN, save_to_file=True)

    # Fig8: Load distribution
    routes_dict = {k: r['routes'] for k, r in results.items()}
    plot_load_distribution(problem, routes_dict, ALGO_NAMES_CN, save_to_file=True)

    # ---- Summary table ----
    print(f"\n  {'Algorithm':<18} {'Cost(yuan)':>12} {'Dist(km)':>10} {'Makespan(min)':>14} {'Delay(h)':>10} {'Gini':>8} {'Feasible':>8}")
    print(f"  {'-'*80}")
    for a in ALGO_ORDER:
        if a in results:
            r = results[a]
            ok = 'Y' if r['is_feasible'] else 'N'
            print(f"  {ALGO_NAMES_CN[a]:<18} {r['total_cost']:>12.2f} {r['total_distance']:>10.2f} "
                  f"{r['makespan']:>14.1f} {r['total_delay_time']:>10.4f} {r['load_gini']:>8.3f} {ok:>8}")

    baseline = results['greedy']['total_cost']
    print(f"  {'-'*80}")
    for a in ['greedy_urgent', 'sa', 'ga', 'random']:
        if a in results:
            imp = baseline - results[a]['total_cost']
            imp_pct = imp / baseline * 100
            arrow = 'v' if imp > 0 else '^'
            print(f"  {ALGO_NAMES_CN[a]} vs Greedy: {arrow} {abs(imp):.2f} yuan ({abs(imp_pct):.1f}% {'better' if imp > 0 else 'worse'})")

    print(f"\n  Charts saved to outputs/")
    return problem, results


if __name__ == '__main__':
    # Standard scenario for PPT
    problem, results = run_scenario('standard')

    # Also generate small scenario data
    print(f"\n\nGenerating small/large scenario data...")
    generate_scenario('small', seed=SEED)
    generate_scenario('large', seed=SEED)

    print(f"\n{'='*70}")
    print(f"All done! 8 charts in outputs/")
    print(f"Data CSV in data/output/")
    print(f"{'='*70}")
