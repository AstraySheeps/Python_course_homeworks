#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数敏感性分析 — 在标准场景下逐参数变化，观察算法性能变化
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCENARIOS, SEED
from data.generate_data import generate_scenario
from src.utils.factories import build_problem
from src.algorithms.sa import SimulatedAnnealing
from src.algorithms.ga import GeneticAlgorithm

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


def analyze_sa_params(scenario_name='standard', n_runs=10):
    """SA 参数敏感性分析"""
    cfg = SCENARIOS[scenario_name]
    customers = generate_scenario(scenario_name, seed=SEED)
    problem = build_problem(customers, cfg['num_drones'])

    # 测试不同冷却速率
    alphas = [0.90, 0.95, 0.98, 0.99, 0.995]
    # 测试不同初始温度
    temps = [100, 500, 1000, 2000, 5000]

    results = {'alpha': {}, 'T0': {}}

    for alpha in alphas:
        costs = []
        for run in range(n_runs):
            config = {'alpha': alpha}
            sa = SimulatedAnnealing(problem, config=config)
            routes, cost, _ = sa.solve()
            costs.append(cost)
        results['alpha'][str(alpha)] = {
            'mean': float(np.mean(costs)),
            'std': float(np.std(costs, ddof=1)),
        }
        print(f"  α={alpha}: cost={np.mean(costs):.2f} ± {np.std(costs, ddof=1):.2f}")

    for T0 in temps:
        costs = []
        for run in range(n_runs):
            config = {'T0': T0}
            sa = SimulatedAnnealing(problem, config=config)
            routes, cost, _ = sa.solve()
            costs.append(cost)
        results['T0'][str(T0)] = {
            'mean': float(np.mean(costs)),
            'std': float(np.std(costs, ddof=1)),
        }
        print(f"  T0={T0}: cost={np.mean(costs):.2f} ± {np.std(costs, ddof=1):.2f}")

    return results


def analyze_ga_params(scenario_name='standard', n_runs=10):
    """GA 参数敏感性分析"""
    cfg = SCENARIOS[scenario_name]
    customers = generate_scenario(scenario_name, seed=SEED)
    problem = build_problem(customers, cfg['num_drones'])

    pop_sizes = [50, 100, 200]
    mut_probs = [0.05, 0.10, 0.15, 0.20, 0.30]
    cx_probs = [0.6, 0.7, 0.8, 0.9]

    results = {'pop_size': {}, 'mut_prob': {}, 'cx_prob': {}}

    for ps in pop_sizes:
        costs = []
        for run in range(n_runs):
            config = {'pop_size': ps}
            ga = GeneticAlgorithm(problem, config=config)
            routes, cost, _ = ga.solve()
            costs.append(cost)
        results['pop_size'][str(ps)] = {
            'mean': float(np.mean(costs)),
            'std': float(np.std(costs, ddof=1)),
        }
        print(f"  pop_size={ps}: cost={np.mean(costs):.2f} ± {np.std(costs, ddof=1):.2f}")

    for mp in mut_probs:
        costs = []
        for run in range(n_runs):
            config = {'mut_prob': mp}
            ga = GeneticAlgorithm(problem, config=config)
            routes, cost, _ = ga.solve()
            costs.append(cost)
        results['mut_prob'][str(mp)] = {
            'mean': float(np.mean(costs)),
            'std': float(np.std(costs, ddof=1)),
        }
        print(f"  mut_prob={mp}: cost={np.mean(costs):.2f} ± {np.std(costs, ddof=1):.2f}")

    for cp in cx_probs:
        costs = []
        for run in range(n_runs):
            config = {'cx_prob': cp}
            ga = GeneticAlgorithm(problem, config=config)
            routes, cost, _ = ga.solve()
            costs.append(cost)
        results['cx_prob'][str(cp)] = {
            'mean': float(np.mean(costs)),
            'std': float(np.std(costs, ddof=1)),
        }
        print(f"  cx_prob={cp}: cost={np.mean(costs):.2f} ± {np.std(costs, ddof=1):.2f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='参数敏感性分析')
    parser.add_argument('--algo', default='both', choices=['sa', 'ga', 'both'])
    parser.add_argument('--runs', type=int, default=5,
                        help='每组参数重复次数')
    args = parser.parse_args()

    all_results = {}

    if args.algo in ('sa', 'both'):
        print("SA 参数敏感性分析...")
        all_results['sa'] = analyze_sa_params(n_runs=args.runs)

    if args.algo in ('ga', 'both'):
        print("GA 参数敏感性分析...")
        all_results['ga'] = analyze_ga_params(n_runs=args.runs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        OUTPUT_DIR, f'sensitivity_{timestamp}.json'
    )
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n敏感性分析结果已保存: {result_file}")
