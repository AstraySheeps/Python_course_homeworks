#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估工具"""

import numpy as np


def evaluate_batch(problem, solve_fn, n_runs=30, base_seed=0):
    """批量运行求解函数，返回多组结果的统计"""
    results = []
    for run in range(n_runs):
        seed = base_seed + run
        np.random.seed(seed)
        routes, cost, history = solve_fn(problem, seed=seed)
        eval_result = problem.evaluate_solution(routes)
        eval_result['run'] = run
        eval_result['seed'] = seed
        eval_result['history'] = history
        results.append(eval_result)

    return results


def compute_statistics(results_list, key='total_cost'):
    """计算均值、标准差、min、max"""
    values = [r[key] for r in results_list]
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),
        'min': np.min(values),
        'max': np.max(values),
        'values': values,
    }
