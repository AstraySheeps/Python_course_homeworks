#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计检验工具"""

import numpy as np
from scipy import stats


def paired_ttest(sample_a, sample_b, alpha=0.05):
    """配对 t 检验"""
    t_stat, p_value = stats.ttest_rel(sample_a, sample_b)
    significant = p_value < alpha
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'alpha': alpha,
    }


def effect_size_cohens_d(sample_a, sample_b):
    """Cohen's d 效应量"""
    mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
    std_a, std_b = np.std(sample_a, ddof=1), np.std(sample_b, ddof=1)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def summary_table(results_by_algo):
    """生成汇总表

    Args:
        results_by_algo: dict[str, list[dict]]
            例如 {'greedy': [...], 'sa': [...], 'ga': [...]}

    Returns:
        dict 含各算法、各指标 statistics
    """
    metrics = ['total_cost', 'flight_cost', 'operation_cost',
               'delay_penalty', 'total_distance', 'total_flight_time',
               'total_delay_time', 'makespan', 'load_gini',
               'num_active_drones', 'runtime', 'is_feasible']
    summary = {}
    for algo, results in results_by_algo.items():
        algo_summary = {}
        for metric in metrics:
            if metric == 'is_feasible':
                values = [1 if r[metric] else 0 for r in results]
                algo_summary['feasible_rate'] = np.mean(values)
            else:
                vals = [r[metric] for r in results]
                algo_summary[metric] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals, ddof=1),
                    'min': np.min(vals),
                    'max': np.max(vals),
                }
        summary[algo] = algo_summary
    return summary
