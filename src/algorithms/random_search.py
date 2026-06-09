#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机搜索基线 — 生成 N 个随机排列，用贪心解码器评估，取最优
"""

import random
import numpy as np
from .base import BaseAlgorithm
from .ga import _decode_permutation
from config import RS_N_ITERATIONS


class RandomSearch(BaseAlgorithm):
    def __init__(self, problem, config=None):
        super().__init__(problem, config)
        self.n_iter = config.get('n_iter', RS_N_ITERATIONS) if config else RS_N_ITERATIONS

    def solve(self):
        problem = self.problem
        n = problem.n
        best_routes = None
        best_cost = float('inf')
        history = []

        for iteration in range(self.n_iter):
            perm = list(range(n))
            random.shuffle(perm)
            routes = _decode_permutation(problem, perm)
            eval_result = problem.evaluate_solution(routes)
            cost = eval_result['total_cost']

            if cost < best_cost:
                best_cost = cost
                best_routes = routes

            if iteration % 1000 == 0:
                history.append((iteration, best_cost))

        history.append((self.n_iter, best_cost))
        self.best_solution = best_routes
        self.best_cost = best_cost
        self.history = history

        return best_routes, best_cost, history


def solve_random_search(problem, seed=None):
    """随机搜索求解入口"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    algo = RandomSearch(problem)
    return algo.solve()
