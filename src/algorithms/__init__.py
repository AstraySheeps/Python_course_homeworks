#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""算法模块 — 求解器注册表"""

from src.algorithms.greedy import solve_greedy, solve_greedy_urgent
from src.algorithms.sa import solve_sa
from src.algorithms.ga import solve_ga
from src.algorithms.random_search import solve_random_search

SOLVERS = {
    'greedy': solve_greedy,
    'greedy_urgent': solve_greedy_urgent,
    'sa': solve_sa,
    'ga': solve_ga,
    'random': solve_random_search,
}
