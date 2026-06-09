#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""算法基类"""

from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    def __init__(self, problem, config=None):
        self.problem = problem
        self.config = config or {}
        self.best_solution = None
        self.best_cost = float('inf')
        self.history = []

    @abstractmethod
    def solve(self):
        """返回 (最优解routes, 最优成本, 收敛历史)"""
        pass
