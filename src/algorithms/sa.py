#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟退火算法
初始解：贪心算法
邻域操作：内部2-opt, 内部swap, 跨机relocate, 跨机exchange, 跨机or-opt
冷却策略：T0=1000, α=0.98, 自适应加速
"""

import random
import numpy as np
import copy
from .base import BaseAlgorithm
from config import (
    SA_INIT_TEMP, SA_COOLING_RATE, SA_MIN_TEMP,
    SA_ITERATIONS_PER_TEMP, SA_ADAPTIVE_PATIENCE,
)


class SimulatedAnnealing(BaseAlgorithm):
    def __init__(self, problem, config=None):
        super().__init__(problem, config)
        self.T0 = config.get('T0', SA_INIT_TEMP) if config else SA_INIT_TEMP
        self.alpha = config.get('alpha', SA_COOLING_RATE) if config else SA_COOLING_RATE
        self.T_min = config.get('T_min', SA_MIN_TEMP) if config else SA_MIN_TEMP
        self.iter_per_T = (config.get('iter_per_T', SA_ITERATIONS_PER_TEMP)
                          if config else SA_ITERATIONS_PER_TEMP)
        self.patience = config.get('patience', SA_ADAPTIVE_PATIENCE) if config else SA_ADAPTIVE_PATIENCE

    def _initial_solution(self):
        """贪心算法生成初始解"""
        from .greedy import _decode_greedy
        n = self.problem.n
        m = self.problem.m
        dist = self.problem.dist_matrix
        order = sorted(range(n), key=lambda i: dist[0, i + 1])
        routes = _decode_greedy(self.problem, order)
        # 扩展到 m 架
        while len(routes) < m:
            routes.append([])
        return routes

    def _neighbor_2opt(self, routes):
        """内部 2-opt: 同一无人机内逆转一段子路径"""
        new_routes = copy.deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if len(r) >= 2]
        if not non_empty:
            return new_routes
        k = random.choice(non_empty)
        route = new_routes[k]
        i, j = sorted(random.sample(range(len(route)), 2))
        route[i:j + 1] = reversed(route[i:j + 1])
        return new_routes

    def _neighbor_swap(self, routes):
        """内部 swap: 同一无人机内交换两个客户位置"""
        new_routes = copy.deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if len(r) >= 2]
        if not non_empty:
            return new_routes
        k = random.choice(non_empty)
        route = new_routes[k]
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return new_routes

    def _neighbor_relocate(self, routes):
        """跨机 relocate: 将一个客户从 A 移到 B"""
        new_routes = copy.deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if r]
        if len(non_empty) < 1:
            return new_routes
        src = random.choice(non_empty)
        src_route = new_routes[src]
        ci_idx = random.randrange(len(src_route))
        ci = src_route.pop(ci_idx)

        dst = random.randrange(len(new_routes))
        if dst == src and not src_route:
            dst = (dst + 1) % len(new_routes)
        insert_pos = random.randrange(len(new_routes[dst]) + 1) if new_routes[dst] else 0
        new_routes[dst].insert(insert_pos, ci)
        return new_routes

    def _neighbor_exchange(self, routes):
        """跨机 exchange: 交换两架无人机各一个客户"""
        new_routes = copy.deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if r]
        if len(non_empty) < 2:
            return new_routes
        a, b = random.sample(non_empty, 2)
        ia = random.randrange(len(new_routes[a]))
        ib = random.randrange(len(new_routes[b]))
        new_routes[a][ia], new_routes[b][ib] = new_routes[b][ib], new_routes[a][ia]
        return new_routes

    def _neighbor_oropt(self, routes):
        """跨机 or-opt: 将 2-3 个连续客户移到另一架"""
        new_routes = copy.deepcopy(routes)
        non_empty = [i for i, r in enumerate(new_routes) if len(r) >= 3]
        if not non_empty:
            return self._neighbor_relocate(routes)
        src = random.choice(non_empty)
        src_route = new_routes[src]
        seg_len = random.randint(2, min(3, len(src_route)))
        start = random.randrange(len(src_route) - seg_len + 1)
        segment = src_route[start:start + seg_len]
        del src_route[start:start + seg_len]

        dst = random.randrange(len(new_routes))
        insert_pos = random.randrange(len(new_routes[dst]) + 1) if new_routes[dst] else 0
        new_routes[dst][insert_pos:insert_pos] = segment
        return new_routes

    def _generate_neighbor(self, routes):
        """随机选择邻域操作"""
        ops = [
            self._neighbor_2opt,
            self._neighbor_swap,
            self._neighbor_relocate,
            self._neighbor_exchange,
            self._neighbor_oropt,
        ]
        op = random.choice(ops)
        return op(routes)

    def solve(self):
        problem = self.problem
        current = self._initial_solution()
        eval_cur = problem.evaluate_solution(current)
        current_cost = eval_cur['total_cost']

        best_solution = copy.deepcopy(current)
        best_cost = current_cost
        best_eval = eval_cur

        T = self.T0
        steps_no_improve = 0
        history = [(0, best_cost)]

        total_iter = 0
        while T > self.T_min:
            for _ in range(self.iter_per_T):
                neighbor = self._generate_neighbor(current)
                eval_neigh = problem.evaluate_solution(neighbor)
                neigh_cost = eval_neigh['total_cost']
                delta = neigh_cost - current_cost

                if delta < 0 or random.random() < np.exp(-delta / T):
                    current = neighbor
                    current_cost = neigh_cost
                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current)
                        best_cost = current_cost
                        best_eval = eval_neigh
                        steps_no_improve = 0
                    else:
                        steps_no_improve += 1

                total_iter += 1
                history.append((total_iter, best_cost))

            # 自适应冷却
            if steps_no_improve >= self.patience:
                T *= (self.alpha * 0.8)
                steps_no_improve = 0
            else:
                T *= self.alpha

        # 清理空路线
        best_solution = [r for r in best_solution if r]
        self.best_solution = best_solution
        self.best_cost = best_cost
        self.history = history

        return best_solution, best_cost, history


def solve_sa(problem, seed=None):
    """模拟退火求解入口"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    algo = SimulatedAnnealing(problem)
    return algo.solve()
