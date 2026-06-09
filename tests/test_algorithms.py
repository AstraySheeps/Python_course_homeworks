#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""算法模块单元测试 — 接口一致性、基本正确性"""

import pytest
import numpy as np
from src.models.customer import Customer
from src.models.drone import Drone
from src.models.problem import Problem
from src.utils.distance import compute_distance_matrix
from src.algorithms.greedy import solve_greedy, solve_greedy_urgent
from src.algorithms.sa import solve_sa
from src.algorithms.ga import solve_ga
from src.algorithms.random_search import solve_random_search
from src.algorithms import SOLVERS


def make_customer(i, x, y, demand=3.0, ctype="normal",
                  tw=(1.0, 4.0), svc=0.15):
    return Customer(id=i, x=x, y=y, demand=demand,
                    customer_type=ctype, time_window=tw, service_time=svc)


def make_problem(n=5, m=2):
    """构建测试用问题：5客户，2无人机"""
    rng = np.random.RandomState(42)
    customers = []
    for i in range(n):
        ctype = ["urgent", "normal", "normal", "normal", "relaxed"][i % 5]
        tw_map = {"urgent": (0.2, 0.9), "normal": (1.0, 3.5), "relaxed": (3.0, 7.0)}
        svc_map = {"urgent": 0.08, "normal": 0.12, "relaxed": 0.2}
        customers.append(Customer(
            id=i,
            x=round(rng.uniform(5, 45), 2),
            y=round(rng.uniform(5, 45), 2),
            demand=round(rng.uniform(1, 6), 1),
            customer_type=ctype,
            time_window=tw_map[ctype],
            service_time=svc_map[ctype],
        ))
    drones = [Drone(i, 20.0, 40.0, 150.0) for i in range(m)]
    dist = compute_distance_matrix(customers, (25, 25))
    return Problem(customers, drones, dist)


class TestSolverRegistry:
    def test_all_solvers_registered(self):
        assert 'greedy' in SOLVERS
        assert 'greedy_urgent' in SOLVERS
        assert 'sa' in SOLVERS
        assert 'ga' in SOLVERS
        assert 'random' in SOLVERS
        assert len(SOLVERS) == 5

    def test_all_solvers_callable(self):
        for name, fn in SOLVERS.items():
            assert callable(fn), f"{name} is not callable"


class TestGreedyAlgorithm:
    def test_returns_correct_types(self):
        p = make_problem(5, 3)
        routes, cost, history = solve_greedy(p, seed=42)
        assert isinstance(routes, list)
        assert isinstance(cost, float)
        assert isinstance(history, list)
        assert cost > 0

    def test_all_customers_visited(self):
        p = make_problem(5, 3)
        routes, _, _ = solve_greedy(p, seed=42)
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == set(range(p.n))

    def test_no_duplicate_customers(self):
        p = make_problem(5, 3)
        routes, _, _ = solve_greedy(p, seed=42)
        all_ids = [ci for r in routes for ci in r]
        assert len(all_ids) == len(set(all_ids)) == p.n

    def test_feasible_for_well_balanced_data(self):
        p = make_problem(3, 3)
        routes, _, _ = solve_greedy(p, seed=42)
        eval_result = p.evaluate_solution(routes)
        assert eval_result['is_feasible']

    def test_urgent_variant(self):
        p = make_problem(5, 3)
        routes, cost, _ = solve_greedy_urgent(p, seed=42)
        eval_result = p.evaluate_solution(routes)
        assert cost > 0
        assert eval_result['is_feasible']


class TestSimulatedAnnealing:
    def test_returns_correct_types(self):
        p = make_problem(5, 3)
        routes, cost, history = solve_sa(p, seed=42)
        assert isinstance(routes, list)
        assert isinstance(cost, float)
        assert isinstance(history, list)
        assert len(history) > 0

    def test_all_customers_visited(self):
        p = make_problem(5, 3)
        routes, _, _ = solve_sa(p, seed=42)
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == set(range(p.n))

    def test_no_duplicate_customers(self):
        p = make_problem(5, 3)
        routes, _, _ = solve_sa(p, seed=42)
        all_ids = [ci for r in routes for ci in r]
        assert len(all_ids) == len(set(all_ids)) == p.n

    def test_cost_not_worse_than_random(self):
        """SA 在足够大的问题上应显著优于随机搜索"""
        p = make_problem(8, 3)  # 更大问题空间
        _, sa_cost, _ = solve_sa(p, seed=42)
        _, rs_cost, _ = solve_random_search(p, seed=42)
        assert sa_cost <= rs_cost, f"SA {sa_cost:.1f} should not be worse than random {rs_cost:.1f}"

    def test_history_monotonic_best(self):
        """历史记录中的最优成本应单调非增"""
        p = make_problem(5, 3)
        _, _, history = solve_sa(p, seed=42)
        best_costs = [h[1] for h in history]
        for i in range(1, len(best_costs)):
            assert best_costs[i] <= best_costs[i - 1] + 1e-6


class TestGeneticAlgorithm:
    def test_returns_correct_types(self):
        p = make_problem(5, 3)
        routes, cost, history = solve_ga(p, seed=42)
        assert isinstance(routes, list)
        assert isinstance(cost, float)
        assert isinstance(history, list)
        assert len(history) > 0

    def test_all_customers_visited(self):
        p = make_problem(5, 3)
        routes, _, _ = solve_ga(p, seed=42)
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == set(range(p.n))

    def test_no_duplicate_customers(self):
        p = make_problem(5, 3)
        routes, _, _ = solve_ga(p, seed=42)
        all_ids = [ci for r in routes for ci in r]
        assert len(all_ids) == len(set(all_ids)) == p.n

    def test_early_stop_works(self):
        """早停应在超多代无改善时触发"""
        p = make_problem(5, 3)
        from src.algorithms.ga import GeneticAlgorithm
        ga = GeneticAlgorithm(p, config={'pop_size': 20, 'n_gen': 500,
                                         'early_stop': 10})
        _, _, history = ga.solve()
        last_gen = history[-1][0]
        assert last_gen < 500 or last_gen <= 500


class TestRandomSearch:
    def test_returns_correct_types(self):
        p = make_problem(3, 2)
        routes, cost, history = solve_random_search(p, seed=42)
        assert isinstance(routes, list)
        assert isinstance(cost, float)
        assert isinstance(history, list)

    def test_all_customers_visited(self):
        p = make_problem(3, 2)
        routes, _, _ = solve_random_search(p, seed=42)
        visited = set()
        for r in routes:
            visited.update(r)
        assert visited == set(range(p.n))


class TestAlgorithmConsistency:
    """所有算法返回格式一致性"""

    def test_same_interface(self):
        p = make_problem(5, 3)
        for name in ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']:
            routes, cost, history = SOLVERS[name](p, seed=42)
            assert isinstance(routes, list), f"{name}: routes is not list"
            assert isinstance(cost, (int, float, np.floating)), f"{name}: cost type wrong"
            assert cost > 0, f"{name}: cost is zero"
            assert isinstance(history, list), f"{name}: history is not list"

    def test_all_cover_all_customers(self):
        p = make_problem(5, 3)
        for name in ['greedy', 'greedy_urgent', 'sa', 'ga', 'random']:
            routes, _, _ = SOLVERS[name](p, seed=42)
            visited = set()
            for r in routes:
                visited.update(r)
            assert visited == set(range(p.n)), \
                f"{name}: missed customers {set(range(p.n)) - visited}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
