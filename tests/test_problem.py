#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Problem 模型单元测试 — 成本函数、约束检查、时间链计算"""

import pytest
import numpy as np
from src.models.customer import Customer
from src.models.drone import Drone
from src.models.problem import Problem
from src.utils.distance import compute_distance_matrix


def make_customer(i, x, y, demand=3.0, ctype="normal",
                  tw=(1.0, 4.0), svc=0.15):
    return Customer(id=i, x=x, y=y, demand=demand,
                    customer_type=ctype, time_window=tw, service_time=svc)


def make_drone(i, cap=20.0, speed=40.0, max_range=150.0):
    return Drone(id=i, capacity=cap, speed=speed, max_range=max_range)


def make_simple_problem(n_customers=5, n_drones=2):
    """构建一个简单的测试问题"""
    customers = [
        make_customer(0, 10, 10, 3, "urgent", (0.2, 0.8), 0.1),
        make_customer(1, 30, 15, 5, "normal", (1.0, 3.0), 0.15),
        make_customer(2, 20, 35, 2, "normal", (2.0, 5.0), 0.12),
        make_customer(3, 40, 40, 4, "relaxed", (3.0, 7.0), 0.2),
        make_customer(4, 15, 45, 6, "urgent", (0.5, 1.5), 0.08),
    ][:n_customers]
    drones = [make_drone(i) for i in range(n_drones)]
    dist = compute_distance_matrix(customers, (25, 25))
    return Problem(customers, drones, dist)


class TestCustomer:
    def test_create_customer(self):
        c = make_customer(0, 10, 20, 5.5, "urgent", (0.5, 1.5), 0.12)
        assert c.id == 0
        assert c.x == 10
        assert c.y == 20
        assert c.demand == 5.5
        assert c.customer_type == "urgent"
        assert c.time_window == (0.5, 1.5)
        assert c.service_time == 0.12


class TestDrone:
    def test_create_drone(self):
        d = make_drone(0, 20, 40, 150)
        assert d.id == 0
        assert d.capacity == 20
        assert d.speed == 40
        assert d.max_range == 150


class TestProblemInit:
    def test_init(self):
        p = make_simple_problem(5, 2)
        assert p.n == 5
        assert p.m == 2
        assert p.dist_matrix.shape == (6, 6)
        assert p.dist_matrix[0, 0] == 0

    def test_symmetric_distance_matrix(self):
        p = make_simple_problem(3, 2)
        for i in range(4):
            for j in range(4):
                assert abs(p.dist_matrix[i, j] - p.dist_matrix[j, i]) < 1e-10

    def test_distance_matrix_includes_depot(self):
        p = make_simple_problem(3, 2)
        assert p.dist_matrix[0, 1] > 0  # depot to customer 0


class TestRouteTimeline:
    def test_empty_route(self):
        p = make_simple_problem(3, 2)
        tl = p.compute_route_timeline([], 0)
        assert tl['flight_time'] == 0
        assert tl['total_distance'] == 0
        assert tl['route_load'] == 0

    def test_single_customer_route(self):
        p = make_simple_problem(3, 2)
        tl = p.compute_route_timeline([0], 0)
        assert tl['flight_time'] > 0
        assert tl['total_distance'] > 0
        assert tl['route_load'] == p.customers[0].demand
        assert len(tl['timeline']) == 1

    def test_early_arrival_waiting(self):
        """提前到达应产生等待时间"""
        c = make_customer(0, 10, 10, 3, "urgent", (5.0, 8.0), 0.1)
        p = Problem([c], [make_drone(0)], compute_distance_matrix([c], (25, 25)))
        tl = p.compute_route_timeline([0], 0)
        assert tl['waiting_time'] > 0
        assert tl['delay_time'] == 0

    def test_late_arrival_delay(self):
        """迟到应产生延迟时间"""
        c = make_customer(0, 10, 10, 3, "urgent", (0.0, 0.01), 0.1)
        p = Problem([c], [make_drone(0)], compute_distance_matrix([c], (25, 25)))
        tl = p.compute_route_timeline([0], 0)
        assert tl['delay_time'] > 0

    def test_timeline_start_service(self):
        """开始服务时间 = max(arrival, time_window_start)"""
        c = make_customer(0, 10, 10, 3, "normal", (0.5, 3.0), 0.15)
        p = Problem([c], [make_drone(0)], compute_distance_matrix([c], (25, 25)))
        tl = p.compute_route_timeline([0], 0)
        t = tl['timeline'][0]
        assert t['start'] >= 0.5  # >= time_window_start


class TestEvaluateSolution:
    def test_valid_solution(self):
        p = make_simple_problem(3, 3)
        routes = [[0], [1], [2]]
        result = p.evaluate_solution(routes)
        assert result['total_cost'] > 0
        assert result['flight_cost'] > 0
        assert result['operation_cost'] > 0
        assert result['is_feasible']

    def test_cost_components_sum(self):
        p = make_simple_problem(3, 3)
        routes = [[0, 1], [2], []]
        result = p.evaluate_solution(routes)
        expected = (result['flight_cost'] + result['operation_cost']
                    + result['delay_penalty'] + result['overload_penalty']
                    + result['excess_range_penalty'])
        assert abs(result['total_cost'] - expected) < 1e-6

    def test_overload_penalty(self):
        """超载应产生罚金"""
        p = make_simple_problem(2, 1)
        # 无人机0载重20，但两个客户需求加起来可能不超载，所以直接设high demand
        c0 = make_customer(0, 10, 10, 15, "normal", (1, 5), 0.1)
        c1 = make_customer(1, 30, 30, 10, "normal", (1, 5), 0.1)
        d = make_drone(0, 20.0)
        p2 = Problem([c0, c1], [d], compute_distance_matrix([c0, c1], (25, 25)))
        routes = [[0, 1]]
        result = p2.evaluate_solution(routes)
        assert result['overload_penalty'] > 0
        assert result['total_overload'] > 0

    def test_feasible_when_no_hard_violations(self):
        p = make_simple_problem(3, 3)
        routes = [[0], [1], [2]]
        result = p.evaluate_solution(routes)
        assert result['is_feasible']

    def test_unfeasible_when_overload(self):
        c = make_customer(0, 10, 10, 25, "normal", (1, 5), 0.1)
        d = make_drone(0, 20.0)
        p = Problem([c], [d], compute_distance_matrix([c], (25, 25)))
        routes = [[0]]
        result = p.evaluate_solution(routes)
        assert not result['is_feasible']

    def test_makespan(self):
        p = make_simple_problem(3, 3)
        routes = [[0], [1], [2]]
        result = p.evaluate_solution(routes)
        assert result['makespan'] > 0
        assert result['makespan_hours'] > 0

    def test_num_active_drones(self):
        p = make_simple_problem(3, 5)
        routes = [[0], [1], [2], [], []]
        result = p.evaluate_solution(routes)
        assert result['num_active_drones'] == 3


class TestViolationReport:
    def test_no_violations(self):
        p = make_simple_problem(3, 3)
        routes = [[0], [1], [2]]
        report = p.get_violation_report(routes)
        assert report['summary']['hard_constraint_satisfied']
        assert report['summary']['capacity_violations'] == 0
        assert report['summary']['range_violations'] == 0

    def test_capacity_violation(self):
        c = make_customer(0, 10, 10, 25, "normal", (1, 5), 0.1)
        d = make_drone(0, 20.0)
        p = Problem([c], [d], compute_distance_matrix([c], (25, 25)))
        routes = [[0]]
        report = p.get_violation_report(routes)
        assert not report['summary']['hard_constraint_satisfied']
        assert report['summary']['capacity_violations'] > 0

    def test_delay_violation(self):
        c = make_customer(0, 10, 10, 3, "urgent", (0.0, 0.005), 0.1)
        p = Problem([c], [make_drone(0)], compute_distance_matrix([c], (25, 25)))
        routes = [[0]]
        report = p.get_violation_report(routes)
        assert report['summary']['delay_customer_count'] > 0

    def test_duplicate_visit_detection(self):
        p = make_simple_problem(2, 3)
        routes = [[0], [0], [1]]
        report = p.get_violation_report(routes)
        assert report['hard_violations']['visit_once']

    def test_empty_drone_depot_return(self):
        p = make_simple_problem(2, 3)
        routes = [[0], [], [1]]
        report = p.get_violation_report(routes)
        assert len(report['hard_violations']['depot_depart_return']) >= 1


class TestGini:
    def test_gini_equal(self):
        assert Problem._gini([5, 5, 5, 5]) == 0.0

    def test_gini_unequal(self):
        g = Problem._gini([10, 0, 0, 0])
        assert g > 0.5

    def test_gini_empty(self):
        assert Problem._gini([]) == 0.0

    def test_gini_single(self):
        assert Problem._gini([7]) == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
