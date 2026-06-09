#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法 — 基于 DEAP
编码：排列编码（先排列，后分组）
解码：贪心分割（按顺序分配给无人机，超载则换下一架）
"""

import random
import numpy as np
from deap import base, creator, tools
from .base import BaseAlgorithm
from config import (
    GA_POP_SIZE, GA_N_GENERATIONS, GA_CX_PROB, GA_MUT_PROB,
    GA_TOURNAMENT_SIZE, GA_ELITE_SIZE, GA_EARLY_STOP_GEN,
)


def _decode_permutation(problem, perm):
    """将排列解码为路线（贪心分割）

    按排列顺序依次将客户分配给当前无人机，
    加入后超载则换下一架。
    """
    n = problem.n
    m = problem.m
    routes = [[] for _ in range(m)]
    drone_idx = 0

    for ci in perm:
        if drone_idx >= m:
            routes[-1].append(ci)
            continue

        drone = problem.drones[drone_idx]
        current_load = sum(problem.customers[i].demand for i in routes[drone_idx])

        if current_load + problem.customers[ci].demand > drone.capacity:
            drone_idx += 1
            if drone_idx >= m:
                routes[-1].append(ci)
                continue

        routes[drone_idx].append(ci)

    return [r for r in routes if r]


def _create_toolbox(problem):
    """创建并注册 DEAP toolbox"""
    n = problem.n

    # 幂等注册
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def create_individual():
        perm = list(range(n))
        random.shuffle(perm)
        return creator.Individual(perm)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        routes = _decode_permutation(problem, ind)
        eval_result = problem.evaluate_solution(routes)
        return (eval_result['total_cost'],)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)

    return toolbox


class GeneticAlgorithm(BaseAlgorithm):
    def __init__(self, problem, config=None):
        super().__init__(problem, config)
        self.pop_size = config.get('pop_size', GA_POP_SIZE) if config else GA_POP_SIZE
        self.n_gen = config.get('n_gen', GA_N_GENERATIONS) if config else GA_N_GENERATIONS
        self.cx_prob = config.get('cx_prob', GA_CX_PROB) if config else GA_CX_PROB
        self.mut_prob = config.get('mut_prob', GA_MUT_PROB) if config else GA_MUT_PROB
        self.elite_size = config.get('elite_size', GA_ELITE_SIZE) if config else GA_ELITE_SIZE
        self.early_stop = config.get('early_stop', GA_EARLY_STOP_GEN) if config else GA_EARLY_STOP_GEN

    def solve(self):
        problem = self.problem
        toolbox = _create_toolbox(problem)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(self.elite_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        # 初始评估
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        hof.update(pop)

        history = [(0, hof[0].fitness.values[0])]
        best_cost = hof[0].fitness.values[0]
        no_improve = 0

        for gen in range(1, self.n_gen + 1):
            # 选择 + 精英保留
            offspring = toolbox.select(pop, len(pop) - self.elite_size)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # 交叉
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.cx_prob:
                    toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # 变异
            for ind in offspring:
                if random.random() < self.mut_prob:
                    toolbox.mutate(ind)
                    del ind.fitness.values

            # 评估新个体
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            # 精英保留
            elites = [toolbox.clone(ind) for ind in hof.items]
            pop[:] = offspring + elites
            hof.update(pop)

            record = stats.compile(pop)
            history.append((gen, record['min']))

            # 早停检查
            if record['min'] < best_cost - 1e-6:
                best_cost = record['min']
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.early_stop:
                break

        # 解码最优解
        best_ind = hof[0]
        routes = _decode_permutation(problem, best_ind)
        eval_result = problem.evaluate_solution(routes)

        self.best_solution = routes
        self.best_cost = eval_result['total_cost']
        self.history = history

        return routes, self.best_cost, history


def solve_ga(problem, seed=None):
    """遗传算法求解入口"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    algo = GeneticAlgorithm(problem)
    return algo.solve()
