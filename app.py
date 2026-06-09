#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - Streamlit 交互式界面
运行方式: streamlit run app.py
"""

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from config import (
    SCENARIOS, SEED, BG, PANEL, GRID, TEXT_PRI, TEXT_SEC, DEPOT_COL, PALETTE,
    DEPOT_COORDS, ALGO_NAMES,
)
from data.generate_data import generate_scenario
from src.utils.factories import build_problem
from src.algorithms import SOLVERS
from visualization.route_map import _style_ax, _draw_depot
from matplotlib.patches import FancyArrowPatch


def plot_single_route(problem, routes, title="路径规划结果"):
    """单算法路径总览图"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    _style_ax(ax)
    _draw_depot(ax)

    customers = problem.customers
    active = [i for i, r in enumerate(routes) if r]

    for drone_idx in active:
        color = PALETTE[drone_idx % len(PALETTE)]
        route = routes[drone_idx]
        path = np.array([DEPOT_COORDS] +
                        [(customers[i].x, customers[i].y) for i in route] +
                        [DEPOT_COORDS])
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, alpha=0.2)
        for k in range(len(path) - 1):
            ax.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle='-|>', mutation_scale=9,
                color=color, linewidth=1.1, alpha=0.65, zorder=3, capstyle='round'))
        for ci in route:
            ax.scatter(customers[ci].x, customers[ci].y, s=32, color=color,
                       zorder=5, alpha=0.7, edgecolors=BG, linewidths=0.6)

    eval_result = problem.evaluate_solution(routes)
    ax.set_title(f"{title}\n成本: {eval_result['total_cost']:.1f}元 | "
                 f"距离: {eval_result['total_distance']:.1f}km | "
                 f"Makespan: {eval_result['makespan']:.1f}min",
                 color=TEXT_PRI, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("X (km)", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("Y (km)", color=TEXT_SEC, fontsize=9)
    plt.tight_layout()
    return fig


def plot_comparison_routes(problem, routes_dict):
    """多算法路线并排对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    customers = problem.customers
    plot_algos = ['greedy', 'ga', 'sa']

    for ax, algo_key in zip(axes, plot_algos):
        _style_ax(ax)
        _draw_depot(ax)
        if algo_key not in routes_dict:
            continue

        routes = routes_dict[algo_key]
        active = [i for i, r in enumerate(routes) if r]
        for drone_idx in active:
            color = PALETTE[drone_idx % len(PALETTE)]
            route = routes[drone_idx]
            path = np.array([DEPOT_COORDS] +
                            [(customers[i].x, customers[i].y) for i in route] +
                            [DEPOT_COORDS])
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, alpha=0.2)
            for k in range(len(path) - 1):
                ax.add_patch(FancyArrowPatch(
                    path[k], path[k + 1], arrowstyle='-|>', mutation_scale=8,
                    color=color, linewidth=1.0, alpha=0.65, zorder=3))
            for ci in route:
                ax.scatter(customers[ci].x, customers[ci].y, s=20, color=color,
                           zorder=5, alpha=0.6, edgecolors=BG, linewidths=0.4)

        eval_result = problem.evaluate_solution(routes)
        name = ALGO_NAMES.get(algo_key, algo_key)
        ax.set_title(f"{name}\n成本:{eval_result['total_cost']:.0f}元 "
                     f"距离:{eval_result['total_distance']:.0f}km",
                     color=TEXT_PRI, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel("X (km)", color=TEXT_SEC, fontsize=8)
        ax.set_ylabel("Y (km)", color=TEXT_SEC, fontsize=8)

    plt.tight_layout()
    return fig


def plot_convergence(history, title="收敛曲线"):
    """收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

    x = [h[0] for h in history]
    y = [h[1] for h in history]
    ax.plot(x, y, color=PALETTE[1], linewidth=2.0)
    ax.set_xlabel("迭代/代数", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("最优成本 (元)", color=TEXT_SEC, fontsize=9)
    ax.set_title(title, color=TEXT_PRI, fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="无人机配送路径规划",
        page_icon="🚁",
        layout="wide",
    )

    st.title("🚁 无人机集群配送路径规划")
    st.markdown(
        "基于贪心算法、模拟退火、遗传算法和随机搜索的"
        "低空物流路径优化系统 — 统一人民币成本模型"
    )

    with st.sidebar:
        st.header("⚙️ 参数设置")

        scenario = st.selectbox(
            "场景选择",
            ["standard", "small", "large"],
            index=0,
            format_func=lambda x: {
                'small': '小规模 (10客户/3架)',
                'standard': '标准规模 (30客户/5架)',
                'large': '大规模 (50客户/8架)',
            }[x],
        )

        algos_selected = st.multiselect(
            "算法选择",
            ['greedy', 'sa', 'ga', 'random'],
            default=['greedy', 'sa', 'ga'],
            format_func=lambda x: ALGO_NAMES[x],
        )

        seed = st.number_input("随机种子", 1, 9999, SEED)
        st.divider()

        with st.expander("🧬 遗传算法超参数", expanded=False):
            ga_pop = st.slider("种群规模", 50, 300, 100, 50)
            ga_gen = st.slider("最大代数", 50, 500, 200, 50)
            ga_cx = st.slider("交叉概率", 0.5, 0.95, 0.8, 0.05)
            ga_mut = st.slider("变异概率", 0.05, 0.40, 0.15, 0.05)

        with st.expander("🔥 模拟退火超参数", expanded=False):
            sa_T0 = st.slider("初始温度", 100, 5000, 1000, 100)
            sa_alpha = st.slider("降温速率", 0.80, 0.999, 0.98, 0.01)

        st.divider()
        run_clicked = st.button("▶ 运行算法", use_container_width=True, type="primary")

    if 'results' not in st.session_state:
        st.session_state.results = None
        st.session_state.problem = None

    if run_clicked:
        with st.spinner("🔄 正在生成数据并运行算法…"):
            cfg = SCENARIOS[scenario]
            customers_dict = generate_scenario(scenario, seed=seed)
            problem = build_problem(customers_dict, cfg['num_drones'])
            results = {}

            for algo_name in algos_selected:
                solve_fn = SOLVERS[algo_name]

                if algo_name == 'ga':
                    from src.algorithms.ga import GeneticAlgorithm
                    config = {'pop_size': ga_pop, 'n_gen': ga_gen,
                              'cx_prob': ga_cx, 'mut_prob': ga_mut}
                    algo = GeneticAlgorithm(problem, config=config)
                    routes, cost, history = algo.solve()
                elif algo_name == 'sa':
                    from src.algorithms.sa import SimulatedAnnealing
                    config = {'T0': sa_T0, 'alpha': sa_alpha}
                    algo = SimulatedAnnealing(problem, config=config)
                    routes, cost, history = algo.solve()
                else:
                    routes, cost, history = solve_fn(problem, seed=seed)

                eval_result = problem.evaluate_solution(routes)
                eval_result['routes'] = routes
                eval_result['history'] = history
                results[algo_name] = eval_result

            st.session_state.results = results
            st.session_state.problem = problem
            st.session_state.scenario_name = scenario

    if st.session_state.results is not None:
        results = st.session_state.results
        problem = st.session_state.problem

        st.divider()

        # 指标卡片
        cols = st.columns(len(results))
        for col, (algo_name, r) in zip(cols, results.items()):
            with col:
                st.metric(
                    label=ALGO_NAMES[algo_name],
                    value=f"{r['total_cost']:.1f} 元",
                    delta=f"距离:{r['total_distance']:.1f}km | Makespan:{r['makespan']:.1f}min",
                )

        st.divider()

        # 约束违反分析
        with st.expander("🔍 约束违反分析", expanded=False):
            for algo_name, r in results.items():
                violation_report = problem.get_violation_report(r['routes'])
                s = violation_report['summary']
                st.markdown(f"**{ALGO_NAMES.get(algo_name, algo_name)}**")
                c1, c2 = st.columns(2)
                with c1:
                    if s['hard_constraint_satisfied']:
                        st.success(f"硬约束: ✓ 全部满足")
                    else:
                        errs = []
                        if s['capacity_violations'] > 0:
                            errs.append(f"载重违反 {s['capacity_violations']} 架")
                        if s['range_violations'] > 0:
                            errs.append(f"航程违反 {s['range_violations']} 架")
                        st.error(f"硬约束: ✗ {'; '.join(errs)}")
                with c2:
                    if s['soft_violation_count'] == 0:
                        st.success(f"时间窗: ✓ 全部准时")
                    else:
                        st.warning(
                            f"延迟 {s['delay_customer_count']}/{problem.n} 客户 "
                            f"({s['delayed_customer_pct']:.0f}%), "
                            f"总延迟 {s['total_delay_min']:.0f}min"
                        )
                st.divider()

        # 路线对比图
        routes_dict = {k: r['routes'] for k, r in results.items()}
        if len(results) >= 2:
            st.subheader("📊 算法路线对比")
            compare_keys = [k for k in ['greedy', 'ga', 'sa'] if k in routes_dict]
            if compare_keys:
                compare_routes = {k: routes_dict[k] for k in compare_keys}
                fig = plot_comparison_routes(problem, compare_routes)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.subheader("🗺️ 配送路径")
            algo_name = list(results.keys())[0]
            fig = plot_single_route(problem, routes_dict[algo_name],
                                    title=ALGO_NAMES[algo_name])
            st.pyplot(fig)
            plt.close(fig)

        # 对比表
        st.subheader("📋 指标对比")
        import pandas as pd
        table_data = []
        baseline_cost = None
        for algo_name, r in results.items():
            if baseline_cost is None:
                baseline_cost = r['total_cost']
            row = {
                '算法': ALGO_NAMES[algo_name],
                '总成本(元) ★': f"{r['total_cost']:.2f}",
                '飞行成本(元)': f"{r['flight_cost']:.2f}",
                '运营成本(元)': f"{r['operation_cost']:.2f}",
                '延迟罚金(元)': f"{r['delay_penalty']:.2f}",
                '超重罚金(元)': f"{r['overload_penalty']:.2f}",
                '总距离(km)': f"{r['total_distance']:.2f}",
                'Makespan(min)': f"{r['makespan']:.1f}",
                '基尼系数': f"{r['load_gini']:.3f}",
                '可行': '✓' if r['is_feasible'] else '✗',
            }
            if baseline_cost and baseline_cost > 0:
                imp = (baseline_cost - r['total_cost']) / baseline_cost * 100
                row['vs最优提升'] = f"{imp:+.1f}%"
            table_data.append(row)

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # 收敛曲线
        if 'ga' in results and results['ga'].get('history'):
            st.subheader("📈 遗传算法进化曲线")
            fig = plot_convergence(results['ga']['history'], "遗传算法进化曲线")
            st.pyplot(fig)
            plt.close(fig)

        if 'sa' in results and results['sa'].get('history'):
            st.subheader("🌡️ 模拟退火收敛曲线")
            fig = plot_convergence(results['sa']['history'], "模拟退火收敛曲线")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("👈 请在侧边栏设置参数后点击「运行算法」按钮")
        st.markdown("""
        ### 使用说明

        1. **选择场景** — 小规模(10客户)/标准(30客户)/大规模(50客户)
        2. **选择算法** — 勾选需要对比的算法
        3. **调整超参** — 在折叠面板中调整GA/SA的超参数
        4. **点击运行** — 查看路径规划图和优化指标

        ---
        #### 算法简介

        | 算法 | 类型 | 特点 |
        |------|------|------|
        | 🎯 贪心算法 | 构造启发式 | 快速求解，作为性能基线 |
        | 🧬 遗传算法 | 元启发式 | 种群进化，全局搜索 |
        | 🔥 模拟退火 | 元启发式 | 温度递减，跳出局部最优 |
        | 🎲 随机搜索 | 随机基线 | 验证优化算法有效性 |

        #### 成本模型

        本系统采用**统一人民币成本模型**:
        - 飞行成本: 0.8 元/km
        - 运营时间成本: 1.0 元/min
        - 延迟罚金: 20.0 元/min
        - 超重罚金: 500.0 元/kg
        - 超航程罚金: 300.0 元/km
        """)


if __name__ == "__main__":
    main()
