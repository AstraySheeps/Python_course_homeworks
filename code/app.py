#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 - Streamlit 交互式界面
运行方式: cd code/ && streamlit run app.py
"""

import sys
import os

# 必须在导入 pyplot 前设置 Agg 后端（避免 plt.show() 弹窗）
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Circle

from common import (
    generate_simulation_data, clean_data, compute_distance_matrix,
    generate_time_windows,
    RANDOM_SEED, NUM_CLIENTS, MAX_CAPACITY, MAX_DISTANCE, DRONE_SPEED,
    DEPOT_COORDS, BG, PANEL, GRID, TEXT_PRI, TEXT_SEC, DEPOT_COL, PALETTE,
    PENALTY_WEIGHT, SERVICE_TIME,
)
from drone_delivery import greedy_assignment


# ==================== 可视化函数 ====================

def _style_dark_ax(ax):
    """暗色主题坐标轴：背景、网格、坐标轴颜色"""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')


def _draw_routes_on_ax(ax, clients, routes, depot):
    """在给定坐标轴上绘制配送路线（暗色主题）"""
    ax.scatter(clients[:, 0], clients[:, 1], s=18, color=TEXT_SEC, alpha=0.35,
               zorder=2, edgecolors='none')

    for idx, route in enumerate(routes):
        color = PALETTE[idx % len(PALETTE)]
        path = np.array([depot] + [clients[i, :2] for i in route['route']] + [depot])

        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, alpha=0.18,
                solid_capstyle='round', zorder=2)
        for k in range(len(path) - 1):
            ax.add_patch(FancyArrowPatch(
                path[k], path[k + 1], arrowstyle='-|>', mutation_scale=9,
                color=color, linewidth=1.1, alpha=0.65, zorder=3, capstyle='round'))
        for ci in route['route']:
            ax.scatter(*clients[ci, :2], s=32, color=color, zorder=5, alpha=0.7,
                       edgecolors=BG, linewidths=0.6)

    # 配送中心（光晕+五角星）
    ax.scatter(*depot, s=280, color=DEPOT_COL, alpha=0.12, zorder=7, edgecolors='none')
    ax.scatter(*depot, s=90, color=DEPOT_COL, marker='*', zorder=8,
               edgecolors=BG, linewidths=0.7)


def plot_single_route(clients, routes, title="路径规划结果"):
    """单算法路径总览图，返回 matplotlib Figure"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    _style_dark_ax(ax)
    depot = np.array(DEPOT_COORDS)
    _draw_routes_on_ax(ax, clients, routes, depot)

    total_dist = sum(r['distance'] for r in routes)
    ax.set_title(f"{title}\n总距离: {total_dist:.1f}  |  趟次: {len(routes)}",
                 color=TEXT_PRI, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("X", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("Y", color=TEXT_SEC, fontsize=9)
    plt.tight_layout()
    return fig


def plot_comparison_side_by_side(triples, names):
    """三算法并排路径对比图

    Args:
        triples: [(clients, routes), ...] 每项为 (客户数据, 路线列表)
        names: 算法名称列表
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    depot = np.array(DEPOT_COORDS)

    for ax, (clients, routes), name in zip(axes, triples, names):
        _style_dark_ax(ax)
        _draw_routes_on_ax(ax, clients, routes, depot)

        total_dist = sum(r['distance'] for r in routes)
        ax.set_title(f"{name}\n总距离: {total_dist:.1f}  |  {len(routes)} 趟",
                     color=TEXT_PRI, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel("X", color=TEXT_SEC, fontsize=8)
        ax.set_ylabel("Y", color=TEXT_SEC, fontsize=8)

    plt.tight_layout()
    return fig


def plot_evolution_curve(logbook):
    """遗传算法进化曲线"""
    gens = np.array(logbook.select("gen"))
    mins = np.array(logbook.select("min"))
    avgs = np.array(logbook.select("avg"))

    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

    ax.fill_between(gens, avgs, mins, alpha=0.08, color="#58a6ff")
    ax.plot(gens, avgs, color="#58a6ff", linewidth=1.4, alpha=0.7,
            label="平均适应度", linestyle="--")
    ax.plot(gens, mins, color="#3fb950", linewidth=2.0, label="最优适应度")

    best_idx = int(np.argmin(mins))
    best_val = mins.min()
    ax.scatter([gens[best_idx]], [best_val], color="#f78166", s=55, zorder=6)
    ax.annotate(f"最优 {best_val:.1f} (第{gens[best_idx]}代)",
                xy=(gens[best_idx], best_val),
                xytext=(gens[best_idx] + len(gens) * 0.06,
                        best_val + (avgs.max() - mins.min()) * 0.12),
                color=TEXT_PRI, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=TEXT_SEC, lw=0.8))

    ax.set_xlabel("迭代代数", color=TEXT_SEC, fontsize=9)
    ax.set_ylabel("适应度（总飞行距离）", color=TEXT_SEC, fontsize=9)
    ax.set_title("遗传算法进化曲线", color=TEXT_PRI, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT_PRI)
    plt.tight_layout()
    return fig


def plot_annealing_curves(all_cost_history):
    """模拟退火搜索轨迹（每次重启一个子图）"""
    n = len(all_cost_history)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n), facecolor=BG, squeeze=False)
    axes = axes.flatten()
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]

    for idx, (cost_hist, temp_hist) in enumerate(all_cost_history):
        ax = axes[idx]
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_SEC, labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

        color = colors[idx % len(colors)]
        iters = np.arange(len(cost_hist))

        ax2 = ax.twinx()
        ax2.tick_params(colors=TEXT_SEC, labelsize=7)
        ax2.plot(iters, temp_hist, color="#f78166", linewidth=0.8, alpha=0.45,
                 linestyle="--")
        ax2.set_ylabel("温度", color=TEXT_SEC, fontsize=7.5)
        ax2.set_yscale("log")

        ax.plot(iters, cost_hist, color=color, linewidth=1.2, alpha=0.9)
        ax.set_ylabel("最优距离", color=TEXT_SEC, fontsize=7.5)

        ax.set_title(f"重启 {idx + 1}  |  最终最优距离: {cost_hist[-1]:.2f}",
                     color=TEXT_PRI, fontsize=9, fontweight='bold')

    ax.set_xlabel("迭代次数", color=TEXT_SEC, fontsize=7.5)
    fig.suptitle("模拟退火搜索轨迹", color=TEXT_PRI, fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    return fig


def plot_distance_bar_chart(results):
    """三算法总距离柱状图"""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.4, axis='y')

    names = []
    values = []
    colors_bar = ["#58a6ff", "#3fb950", "#f78166"]
    for key, name in [('greedy', '贪心算法'), ('genetic', '遗传算法'), ('sa', '模拟退火')]:
        if key in results:
            names.append(name)
            values.append(results[key]['total_distance'])

    bars = ax.bar(names, values, color=colors_bar[:len(names)], alpha=0.8,
                  edgecolor=BG, linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f'{val:.1f}', ha='center', va='bottom', color=TEXT_PRI, fontsize=10,
                fontweight='bold')

    ax.set_ylabel("总飞行距离", color=TEXT_SEC, fontsize=9)
    ax.set_title("算法总距离对比", color=TEXT_PRI, fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


# ==================== 主应用 ====================

def main():
    st.set_page_config(
        page_title="无人机配送路径规划",
        page_icon="🚁",
        layout="wide",
    )

    st.title("🚁 无人机集群配送路径规划")
    st.markdown("基于贪心算法、遗传算法和模拟退火的低空物流路径优化系统")

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header("⚙️ 参数设置")

        algo = st.selectbox(
            "算法选择",
            ["全部对比", "贪心算法", "遗传算法", "模拟退火"],
            index=0,
        )

        n_clients = st.slider("客户点数量", 10, 50, 20, 5,
                              help="仿真生成的客户点总数")
        n_drones = st.slider("无人机数量", 3, 20, 10,
                             help="机队规模（遗传/退火算法使用）")
        seed = st.number_input("随机种子", 1, 9999, 42,
                               help="控制数据生成和算法初始化的随机性")

        use_tw = st.checkbox("启用时间窗约束",
                             help="每个客户有配送时间窗口，早到等待、迟到惩罚")

        st.divider()

        # 遗传算法超参数
        if algo in ("全部对比", "遗传算法"):
            with st.expander("🧬 遗传算法超参数", expanded=False):
                ga_pop = st.slider("种群规模", 50, 500, 200, 50,
                                   help="每代的个体数量")
                ga_gen = st.slider("迭代代数", 50, 500, 200, 50,
                                   help="最大进化代数")
                ga_cx = st.slider("交叉概率", 0.50, 0.95, 0.80, 0.05,
                                  help="有序交叉的执行概率")
                ga_mut = st.slider("变异概率", 0.05, 0.40, 0.30, 0.05,
                                   help="索引洗牌变异的执行概率")

        # 模拟退火超参数
        if algo in ("全部对比", "模拟退火"):
            with st.expander("🔥 模拟退火超参数", expanded=False):
                sa_t0 = st.slider("初始温度", 100, 5000, 500, 100,
                                  help="起始温度，越高搜索越广")
                sa_alpha = st.slider("降温速率", 0.80, 0.999, 0.98, 0.01,
                                     help="越接近1降温越慢、搜索越充分")
                sa_restarts = st.slider("重启次数", 1, 5, 3,
                                        help="多次独立搜索取最优")

        st.divider()
        run_clicked = st.button("▶ 运行算法", use_container_width=True, type="primary")

    # ---- 初始化 session_state ----
    if 'results' not in st.session_state:
        st.session_state.results = None
        st.session_state.algo_label = None
        st.session_state.clients = None

    # ---- 执行算法 ----
    if run_clicked:
        with st.spinner("🔄 正在生成数据并运行算法…"):
            clients = generate_simulation_data(num_clients=n_clients, seed=seed)
            clients = clean_data(clients)
            dist_matrix = compute_distance_matrix(clients)

            time_windows = None
            if use_tw:
                time_windows = generate_time_windows(clients, dist_matrix, seed=seed)

            results = {}

            # 贪心算法
            if algo in ("全部对比", "贪心算法"):
                routes = greedy_assignment(clients, dist_matrix, MAX_CAPACITY, MAX_DISTANCE,
                                            time_windows=time_windows)
                results['greedy'] = {
                    'clients': clients,
                    'routes': routes,
                    'total_distance': sum(r['distance'] for r in routes),
                    'total_trips': len(routes),
                }

            # 遗传算法
            if algo in ("全部对比", "遗传算法"):
                import drone_delivery_genetic as ga_mod
                ga_mod.GA_POP_SIZE = ga_pop
                ga_mod.GA_NGEN = ga_gen
                ga_mod.GA_CXPB = ga_cx
                ga_mod.GA_MUTPB = ga_mut
                best_routes, logbook = ga_mod.run_genetic_algorithm(
                    clients, dist_matrix, n_drones, time_windows=time_windows,
                )
                results['genetic'] = {
                    'clients': clients,
                    'routes': best_routes,
                    'total_distance': sum(r['distance'] for r in best_routes),
                    'total_trips': len(best_routes),
                    'logbook': logbook,
                }

            # 模拟退火
            if algo in ("全部对比", "模拟退火"):
                import drone_delivery_sa as sa_mod
                sa_mod.SA_MAX_RESTARTS = sa_restarts
                _, best_routes, best_cost, cost_history = sa_mod.simulated_annealing(
                    clients, dist_matrix, n_drones,
                    t0=sa_t0, alpha=sa_alpha, time_windows=time_windows,
                )
                results['sa'] = {
                    'clients': clients,
                    'routes': best_routes,
                    'total_distance': best_cost,
                    'total_trips': len(best_routes),
                    'cost_history': cost_history,
                }

            st.session_state.results = results
            st.session_state.algo_label = algo
            st.session_state.clients = clients

    # ---- 展示结果 ----
    if st.session_state.results is not None:
        results = st.session_state.results
        algo_label = st.session_state.algo_label
        clients = st.session_state.clients

        st.divider()

        # ===== 指标卡片行 =====
        metric_specs = [
            ('greedy', '🎯 贪心算法'),
            ('genetic', '🧬 遗传算法'),
            ('sa', '🔥 模拟退火'),
        ]
        existing = [(k, name) for k, name in metric_specs if k in results]
        cols = st.columns(len(existing))
        for col, (key, name) in zip(cols, existing):
            r = results[key]
            with col:
                st.metric(label=name, value=f"{r['total_distance']:.1f}",
                          delta=f"{r['total_trips']} 趟次")

        st.divider()

        # ===== 全部对比模式 =====
        if algo_label == "全部对比":
            st.subheader("📊 三算法路径对比")

            triples = []
            names = []
            for key, name in [('greedy', '贪心'), ('genetic', '遗传'), ('sa', '退火')]:
                if key in results:
                    triples.append((results[key]['clients'], results[key]['routes']))
                    names.append(name)

            if len(triples) == 3:
                fig = plot_comparison_side_by_side(triples, names)
                st.pyplot(fig)
                plt.close(fig)

            # 对比表
            st.subheader("📋 指标对比")
            greedy_dist = results.get('greedy', {}).get('total_distance', 1)
            import pandas as pd

            table_data = []
            for key, name in [('greedy', '贪心算法'), ('genetic', '遗传算法'), ('sa', '模拟退火')]:
                if key in results:
                    r = results[key]
                    imp = (greedy_dist - r['total_distance']) / greedy_dist * 100
                    table_data.append({
                        '算法': name,
                        '总距离': f"{r['total_distance']:.2f}",
                        '趟次': r['total_trips'],
                        f'vs贪心提升': f"{imp:+.1f}%",
                    })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            # 柱状图
            st.subheader("📊 总距离柱状对比")
            fig_bar = plot_distance_bar_chart(results)
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            # 进化曲线
            if 'genetic' in results and 'logbook' in results['genetic']:
                st.subheader("📈 遗传算法进化曲线")
                fig_evo = plot_evolution_curve(results['genetic']['logbook'])
                st.pyplot(fig_evo)
                plt.close(fig_evo)

            # 退火轨迹
            if 'sa' in results and 'cost_history' in results['sa']:
                st.subheader("🌡️ 模拟退火搜索轨迹")
                fig_sa = plot_annealing_curves(results['sa']['cost_history'])
                st.pyplot(fig_sa)
                plt.close(fig_sa)

        # ===== 单算法模式 =====
        else:
            algo_to_key = {'贪心算法': 'greedy', '遗传算法': 'genetic', '模拟退火': 'sa'}
            key = algo_to_key[algo_label]

            if key in results:
                r = results[key]

                st.subheader(f"🗺️ {algo_label}配送路径")
                fig = plot_single_route(r['clients'], r['routes'], title=algo_label)
                st.pyplot(fig)
                plt.close(fig)

                if key == 'genetic' and 'logbook' in r:
                    st.subheader("📈 进化曲线")
                    fig_evo = plot_evolution_curve(r['logbook'])
                    st.pyplot(fig_evo)
                    plt.close(fig_evo)

                if key == 'sa' and 'cost_history' in r:
                    st.subheader("🌡️ 退火搜索轨迹")
                    fig_sa = plot_annealing_curves(r['cost_history'])
                    st.pyplot(fig_sa)
                    plt.close(fig_sa)

    else:
        st.info("👈 请在侧边栏设置参数后点击「运行算法」按钮")
        st.markdown("""
        ### 使用说明

        1. **选择算法** — 贪心、遗传、模拟退火或全部对比
        2. **调整参数** — 客户数量、无人机数量、各算法超参数
        3. **点击运行** — 查看路径规划图和优化指标
        4. **对比分析** — 「全部对比」模式下可横向比较三种算法性能

        ---
        #### 算法简介

        | 算法 | 类型 | 特点 |
        |------|------|------|
        | 🎯 贪心算法 | 构造启发式 | 快速求解，作为性能基线 |
        | 🧬 遗传算法 | 元启发式 | 种群进化，全局搜索 |
        | 🔥 模拟退火 | 元启发式 | 温度递减，跳出局部最优 |
        """)


if __name__ == "__main__":
    main()
