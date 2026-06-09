#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机配送路径规划 — 全局参数配置
所有参数集中管理，便于调参和复现实验
"""

# ========== 场景参数 ==========
SCENARIOS = {
    "small":    {"num_customers": 10, "num_drones": 3},
    "standard": {"num_customers": 30, "num_drones": 5},
    "large":    {"num_customers": 50, "num_drones": 8},
}
AREA_SIZE = (0, 50)
DEPOT_COORDS = (25, 25)
SEED = 42
NUM_INDEPENDENT_RUNS = 30

# ========== 无人机参数 ==========
DRONE_CAPACITY = 20       # kg（单趟载重上限）
DRONE_SPEED = 40          # km/h
DRONE_MAX_RANGE = 150     # km（50x50区域含β对角线约92km，150有余量）
ROAD_FACTOR = 1.3         # 道路曲折系数

# ========== 客户参数 ==========
DEMAND_RANGE = (1, 8)                  # kg
SERVICE_TIME_BASE = (5, 15)            # 分钟（基础服务时间范围）
MAX_TOTAL_DEMAND_RATIO = 0.85          # 总需求 ≤ 此比例 × M × C
CUSTOMER_TYPE_DIST = {
    "urgent":  0.20,
    "normal":  0.60,
    "relaxed": 0.20,
}
TIME_WINDOW_SLOTS = {
    "urgent":  (0.0, 1.0),    # 08:00-09:00, 窗宽 30-45 min
    "normal":  (1.0, 4.0),    # 09:00-12:00, 窗宽 60-120 min
    "relaxed": (2.0, 8.0),    # 10:00-16:00, 窗宽 120-240 min
}
TIME_WINDOW_WIDTHS = {
    "urgent":  (0.5, 0.75),   # 30-45 min
    "normal":  (1.0, 2.0),    # 60-120 min
    "relaxed": (2.0, 4.0),    # 120-240 min
}

# ========== 成本系数（统一人民币） ==========
COST_PER_KM = 0.8                  # 元/km
COST_PER_MIN = 1.0                 # 元/min（运营时间成本）
PENALTY_DELAY_PER_MIN = 20.0       # 元/min（延迟罚金）
PENALTY_OVERLOAD_PER_KG = 500.0    # 元/kg
PENALTY_EXCESS_RANGE_PER_KM = 300.0  # 元/km

# ========== 模拟退火参数 ==========
SA_INIT_TEMP = 1000
SA_COOLING_RATE = 0.98
SA_MIN_TEMP = 0.01
SA_ITERATIONS_PER_TEMP = 100
SA_ADAPTIVE_PATIENCE = 50

# ========== 遗传算法参数 ==========
GA_POP_SIZE = 100
GA_N_GENERATIONS = 200
GA_CX_PROB = 0.8
GA_MUT_PROB = 0.15
GA_TOURNAMENT_SIZE = 3
GA_ELITE_SIZE = 5
GA_EARLY_STOP_GEN = 50

# ========== 随机搜索参数 ==========
RS_N_ITERATIONS = 10000

# ========== 可视化配色（暗色主题） ==========
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
DEPOT_COL = "#ff6b6b"
# ========== 算法名称映射 ==========
ALGO_NAMES = {
    'greedy': '贪心算法',
    'greedy_urgent': '贪心(紧急优先)',
    'sa': '模拟退火',
    'ga': '遗传算法',
    'random': '随机搜索',
}

PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
    "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb347",
    "#63e6be", "#f8a5c2", "#a9dc76", "#fc9867", "#ab9df2", "#78dce8",
]
