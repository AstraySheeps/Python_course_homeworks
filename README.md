# 无人机配送路径规划

基于 Python 的无人机集群配送路径优化系统，使用贪心算法、遗传算法和模拟退火算法对多无人机多客户配送场景进行路径规划与对比分析。

## 项目结构

```
无人机配送规划/
├── code/                               # 源代码
│   ├── common.py                       # 共享模块：配置常量、数据生成、清洗、距离矩阵
│   ├── drone_delivery.py               # 贪心算法基线
│   ├── drone_delivery_genetic.py       # 遗传算法（基于 DEAP）
│   ├── drone_delivery_sa.py            # 模拟退火算法
│   ├── main.py                         # 统一入口，支持算法切换与对比
│   └── requirements.txt                # Python 依赖
├── docs/                               # 文档
│   ├── 基线实现目标.txt                 # 项目初始任务说明
│   ├── 完善记录.md                      # 历次修改记录
│   ├── 遗传算法代码理解.md              # 遗传算法代码详解
│   └── 模拟退火算法代码理解.md          # 模拟退火算法代码详解
├── outputs/                            # 输出结果（图表、文本）
├── report/                             # 选题报告与中期报告
├── 项目要求/                            # 课程项目任务书
└── screenshots/                        # 截图
```

## 快速开始

```bash
cd code/
pip install -r requirements.txt
python main.py --algo all
```

## 问题描述

- **客户点**：20 个平面坐标，范围 [0, 100] × [0, 100]，每个客户包裹重量 1–5 kg
- **无人机**：N 架（默认 10），单架最大载重 20 kg，单架最大里程 200 单位
- **配送中心**：位于 (50, 50)
- **目标**：最小化总飞行距离
- **约束**：单架载重 ≤ 20 kg，单架里程 ≤ 200 单位

## 三种算法

| 算法 | 文件 | 策略 |
|------|------|------|
| 贪心算法 | `drone_delivery.py` | 每趟从最近客户开始装，装不下则尝试次近，直到填满 |
| 遗传算法 | `drone_delivery_genetic.py` | 排列编码 + 锦标赛选择 + 有序交叉 + 索引洗牌变异 + 精英保留 |
| 模拟退火 | `drone_delivery_sa.py` | 排列编码 + Swap/Reverse/Insert 邻域 + Metropolis 准则 + 指数降温 + 多轮重启 |

三种算法共享相同的编码/解码方式：将客户访问顺序编码为排列，解码时按顺序贪心打包成满足约束的路线。

## 运行方式

```bash
# 单独运行某个算法
python main.py --algo greedy       # 仅贪心算法
python main.py --algo genetic      # 仅遗传算法
python main.py --algo sa           # 仅模拟退火
python main.py --algo both         # 贪心 vs 遗传（默认）
python main.py --algo all          # 三种算法对比

# 设置无人机数量
python main.py --algo all --num-drones 10

# 批处理模式（不弹窗）
python main.py --algo all --no-show

# 指定输出目录
python main.py --algo all --output-dir ./my_outputs

# 模拟退火调参
python drone_delivery_sa.py --t0 1000 --alpha 0.99
```

## 典型结果

在随机种子 42 下，三种算法对比如下：

| 算法 | 总飞行距离 | 趟次 | 相比贪心提升 |
|------|-----------|------|-------------|
| 贪心算法 | 716.65 | 5 | — |
| 遗传算法 | 549.58 | 4 | +23.3% |
| 模拟退火 | 532.99 | 4 | +25.6% |

## 输出文件

每次运行在 `outputs/` 目录下生成：

| 文件 | 内容 |
|------|------|
| `drone_delivery_*_*.txt` | 贪心算法文本结果 |
| `drone_delivery_*_*.png` | 贪心算法路线图 |
| `drone_delivery_ga_*_*.txt` | 遗传算法文本结果 |
| `drone_delivery_ga_*_*_overview.png` | 遗传算法总览图 |
| `drone_delivery_ga_*_*_details.png` | 遗传算法各飞机详情 |
| `drone_delivery_ga_*_*_evolution.png` | 遗传算法进化曲线 |
| `drone_delivery_sa_*_*.txt` | 模拟退火文本结果 |
| `drone_delivery_sa_*_*_overview.png` | 模拟退火总览图 |
| `drone_delivery_sa_*_*_annealing.png` | 退火搜索轨迹 |
| `drone_delivery_sa_*_*_convergence.png` | 重启收敛对比 |
| `comparison_all_*_*.png` | 三算法对比图 |

## 依赖

- **NumPy** — 数值计算与随机数据生成
- **Matplotlib** — 路径可视化与收敛曲线
- **DEAP** — 遗传算法框架（仅 `drone_delivery_genetic.py` 需要）

## 文档

- [`docs/遗传算法代码理解.md`](docs/遗传算法代码理解.md) — 遗传算法的编码/解码/进化循环详细解释
- [`docs/模拟退火算法代码理解.md`](docs/模拟退火算法代码理解.md) — 模拟退火的 Metropolis 准则/邻域算子/降温策略详解
- [`docs/完善记录.md`](docs/完善记录.md) — 历次 bug 修复与功能改进记录
