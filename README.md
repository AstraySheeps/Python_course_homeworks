# 无人机集群配送路径规划

基于 Python 的无人机集群配送路径优化系统。在贴近真实低空物流场景下，设计并对比四种优化算法（贪心、模拟退火、遗传、随机搜索），采用统一人民币成本模型，支持多场景实验与统计显著性分析。

## 项目结构

```
无人机配送规划/
├── config.py                            # 全局参数（场景/无人机/成本/算法/配色）
├── main.py                              # 一键运行主脚本
├── app.py                               # Streamlit 交互式 Web 界面
├── requirements.txt                     # Python 依赖
├── data/
│   ├── generate_data.py                 # 多场景数据生成（含可行性校验）
│   └── output/                          # CSV 数据文件
├── src/
│   ├── models/                          # 数据模型（Customer/Drone/Problem）
│   ├── algorithms/                      # 算法实现（Base/Greedy/SA/GA/Random）
│   └── utils/                           # 工具（距离/验证/评估/统计检验）
├── visualization/                       # 可视化（8张图表）
│   └── output/                          # 图表 PNG 输出
├── experiments/                         # 批量实验与敏感性分析
│   └── output/                          # JSON 结果文件
├── legacy/                              # v2.0 旧代码（保留参考）
├── docs/                                # 文档
│   ├── 最终实施方案（单趟路径问题）.md    # 最终执行方案
│   ├── 总体代码理解.md                   # 整体架构快速理解
│   ├── 完善记录.md                       # 历次修改记录
│   └── ...
└── tests/                               # 单元测试（预留）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键运行（标准场景，4种算法对比 + 全部可视化）
python main.py

# 3. 其他场景
python main.py --scenario small     # 小规模验证
python main.py --scenario large     # 大规模压力测试

# 4. 批量实验（30次独立运行 + 统计检验）
python main.py --runs 30

# 5. 交互式Web界面（推荐用于演示）
streamlit run app.py
```

## 问题描述

- **场景**：50×50 km 区域，仓库位于中心(25, 25)
- **客户**：10/30/50个，含三种类型（紧急20%/普通60%/宽松20%），各有不同的时间窗
- **无人机**：3/5/8架，载重20kg，速度40km/h，最大航程150km
- **目标**：最小化统一人民币总成本
- **成本模型**：
  - 飞行成本: 0.8 元/km
  - 运营时间成本: 1.0 元/min
  - 延迟罚金: 20.0 元/min
  - 超重罚金: 500.0 元/kg
  - 超航程罚金: 300.0 元/km

## 四种算法

| 算法 | 文件 | 策略 |
|------|------|------|
| 贪心算法 | `src/algorithms/greedy.py` | 最近客户优先 + 紧急优先两种变体 |
| 模拟退火 | `src/algorithms/sa.py` | 5种邻域操作 + 自适应冷却 + 重启机制 |
| 遗传算法 | `src/algorithms/ga.py` | DEAP排列编码 + OX交叉 + 锦标赛选择 + 早停 |
| 随机搜索 | `src/algorithms/random_search.py` | 10000次随机排列 + 贪心解码 |

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scenario` | 场景选择：small/standard/large | standard |
| `--seed` | 随机种子 | 42 |
| `--runs` | 独立运行次数 | 1 |
| `--algo` | 算法列表 | greedy sa ga random |

## 输出文件

运行后在 `visualization/output/` 生成 8 张图表：
- fig1: 客户分布图
- fig2: 最优配送路线图
- fig3: 算法路线对比
- fig4: 成本对比柱状图+误差棒
- fig5: 收敛曲线
- fig6: 多指标分组柱状图
- fig7: 运行时间vs成本散点图
- fig8: 负载分布堆叠柱状图

## 依赖

- NumPy — 数值计算
- SciPy — 统计检验
- Matplotlib — 可视化
- DEAP — 遗传算法框架
- Pandas — 数据处理
- Streamlit — 交互式Web界面

## 文档

- [`docs/最终实施方案（单趟路径问题）.md`](docs/最终实施方案（单趟路径问题）.md) — 完整执行方案
- [`docs/总体代码理解.md`](docs/总体代码理解.md) — 项目架构与核心代码快速理解
- [`docs/完善记录.md`](docs/完善记录.md) — 历次修改记录
- [`docs/项目完成度评估.md`](docs/项目完成度评估.md) — 各维度完成度评估
- [`docs/后续优化方案.md`](docs/后续优化方案.md) — 后续优化行动计划
