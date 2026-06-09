#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""客户数据模型"""

from dataclasses import dataclass


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float                        # kg
    customer_type: str                   # "urgent" | "normal" | "relaxed"
    time_window: tuple[float, float]     # (earliest, latest) 单位：小时
    service_time: float                  # 小时
