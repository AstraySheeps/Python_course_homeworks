#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""无人机数据模型"""

from dataclasses import dataclass


@dataclass
class Drone:
    id: int
    capacity: float       # kg
    speed: float          # km/h
    max_range: float      # km
