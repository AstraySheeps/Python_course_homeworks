#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""共享工厂函数"""

from config import DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE, DEPOT_COORDS
from src.models.customer import Customer
from src.models.drone import Drone
from src.models.problem import Problem
from src.utils.distance import compute_distance_matrix


def build_problem(customers_dict, num_drones):
    customers = [
        Customer(
            id=c['id'], x=c['x'], y=c['y'], demand=c['demand'],
            customer_type=c['customer_type'],
            time_window=(c['time_window_start'], c['time_window_end']),
            service_time=c['service_time'],
        )
        for c in customers_dict
    ]
    drones = [
        Drone(i, DRONE_CAPACITY, DRONE_SPEED, DRONE_MAX_RANGE)
        for i in range(num_drones)
    ]
    dist_matrix = compute_distance_matrix(customers, DEPOT_COORDS)
    return Problem(customers, drones, dist_matrix)
