#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np

def get_sliding_window_function(points, thresh_count=15):
    if len(points) < thresh_count:
        return False, None

    xs, ys = [], []

    for x, y in points:
        xs.append(x)
        ys.append(y)
    
    xs = np.array(xs)
    ys = np.array(ys)

    x0 = np.polyfit(ys, xs, 2)
    result_f = np.poly1d(x0)
    return True, result_f


def get_linear_speed1(angle, max_speed):
    func = np.poly1d([-0.01, 0, max_speed])
    return func(angle)


def get_linear_speed2(angle, max_speed):
    return max_speed * np.cos(angle / 80.0)


def get_linear_steering_angle1(angle):
    return 24 * np.tan(angle/(np.pi * 11))


def get_linear_steering_angle2(angle):
    return 30 * np.tanh(0.049 * (angle - 23.5)) + 24.4


def get_linear_transform(func, delta_x, delta_y):
    a, b, c = func.c
    return func + [0, -2*a*delta_x, a*delta_x**2 - b*delta_x + delta_y]


if __name__ == "__main__":
    for i in range(1, 51):
        print(i, get_linear_steering_angle2(i))
        np.tanh