#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


def get_left_right_line_points(linesP, frame, slope_low_thresh=0, slope_high_thresh=20):
    height, width = frame.shape[:2]
    center_x = width // 2

    # Hough Line에서 기울기와 위치를 통해 왼쪽/오른쪽 영역 구분
    left_line_points = []
    right_line_points = []

    for line in linesP:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:
            continue

        delta_x = x2 - x1
        delta_y = y2 - y1
        slope = 1.0 * delta_y / delta_x
    
        if slope_low_thresh < abs(slope) < slope_high_thresh:
            if slope < 0 and x2 < center_x - 150:
                left_line_points.append(((x1, y1), (x2, y2)))
            elif slope > 0 and x1 > center_x + 150:
                right_line_points.append(((x1, y1), (x2, y2)))

    return left_line_points, right_line_points


def get_average_slop_and_y_intercept(line_points, scan_bottom=320):
    if len(line_points) == 0:
        return None, None

    length = len(line_points)

    sum_x = 0
    sum_y = 0
    sum_slope = 0

    for (x1, y1), (x2, y2) in line_points:
        # line의 모든 x, y좌표 더하기
        sum_x += x1 + x2
        sum_y += y1 + y2

        # 기울기 계산
        delta_x = x2 - x1
        delta_y = y2 - y1
        slope = 1.0 * delta_y / delta_x
        # line의 모든 기울기 더하기
        sum_slope += slope

    avg_x = 1.0 * sum_x / (2 * length)
    avg_y = 1.0 * sum_y/(2 * length) + scan_bottom

    avg_slope = sum_slope / length
    avg_y_intercept = avg_y - avg_slope * avg_x

    return avg_slope, avg_y_intercept


def get_vanishing_point(frame, left_slope, left_y_intercept, right_slope, right_y_intercept):
    height, width = frame.shape[:2]
    center_x = width // 2

    # 왼쪽/오른쪽 영역에서 기울기가 없는 경우, 소실점은 상단 중앙으로 설정
    if left_slope is None and right_slope is None:
        return center_x, 0
    
    # 왼쪽 영역에서 기울기가 없는 경우,
    # y좌표는 상단, x좌표는 오른쪽 영역의 기울기와 y절편을 통해 상단에서 만나는 x좌표로 설정
    if left_slope is None:
        y = 0
        x = -right_y_intercept / right_slope
        # print("[Hough Line]: left slope is missing ({}, {})".format(x, y))
    # 오른쪽 영역에서 기울기가 없는 경우,
    # y좌표는 상단, x좌표는 왼쪽 영역의 기울기와 y절편을 통해 상단에서 만나는 x좌표로 설정
    elif right_slope is None:
        y = 0
        x = -left_y_intercept / left_slope
        # print("[Hough Line]: right slope is missing ({}, {})".format(x, y))
    # 왼쪽/오른쪽 영역에서 기울기가 없는 경우,
    else:
        delta_y_intercept = right_y_intercept - left_y_intercept
        delta_slope = right_slope - left_slope
        x = -delta_y_intercept / delta_slope
        y = left_slope * x + left_y_intercept
    return x, y


def get_steering_radian(frame):
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # 해당 이미지의 Hough Line 추출
    # 원본 코드: all_lines = cv2.HoughLinesP(frame, 1, math.pi/180, 30, 30, 10)

    linesP = cv.HoughLinesP(frame, rho=1, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)

    # line이 없는 경우
    if linesP is None:
        return False, None

    # 전체 Hough Line들에서 왼쪽/오른쪽 영역에 존재하는 Hough Line으로 구분
    left_line_points, right_line_points = get_left_right_line_points(linesP, frame)

    # 각 왼쪽/오른쪽 영역의 Hough Line에서 평균 기울기, 평균 y절편 계산
    left_slope, left_y_intercept = get_average_slop_and_y_intercept(left_line_points)
    right_slope, right_y_intercept = get_average_slop_and_y_intercept(right_line_points)

    # 왼쪽/오른쪽 영역의 기울기와 y절편을 통해 소실점 계산
    vanishing_x, vanishing_y = get_vanishing_point(frame, left_slope, left_y_intercept, right_slope, right_y_intercept)
    if center_x == vanishing_x:
        return False, None
    else:
        delta_x = center_x - vanishing_x
        delta_y = center_y - vanishing_y
        slope = -1.0 * delta_x / delta_y
        radian = np.arctan(slope)

    return True, radian
    