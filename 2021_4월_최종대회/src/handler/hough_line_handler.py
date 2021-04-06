#!/usr/bin/env python
# -*- coding: utf-8 -*-

from handler import AbstractHandler
from module.infos.motor_info import MotorInfo
from module.utils.discreate_filter import DiscreateFilter
from module.hough_line.preprocessing import preprocessing_hough_line
from module.hough_line.hough_line import get_steering_radian

class HoughLineHandler(AbstractHandler):

    def __init__(self):
        self.filter_deg = DiscreateFilter(f_cut=3000, freq=20000, num_of_signal=1)


    def handle(self, handler_info):
        frame = handler_info.image
        
        # 영상 전처리
        preprcessed_frame = preprocessing_hough_line(frame)

        # steering radian 계산
        ret, steering_rad = get_steering_radian(preprcessed_frame)
        if ret is False:
            steering_angle = self.filter_deg.prev_lpf[0]
        else:
            # steering radian 튜닝
            steering_angle = steering_rad / 1.5 * 40.0

        # steering angle 필터 적용
        steering_angle = self.filter_deg.get_lpf(steering_angle)[0]

        # steering angle에 따른 속도 조정
        speed = self._get_speed_by_angle(steering_angle)
        return MotorInfo(steering_angle, speed), handler_info

    
    def _get_speed_by_angle(self, angle, cof_v_rev=0.8):
        speed = 40.0

        ap = -(1 - cof_v_rev) / 10
        am = (1 - cof_v_rev) / 10
        
        bp = 1 - 20*ap
        bm = 1 + 20*am

        # angle이 10이하일 경우
        if abs(angle) < 10 :
            return speed
        # angle이 30이하일 경우
        elif abs(angle) < 30:
            if angle > 0:
                cof = ap*angle + bp
                return speed * cof
            else:
                cof = am*angle + bm
                return cof * speed
        # 그 외의 경우
        return cof_v_rev * speed
