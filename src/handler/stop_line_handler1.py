#!/usr/bin/env python
# -*- coding: utf-8 -*-
from handler import AbstractHandler
from module.infos.motor_info import MotorInfo
from module.stop_line.preprocessing import preprocessing_stopline
from module.stop_line.stop_line import is_detect_crossline

class StopLineHandler(AbstractHandler):

    def __init__(self, speed=50):
        self.speed = speed


    def handle(self, handler_info):
        frame = handler_info.image
        prev_angle = handler_info.prev_angle
        
        preprcessed_frame = preprocessing_stopline(frame, thres_L=200)
        ret, where = is_detect_crossline(preprcessed_frame, self.speed)

        if ret:
            motor_info_list = []

            a = 50
            b = 70
            cycle = int(round((600 - where + a) / b))
            motor_info_list.append(MotorInfo(angle=0, speed=self.speed*1//2, iterations=cycle, delay_sec=0.05))

            motor_info_list.append(MotorInfo(angle=0, speed=-3, iterations=2, delay_sec=0.05))
            motor_info_list.append(MotorInfo(angle=0, speed=0, iterations=58, delay_sec=0.1)) # stop for 6 sec
            motor_info_list.append(MotorInfo(angle=prev_angle/20, speed=self.speed, iterations=7, delay_sec=0.1))
            handler_info.laps_count += 1
            print("laps:",handler_info.laps_count)
            
            return motor_info_list, handler_info
        else:
            return self._next_handler.handle(handler_info)
