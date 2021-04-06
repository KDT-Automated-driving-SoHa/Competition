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
        
        preprcessed_frame = preprocessing_stopline(frame, thres_L=190)
        ret, where = is_detect_crossline(preprcessed_frame, self.speed)

        if ret:
            motor_info_list = []
            '''
            # speed 50
            cycles = [11, #<90
                      14, #<190
                      4, #<290
                      3, #<390
                      1, #<490
                      1] #<600
            '''
            cycles = {90: 11,  #<90
                      116: 17,
                      135: 15,  # <190
                      190: 17,  #<190
                      200: 12,
                      240: 10,  #<240
                      310: 6,  #<290
                      350: 5,  # <390
                      390: 3,  #<390
                      450: 1,  # <490
                      490: 1,  #<490
                      580: 2,
                      600: 5} #<600

            if where <=90:
                motor_info_list.append(
                    MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[90], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-3, iterations=2, delay_sec=0.05))
            elif where < 116:
                motor_info_list.append(
                    MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[116], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-3, iterations=2, delay_sec=0.05))
            elif where < 135:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[135], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-3, iterations=2, delay_sec=0.05))
            elif where < 190:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[190], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-3, iterations=2, delay_sec=0.05))
            elif where < 200:
                motor_info_list.append(
                    MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[200], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-3, iterations=2, delay_sec=0.05))
            elif where < 240:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[240], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-3, iterations=2, delay_sec=0.05))
            elif where < 310:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[310], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-4, iterations=4, delay_sec=0.05))
            elif where < 350:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[350], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-4, iterations=4, delay_sec=0.05))
            elif where < 390:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[390], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-4, iterations=4, delay_sec=0.05))
            elif where < 450:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[450], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-4, iterations=9, delay_sec=0.05))
            elif where < 490:
                motor_info_list.append(MotorInfo(angle=0, speed=self.speed * 1 // 2, iterations=cycles[490], delay_sec=0.05))
                motor_info_list.append(MotorInfo(angle=0, speed=-4, iterations=8, delay_sec=0.05))
            elif where < 580:
                motor_info_list.append(MotorInfo(angle=0, speed=-5, iterations=cycles[580], delay_sec=0.05))
            else:
                #a = 50
                #b = 70
                #cycle = int(round((600 - where + a) / b))
                motor_info_list.append(MotorInfo(angle=0, speed=-5, iterations=cycles[600], delay_sec=0.05))
                #motor_info_list.append(MotorInfo(angle=0, speed=-5, iterations=5, delay_sec=0.05))

            motor_info_list.append(MotorInfo(angle=0, speed=0, iterations=58, delay_sec=0.1)) # stop for 6 sec
            motor_info_list.append(MotorInfo(angle=0, speed=self.speed, iterations=10, delay_sec=0.1))
            handler_info.laps_count += 1
            print("laps:",handler_info.laps_count)
            
            return motor_info_list, handler_info
        else:
            return self._next_handler.handle(handler_info)
