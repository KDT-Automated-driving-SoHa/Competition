#!/usr/bin/env python
# -*- coding: utf-8 -*-
from handler import AbstractHandler
from module.infos.motor_info import MotorInfo
from module.ar_parking.ar_parking import ParkingBehavior

class ARParkingHandler(AbstractHandler):

    def __init__(self):
        self.behavior = ParkingBehavior()


    def handle(self, handler_info):
        ultrasonic_info = handler_info.ultrasonic_info
        ar_info1 = handler_info.ar_info1
        ar_info2 = handler_info.ar_info2
        laps_count = handler_info.laps_count
        is_parking = handler_info.is_parking

        if is_parking or (laps_count >= 3 and (ar_info1 or ar_info2)):
            motor_info = self.behavior.get_motor_info(ultrasonic_info, ar_info1, ar_info2)
            motor_info.delay_sec = 0.1
            handler_info.is_parking = True
            return motor_info, handler_info
        else:
            return self._next_handler.handle(handler_info)
