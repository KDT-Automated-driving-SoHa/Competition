#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

import numpy as np

from std_msgs.msg import Int32MultiArray
from module.infos.ultrasonic_info import UltrasonicInfo


class UltrasonicSubscriber():
    def __init__(self, name="/xycar_ultrasonic"):
        # 원본 초음파 데이터
        self.raw_data = None

        # rospy subscriber
        rospy.Subscriber(name, Int32MultiArray, self._callback, queue_size=1)


    def get(self):
        if self.raw_data is None:
            return False, None
        return True, UltrasonicInfo(self.raw_data)
            

    def _callback(self, msg):
        self.raw_data = msg.data
