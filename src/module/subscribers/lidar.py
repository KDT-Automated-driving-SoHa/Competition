#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

from sensor_msgs.msg import LaserScan


class LidarSubscriber():
    def __init__(self, name="/scan"):
        # 원본 라이다 데이터
        self.raw_lidar = None

        # 라이다 파라미터
        self.ranges = None
        self.number = None
        self.del_alpha = None

        # rospy subscriber
        self.sub_lidar = rospy.Subscriber(name, LaserScan, self._callback, queue_size=1)


    def get_param(self):
        if self.raw_lidar is None:
            return False, None
        
        self.del_alpha = self.raw_lidar.angle_increment
        self.ranges = self.raw_lidar.ranges
        self.number = len(self.ranges)
        return True, [ self.del_alpha, self.number ]
        

    def get(self):
        if self.raw_lidar is None:
            return False, None

        self.ranges = self.raw_lidar.ranges
        return True, self.ranges
            

    def _callback(self, msg):
        self.raw_lidar = msg
