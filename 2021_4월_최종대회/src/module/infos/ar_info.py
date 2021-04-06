#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

import numpy as np

from tf.transformations import euler_from_quaternion

class ARInfo():
    def __init__(self, position, orientation):
        # AR의 위치
        self.dx = position.x * 100
        self.dy = position.y * 100
        self.dz = position.z * 100

        # AR의 방향(quaternion)
        self.ax = orientation.x
        self.ay = orientation.y
        self.az = orientation.z
        self.aw = orientation.w

        # AR의 방향(rpy)
        self.roll, self.pitch, self.yaw = euler_from_quaternion((self.ax, self.ay, self.ax, self.aw))

        # AR의 방향(rpy, degree)
        self.roll_deg = np.degrees(self.roll)
        self.pitch_deg = np.degrees(self.pitch)
        self.yaw_deg = np.degrees(self.yaw)
