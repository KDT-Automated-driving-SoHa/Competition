#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

from ar_track_alvar_msgs.msg import AlvarMarkers
from module.infos.ar_info import ARInfo


class ARSubscriber():
    def __init__(self, name="/ar_pose_marker"):
        # 원본 라이다 데이터
        self.raw_sign = None        # 주차 신호 정보
        self.raw_parking = None     # 주차 위치 정보

        # rospy subscriber
        rospy.Subscriber(name, AlvarMarkers, self._callback, queue_size=1)


    def _callback(self, msg):
        self.raw_sign = None
        self.raw_parking = None

        for marker in msg.markers:
            if marker.id == 1:
                self.raw_sign = marker
            elif marker.id == 2:
                self.raw_parking = marker


    def get_parking_sign(self):
        if self.raw_sign is None:
            return False, None

        position = self.raw_sign.pose.pose.position
        orientation = self.raw_sign.pose.pose.orientation
        return True, ARInfo(position, orientation)


    def get_parking_line(self):
        if self.raw_parking is None:
            return False, None

        position = self.raw_parking.pose.pose.position
        orientation = self.raw_parking.pose.pose.orientation
        return True, ARInfo(position, orientation)
