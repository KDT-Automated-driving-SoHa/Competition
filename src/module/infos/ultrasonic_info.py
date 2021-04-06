#!/usr/bin/env python
# -*- coding: utf-8 -*-

class UltrasonicInfo():
    def __init__(self, data=None):
        if data:
            # Xycar 방향에 따른 초음파 거리값
            self.left           = data[0]
            self.front_left     = data[1]
            self.front_middle   = data[2]
            self.front_right    = data[3]
            self.right          = data[4]
            self.back_right     = data[5]
            self.back_middle    = data[6]
            self.back_left      = data[7]
        else:
            # Xycar 방향에 따른 초음파 거리값
            self.left           = None
            self.front_left     = None
            self.front_middle   = None
            self.front_right    = None
            self.right          = None
            self.back_right     = None
            self.back_middle    = None
            self.back_left      = None


    def __repr__(self):
        if self.left:
            return "BL {} | BM {} | BR {}".format(self.back_left, self.back_middle, self.back_right)
        return ""
