#!/usr/bin/env python
#-*- coding: utf-8 -*-

class MotorInfo():
    def __init__(self, angle=0, speed=0, iterations=1, delay_sec=0):
        self.angle = angle
        self.speed = speed
        self.iterations = iterations
        self.delay_sec = delay_sec


    def __repr__(self):
        return "angle {: >5.1f} | speed {: >4.1f} | iterations {: >2d} | delay {: >2.1f} sec".format(
                self.angle, self.speed, self.iterations, self.delay_sec)
