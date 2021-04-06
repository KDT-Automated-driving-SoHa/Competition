#!/usr/bin/env python
import numpy as np
import rospy
import cv2
import time

from sensor_msgs.msg import Image

class YOLO:
    angle = 0
    speed = 20
    
    def __init__(self):
        self.width = 640
        self.height = 480
        self.displaymode = True

        self.motor = 20
        self.angle = 0
        self.time = 0
        self.time_interval = 100

        self.flag_rear = False
        self.center = 0
        self.time_incremental = 0

    def set_motor(self, angle, speed):
        self.angle = angle
        self.motor = speed

    def set_data(self, data):
        global angle
        label, bounding_boxes, image = data
        max_probability = 0.1
        for box in bounding_boxes:
            # if box["probability"] > max_probability:
                # print(len(box["class"]), box["class"], box["probability"])

            if "traffic_light" == box["class"] and box["probability"] > 0.5:
                self.angle = -50
                

            if label == box["class"] and box["probability"] > max_probability:
                # max_probability = box["probability"]
                # xmax : right
                # self.time_incremental += 1
                # if self.flag_dir == False:
                self.center = ( box["xmax"] - box["xmin"] ) / 2 + box["xmin"]    
                #     self.flag_dir = True
                # else:
                #     if self.time_incremental < 100:
                #         pass
                #     elif self.time_incremental < 200:
                #         self.center = - self.center
                #     else:
                #         self.center = 0
                #         self.flag_dir = False
                #         self.time_incremental = 0
                    
                # if self.displaymode:
                #     start = (box["xmin"], box["ymin"])
                #     end = (box["xmax"], box["ymax"])
                #     cv2.rectangle(image, start, end, (0,255,0), 2)
                
                
                self.angle = np.sign(self.center - 320) * 30
                
                print('detected!, ',box["probability"], self.center, self.angle)
            
            
            self.time_incremental += 1

            if self.time_incremental > 100:
                self.time_incremental = 0
                self.angle = 0

    def get_motor(self):
        return self.angle, self.motor

