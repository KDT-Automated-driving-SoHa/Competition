#!/usr/bin/env python

import cv2, math, time
import numpy as np
from tf.transformations import euler_from_quaternion

class PARK:

    def __init__(self):
	self.angle = 0
        self.speed = 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self.is_data = False
        self.is_back = False
        self.arData = {"DX":0.0,"DY":0.0,"DZ":0.0,"AX":0.0,"AY":0.0,"AZ":0.0,"AW":0.0}
        self.cycle = 0

    def set_motor(self, angle, speed):
        self.angle, self.speed = angle, speed
	
    def get_motor(self):
        return self.angle, self.speed
    
    def drive(self,Angle,Speed):
	self.set_motor(Angle, Speed)

    def back(self,ang,cy):
        print("back")
	self.is_back=True
        self.angle = ang
        self.cycle = cy
	#self.set_motor(ang,-25)
        #time.sleep(0.1)
	'''
        #cy=30
        for i in range(2):
            self.drive(0,0)
            time.sleep(0.1)

        for i in range(cy):
            self.drive(self.angle,-20)
            time.sleep(0.1)

        for _ in range(cy//3):
            self.drive(-self.angle,-20)
            time.sleep(0.1)
	'''

    def callback(self,Data):
   	#global arData, is_data

    	if len(Data) == 0:
            self.is_data = False
            self.arData["DX"] = 0.0
            self.arData["DY"] = 0.0
            self.arData["DZ"] = 0.0

            self.arData["AX"] = 0.0
            self.arData["AY"] = 0.0
            self.arData["AZ"] = 0.0
            self.arData["AW"] = 0.0
        else:

            self.is_data = True

            self.arData["DX"] = Data["pos_x"] * 100
            self.arData["DY"] = Data["pos_y"] * 100
            self.arData["DZ"] = Data["pos_z"] * 100

            self.arData["AX"] = Data["ori_x"]
            self.arData["AY"] = Data["ori_y"]
            self.arData["AZ"] = Data["ori_z"]
            self.arData["AW"] = Data["ori_w"]


    def set_data(self, data):
	#check len of data
        if len(data)==0:
            pass
        else:
            data2=data[0]
            self.callback(data2)

            (roll, pitch, yaw) = euler_from_quaternion((self.arData["AX"], self.arData["AY"], self.arData["AZ"], self.arData["AW"]))

            self.roll = math.degrees(roll)
            self.pitch = math.degrees(pitch)
            self.yaw = math.degrees(yaw)

            theta = math.atan(self.arData["DX"]/self.arData["DZ"])*180/math.pi if self.arData["DZ"] !=0 else 0
            w_t = 1
            #w_p = 0.5
            theta = self.arData["DX"]+9
            self.angle = int(theta * w_t)

            if self.is_data:
                print(self.arData,self.pitch)
                if int(self.arData["DZ"]) == 0:
                    self.speed,self.angle = 0,0
                elif int(self.arData["DZ"]) > 0 and int(self.arData["DZ"]) <= 30:
                    if (self.pitch) >= 10:
                        self.back(-50,30) # angle, cycle

                    elif self.pitch <=-10:
                        self.back(50,30)

                    else:
                        '''
                        if self.arData["DX"]>5.0:
                            self.back(-50,30)
                        elif self.arData["DX"]<-6.5:
                            self.back(50,30)
                        else:
                            self.speed = 0
                            self.is_back=False
                        '''
                        self.speed=0
                        self.is_back=False
                elif int(self.arData["DZ"]) > 30 and int(self.arData["DZ"]) <=50:
                    self.speed = 22
                    self.is_back=False
                else:
                    self.speed = 25
                    self.is_back=False
            else:
                self.speed,self.angle =0,0

    
