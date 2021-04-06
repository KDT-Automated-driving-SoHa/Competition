#!/usr/bin/env python
import time


class BACKWARD:
    
    def __init__(self):
	angle = 0
        speed = 0
    

    def set_motor(self, angle, speed):
        self.angle = angle
        self.speed = speed
	

    def set_data(self, data):
		
	if data==False:
	    self.set_motor(38, -21)
 	elif data:
	    self.set_motor(-38,21)	
		
    def get_motor(self):
        return self.angle, self.speed
