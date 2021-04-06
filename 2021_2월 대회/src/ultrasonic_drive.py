#!/usr/bin/env python
#import discreatefilter
import numpy as np
import time

class ULTRASONIC:
    def __init__(self):
        # threshold
        self.angle = 0
        self.speed =0
        self.new_dict={0:'L',1:'FL',2:'FM',3:'FR',4:'R',5:'BR',6:'BM',7:'BL'}
        self.thresh_side_away = 120
        self.thresh_side = 20
        self.thresh_front = 20

        self.thresh_bearing = 30

        self.flag = 0
        self.flag_back = False
        '''
        
        ## LPF
        freq = 20 * 1000
        f_cut = 500
        self.df_ultra = discreatefilter.filter(f_cut, freq, title='ultra')
        '''

    def set_motor(self, angle, speed):
        self.angle = angle
        self.speed = speed

    def set_speed_back(self, data):
        new_dict=self.new_dict
        
        for i in [2, 1, 3]:    
            if data[new_dict[i]] < self.thresh_front or self.flag_back:
                print("checking")
                self.angle = 0
                #self.flag_back = True
                self.flag += 1
            
        if self.flag>5:
            self.angle, self.speed = 0,-20
            self.flag_back = True
            

        '''
                if self.flag < 5:
                    self.speed = 0
                    time.sleep(0.1)
                    return 0    
                else:
                    self.speed = -20
                    time.sleep(0.1)
                    break
        '''
        '''
                       
        if self.flag > 10:
            self.flag = 0
            self.flag_back = False
            self.speed = - self.speed
        return 0

        '''

    def set_data(self, data):
        #print(data)
        new_dict=self.new_dict
        data = data[0]
        #print(data)
        front = data['FM']
        v_left = data[new_dict[0]] + data[new_dict[1]] * 1.407
        v_right = data[new_dict[4]] + data[new_dict[3]] * 1.407
        bearing = - v_left + v_right

        '''
        ## LPF
        bearing = int(self.df_ultra.LPF(np.array([bearing])))
        
        for i in [0, 4]:    
            if data[new_dict[i]] < self.thresh_side - 40:
                bearing = -(i-2)*self.thresh_bearing
        '''
        if bearing < -50:
            bearing = -50
        elif bearing > 50:
            bearing = 50

        self.speed=25

        if data['FM']<50:
            if data['FL']<70 and data['FR']<70:
                self.set_speed_back(data)
            else:
                self.angle = bearing
        else:
            if data['FL']<50:
                self.angle = 50
                if data['FL']<30:
                    self.set_speed_back(data)
                return 0

            if data['FR']<50:
                self.angle = -50
                if data['FR']:
                    self.set_speed_back(data)
                return 0

            if abs(bearing) < 10:
                self.angle = 0
            else:
                self.angle = bearing




        """
        self.speed = 25
        k=1.0
        self.angle = bearing*k

        self.set_speed_back(data)
        """

    def get_motor(self):
        return self.angle, self.speed
