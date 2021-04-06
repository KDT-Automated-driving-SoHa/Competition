#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
import cv2, math
import numpy as np


class TrnasformCP():

    def __init__(self, param, K = 1.0):
        ## distance ~= 435.662 PIXEL or 0.845m 
        ## 515.58 PIXEL / m
        ## Offset from lidar to ROI : 0.400 m

        ############### param set #########################
        
        self.C_roi_p_x = 480
        self.C_roi_p_y = 480
        self.C_roi_x = 0.931
        self.C_roi_y = 0.931
        self.Offset_roi = 0.350
        self.Extend_roi = 0.3
        self.Extend_roi_away = 0.2
        self.m_to_PIXEL = 515.58
        self.K_avoidance = 0.8
        ## offset : distance from LIDAR to camera ROI with m unit
        ## C_x, C_y : Width & Height of camera ROI with m unit
        
        self.alphas = []
        self.del_alphas = param[0]
        self.number = param[1]
        self.speed = 25

        self.check_cartesian_pt = 0


    def set_azimuth(self):
        self.alphas = [ self.del_alphas * i for i in range(self.number)]


    def rm_spherical(self, data, number):
        stand = int(self.number/2)
        number = int(number/2)
        return data[stand-number:stand+number], self.alphas[stand-number:stand+number]


    def get_cartesian(self, rs, als):
        cartesian = np.array([[0],[0],[0]])
        for r, alpha in zip (rs, als):
            cartesian = np.append(cartesian, np.array([[r*np.cos(alpha)], [r*np.sin(alpha)], [1]]), axis=1)
        return cartesian[:,1:]
        

    def transform_to_ROI(self, data, theta=-np.pi/2, dir_x=-1, dir_y=1):
        ## theta : rotate angle by z axis
        ## dir_x, dir_y : for invese axis, '-1' means invert result about a perendicular axis 
        data = np.array([dir_x*data[0,:], dir_y*data[1,:], data[2,:]])
        T = np.array([[np.cos(theta), -np.sin(theta), self.C_roi_y/2.0],
                    [np.sin(theta), np.cos(theta), +self.Offset_roi +self.C_roi_x],
                    [0, 0, 1]])
        return np.matmul(T, data)


    def rm_cartesian(self, cartesian, offset=0.0):
        result = np.array([[0],[0],[1]])
        num_car = cartesian.shape[1]
        saved = [cartesian[0,0], cartesian[1,0]]
        for i in range(num_car):
            temp = cartesian[:,i]
            if np.abs(temp[0] - saved[0]) < 0.001 or np.abs(temp[1] - saved[1]) < 0.001:
                # print("removed", temp)
                continue
            else:
                saved = temp

            if temp[0] > 0 and temp[0] < self.C_roi_x and temp[1] > 0 - self.Extend_roi_away and temp[1] < self.C_roi_y + self.Extend_roi:
                result = np.append(result, np.array([[temp[0]],[temp[1]],[temp[2]]]), axis=1)
        return result.shape[1]-1, self.m_to_PIXEL*result[:,1:]
    

    def get_line(self, image, num, data):
        CP_bool = np.zeros([self.C_roi_p_x, self.C_roi_p_y], dtype=np.uint8)
        for i in range(num):
            temp_x = int(data[0,i])
            temp_y = int(data[1,i])
            if temp_x < self.C_roi_p_x and temp_y < self.C_roi_p_y:
                CP_bool[temp_y, temp_x] = 255
        lines = cv2.HoughLinesP(CP_bool,1,math.pi/180,1,1,1)
            
        if lines == None:
            return False, image
            # exit(0)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            image = cv2.line(image, (x1, y1), (x2, y2), (0,0,255), 2)
        return True, image


    def poly_left(self, data):
        return 0


    def poly_right(self, data):
        return 479


    def get_maxpt(self, image, data, left, right, speed):
        max_dist_left = 0
        max_dist_right = 0
        max_ptset_left = None
        max_ptset_right = None
        num_pt = [0, 0]

        for i in range(data.shape[1]):
            temp_x = data[0,i]
            temp_y = data[1,i]
            left_x = left(temp_y)
            right_x = right(temp_y)

            dist_left = temp_x - left_x
            dist_right = right_x - temp_x
            center = left_x + (right_x - left_x) / 2
            if dist_left > 0 and dist_right > 0:
                print("pts", temp_x,temp_y)     
                if temp_x < center:
                    num_pt[0] += 1
                    cv2.circle(image, (int(temp_x), int(temp_y)), 5, (255,0,0), -1)
                    if dist_left > max_dist_left:
                        max_dist_left = dist_left
                        max_ptset_left = (int(temp_x), int(temp_y))
                elif temp_x > center: 
                    cv2.circle(image, (int(temp_x), int(temp_y)), 5, (255,255,0), -1)
                    num_pt[1] += 1
                    if dist_right > max_dist_right:
                        max_dist_right = dist_right
                        max_ptset_right = (int(temp_x), int(temp_y))
        # print("numpt", num_pt)
        ## obstacle in left side
        if num_pt[0] > num_pt[1]:
            if not max_ptset_right == None:      
                cv2.circle(image, max_ptset_right, 5, (0, 0, 255), -1)
                return [(max_ptset_right[0] + right(max_ptset_right[1])) * 4 // 7, max_ptset_right[1]], self.speed
            else: 
                cv2.circle(image, max_ptset_left, 5, (0, 0, 255), -1)
                return [(max_ptset_left[0] + right(max_ptset_left[1]))* 4 // 7, max_ptset_left[1]], self.speed
                
        elif num_pt[0] < num_pt[1]:
            if not max_ptset_left == None:    
                cv2.circle(image, max_ptset_left, 5, (0, 0, 255), -1)
                return [(max_ptset_left[0] + left(max_ptset_left[1])) * 3 // 7, max_ptset_left[1]], self.speed
            else: 
                cv2.circle(image, max_ptset_right, 5, (0, 0, 255), -1)
                return [(max_ptset_right[0] + left(max_ptset_right[1])) * 3 // 7, max_ptset_right[1]], self.speed
        return [240, 240], speed


    def get_segment(self, image, cp, left, right, speed):
        if left == None and right == None:
            target = [240, 240]
        elif left == None:
            target, speed = self.get_maxpt(image, cp, self.poly_left, right, speed)
        elif right == None:
            target, speed = self.get_maxpt(image, cp, left, self.poly_right, speed)
        else:
            target, speed = self.get_maxpt(image, cp, left, right, speed)
       
        slope = np.arctan(-(240 - target[0])/((479 + self.Extend_roi*self.m_to_PIXEL + 1e-4) - target[1]))
        
        print(target)
        angle = self.K_avoidance * np.degrees(slope)
       
        return angle, image, speed

