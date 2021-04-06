#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

import cv2 as cv
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo


class CameraSubscriber():
    def __init__(self, name="usb_cam/image_raw", info_name="/usb_cam/camera_info"):
        # 원본 이미지
        self.raw_image = None

        # 캘리브레이션 상세 정보
        self.width = None
        self.height = None
        self.distortion_coefficients = None
        self.src_camera_matrix = None
        self.dst_camera_matrix = None
        self.map_x = None
        self.map_y = None

        # msg to bgr
        self.bridge = CvBridge()

        # rospy subscriber
        self.sub_info = rospy.Subscriber(info_name, CameraInfo, self._callback_info, queue_size=1)
        self.sub_image = rospy.Subscriber(name, Image, self._callback_image, queue_size=1)


    def get(self):
        if self.raw_image is None:
            return False, None
        
        if self.map_x is not None   \
        and self.map_y is not None:
            cal_image = cv.remap(self.raw_image, self.map_x, self.map_y, cv.INTER_LINEAR)
            cal_x, cal_y, cal_width, cal_height = self.calibrated_roi
            cal_roi_image = cal_image[cal_y:cal_y+cal_height, cal_x:cal_x+cal_width]
            return True, cv.resize(cal_roi_image, (self.width, self.height))
        
        return True, self.raw_image
            

    def _callback_image(self, msg):
        self.raw_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    def _callback_info(self, msg):
        # calibration에 필요한 정보 저장
        self.width = msg.width
        self.height = msg.height
        self.distortion_coefficients = np.array(msg.D).reshape((1, 5))
        self.src_camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dst_camera_matrix, self.calibrated_roi = cv.getOptimalNewCameraMatrix(
                                                        self.src_camera_matrix,
                                                        self.distortion_coefficients,
                                                        (self.width, self.height),
                                                        1,
                                                        (self.width, self.height))
        self.dst_camera_matrix_stopline, self.calibrated_roi_stopline = cv.getOptimalNewCameraMatrix(
                                                        self.src_camera_matrix,
                                                        self.distortion_coefficients,
                                                        (self.width, self.height),
                                                        1,
                                                        (self.width, self.height))
        self.map_x, self.map_y = cv.initUndistortRectifyMap(
                                    self.src_camera_matrix,
                                    self.distortion_coefficients,
                                    None,
                                    self.dst_camera_matrix,
                                    (self.width, self.height),
                                    cv.CV_32FC1)

        # 필요한 정보를 받았으므로, 구독 해제
        self.sub_info.unregister()
