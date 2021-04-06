#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv
from module.utils.trackbar import Trackbar

# 파라미터 폴더 경로
dir_path = os.path.dirname(os.path.join(__file__))
module_path = os.path.dirname(dir_path)
src_path = os.path.dirname(module_path)
params_dir = os.path.join(src_path, "params", "image_processing")

tb_adap = Trackbar(os.path.join(params_dir, "adaptiveThreshold"), "adaptiveThreshold", debug=False)
tb_persp = Trackbar(os.path.join(params_dir, "perspective_transform"), "perspective_transform", debug=False)
tb_Canny = Trackbar(os.path.join(params_dir, "Canny"), "Canny", debug=False)
tb_Hough = Trackbar(os.path.join(params_dir, "HoughLinesP"), "HoughLinesP", debug=False)
tb_adap2 = Trackbar(os.path.join(params_dir, "adaptiveThreshold"), "adaptiveThreshold2", debug=False)


def preprocessing_hough_line(frame):
    roi_up = 320
    roi_down = roi_up + 40

    # gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # canny edge
    low_threshold = 90
    high_threshold = 120
    edge_img = cv.Canny(blur_gray, low_threshold, high_threshold)

    # ROI 
    roi = edge_img[roi_up:roi_down]
    return roi
    