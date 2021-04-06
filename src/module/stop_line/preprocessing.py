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


def preprocessing_stopline(frame, thres_L=180):
    blur = cv.GaussianBlur(frame, (5, 5), 0)
    _, L, _ = cv.split(cv.cvtColor(blur, cv.COLOR_BGR2HLS))
    # _, binary = cv.threshold(L, thres_L, 255, cv.THRESH_BINARY)

    # Perspective Transform 적용
    persp_mtx = {
        "margin": [0, 0, 30],
        "size": [1, 600, 640]}
    frame_height, frame_width = frame.shape[:2]
    margin = persp_mtx["margin"][1]

    roi_x1, roi_x2, roi_x3, roi_x4 = [265, 90, 375, 560]
    roi_y1, roi_y2 = 230, 310

    roi_x, roi_y = max(0, min(roi_x1, roi_x2)), max(0, min(roi_y1, roi_y2))
    roi_width, roi_height = min(max(roi_x3, roi_x4), frame_width) - roi_x, max(0, roi_y2 - roi_y1)

    tf_dst_size = persp_mtx["size"][1]
    tf_src_pts = np.array([[roi_x1, roi_y1], [roi_x2, roi_y2], [roi_x3, roi_y1], [roi_x4, roi_y2]],
                            dtype=np.float32)
    tf_dst_pts = np.array([[0, 0], [0, tf_dst_size], [tf_dst_size, 0], [tf_dst_size, tf_dst_size]],
                            dtype=np.float32)

    tf_matrix = cv.getPerspectiveTransform(tf_src_pts, tf_dst_pts)
    tf_image = cv.warpPerspective(L, tf_matrix, (480, tf_dst_size), flags=cv.INTER_LINEAR)
    
    # tf_image = cv.warpPerspective(binary, tf_matrix, (480, tf_dst_size), flags=cv.INTER_LINEAR)
    _, binary = cv.threshold(tf_image, thres_L, 255, cv.THRESH_BINARY)

    # Drawing ROI
    # roi = [(roi_x1, roi_y1), (roi_x2, roi_y2), (roi_x4, roi_y2), (roi_x3, roi_y1), (roi_x1, roi_y1)]
    # explain_image = cv.cvtColor(L, cv.COLOR_GRAY2BGR)
    # for pt1, pt2 in zip(roi[:-1], roi[1:]):
    #     cv.line(explain_image, pt1, pt2, (0, 0, 255))
    # cv.imshow("frame", explain_image)
    # cv.waitKey(1)
    return binary