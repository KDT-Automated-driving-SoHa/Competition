#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, rospkg
import numpy as np
import cv2, random, math
from cv_bridge import CvBridge
from xycar_motor.msg import xycar_motor
from sensor_msgs.msg import Image
import filter
import matplotlib.pyplot as plt

import os
from utils.trackbar2 import Trackbar
from utils.controller import Controller
from IP.processing import *




mm5 = filter.MovingAverage(5)
mm10 = filter.MovingAverage(10)
pub = None

bridge = CvBridge()
image = np.empty(shape=[0])

Width = 640
Height = 480

Offset = 340
Gap = 40


history_x = None



def img_callback(data):
    global image    
    image = bridge.imgmsg_to_cv2(data, "bgr8")

# publish xycar_motor msg
def drive(Angle, Speed): 
    global pub

    msg = xycar_motor()
    msg.angle = Angle
    msg.speed = Speed

    pub.publish(msg)

# draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), color, 2)
    return img

# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 5, 15 + offset),
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (center-5, 15 + offset),
                       (center+5, 25 + offset),
                       (0, 255, 0), 2)    
    cv2.rectangle(img, (315, 15 + offset),
                       (325, 25 + offset),
                       (0, 0, 255), 2)
    return img

# left lines, right lines
def divide_left_right(lines):
    global Width

    low_slope_threshold = 0
    high_slope_threshold = 10

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1) / float(x2-x1)
        
        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x2 < Width/2 - 90):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width/2 + 90):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = float(x_sum) / float(size * 2)
    y_avg = float(y_sum) / float(size * 2)

    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(lines, left=False, right=False):
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)
    
    x1, x2 = 0, 0
    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)

    return x1, x2, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap
    global cann

    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # canny edge
    low_threshold = cann[0] #60
    high_threshold = cann[1] #100
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)
    #cv2.imshow('test', edge_img)
    # HoughLinesP
    roi = edge_img[Offset : Offset+Gap, 0 : Width]
    all_lines = cv2.HoughLinesP(roi,1,math.pi/180,30,30,10)

    # divide left, right lines
    if all_lines is None:
        return (0, 640), frame

    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    lx1, lx2, lpos = get_line_pos(left_lines, left=True)
    rx1, rx2, rpos = get_line_pos(right_lines, right=True)

    frame = cv2.line(frame, (int(lx1), Height), (int(lx2), (Height/2)), (255, 0,0), 3)
    frame = cv2.line(frame, (int(rx1), Height), (int(rx2), (Height/2)), (255, 0,0), 3)

    # draw lines
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255,255,255), 2)
                                 
    # draw rectangle
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)

    return (lpos, rpos), frame

def plot_moving_average():
    global angle_list, angle_p, angle_pid, time_list
    plt.figure()
    plt.plot(time_list, angle_list)
    plt.plot(time_list, angle_p)
    plt.plot(time_list, angle_pid)
    plt.legend(['ori', 'P', 'PID'])
    #plt.show()
    path = rospkg.RosPack().get_path('hough_drive_3') + '/src/pid.png'
    plt.savefig(path)
    print("bye!")


def hough_drive():
	global pub
	global image
    	global Width, Height
    	global angle_list, angle_pid, time_list
    	global params
	global error
 
        

        pos, frame = process_image(image)
        
        center = (pos[0] + pos[1]) / 2        
        error = (320 - center)
	
        
        cv2.imshow('steer',frame)


def slidig_file():
	global tb_clahe, tb_adapt, tb_perspective, tb_sliding, tb_contours
    
	history_x = None

	src_dir = os.path.dirname(os.path.realpath(__file__))
	pkg_dir = os.path.dirname(src_dir)
	parameter_dir = os.path.join(pkg_dir, "src/dataset/parameters")


	tb_clahe = Trackbar()
	tb_adapt = Trackbar(os.path.join(parameter_dir, "adaptiveThreshold"), "adaptiveThreshold")
	tb_perspective = Trackbar(os.path.join(parameter_dir, "perspective"), "perspective", debug=True)
	tb_sliding = Trackbar(os.path.join(parameter_dir, "slidingWindow"), "slidingWindow", debug=True)
	tb_contours = Trackbar(os.path.join(parameter_dir, "contours"), "contours")

	default_speed = 30
	history_x = [0, tb_sliding.getValue("win_width")]



def sliding_drive(frame):
    global history_x
    global tb_clahe, tb_adapt, tb_perspective, tb_sliding, tb_contours
    
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    clipLimit = tb_clahe.getValue("clipLimit", "default", 1, 10, 1000)
    tileGridSize = tb_clahe.getValue("tileGridSize", "default", 1, 5, 20)
    clahe = cv.createCLAHE(clipLimit=clipLimit/10.0, tileGridSize=(tileGridSize, tileGridSize))
    
    hls = clahe.apply(gray)

   
    adaptL = adaptiveThreshold(frame, tb_adapt)

    binary = adaptL



    pt_x1, pt_x2 = tb_perspective.getValue("x1"), tb_perspective.getValue("x2")
    pt_x3, pt_x4 = tb_perspective.getValue("x3"), tb_perspective.getValue("x4")
    pt_y1, pt_y2 = tb_perspective.getValue("y1"), tb_perspective.getValue("y2")

   
    src_pts = np.array([[pt_x1, pt_y1], [pt_x2, pt_y2], [pt_x3, pt_y1], [pt_x4, pt_y2]], dtype=np.float32)

   
    dst_height, dst_width = 480, 480

    dst_pts = np.array([[0, 0], [0, dst_height], [dst_width, 0], [dst_width, dst_height]], dtype=np.float32)

    tf_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    tf_matrix_inv = cv.getPerspectiveTransform(dst_pts, src_pts)

    for x, y in [[pt_x1, pt_y1], [pt_x2, pt_y2], [pt_x3, pt_y1], [pt_x4, pt_y2]]:
        cv.circle(frame, (x, y), 3, (0, 0, 255), -1)
    
    tf_image = cv.warpPerspective(binary, tf_matrix, (dst_width, dst_height), flags=cv.INTER_LINEAR)
    
    """filtering"""
    thresh_area = tb_contours.getValue("threshold_area")
    filter1 = removeContours(tf_image, thresh_area)

    win_width = tb_sliding.getValue("win_width")
    num_of_windows = tb_sliding.getValue("num_of_windows")
    step_window = tb_sliding.getValue("step_window")
    scan_width = tb_sliding.getValue("scan_width")
    scan_height = tb_sliding.getValue("scan_height")
    threshold = tb_sliding.getValue("threshold", "default", 0, 25, 100) / 100.0
    
    filtered_image = filter1
    #print(filter1.shape)

    left_coeff, right_coeff, viewer = sliding_window(filtered_image, win_width, (scan_width, scan_height), threshold, num_of_windows, step_window, tb_sliding)
    
    cv2.imshow("sliding", viewer)
    
    # TODO
    # left_coeff = [ coeff1, coeff2, coeff2 ]
    # len(left_coeff)
    decide_left_coeff = left_coeff[0] if len(left_coeff) > 0 else None
    decide_right_coeff = right_coeff[-1] if len(right_coeff) > 0 else None
    judge_y = tb_sliding.getValue("judge_y")

    


    center_degree, history_x, _ = get_radian_with_sliding(judge_y, tf_image.shape, decide_left_coeff, decide_right_coeff, history_x)

    
    center_degree = 90 - center_degree


    #return  center_degree
    return len(left_coeff), len(right_coeff)

        


def start():
	
    global pub
    global image
    global params, window
    
    slidig_file()
    
    rospy.init_node('auto_drive')
    pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1) 
    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)
    
    while not image.size == (640*480*3):
        continue

    #cap = cv2.VideoCapture('track.avi')
    
    
    window = 20
    kp = params[1]
    ki = params[2]
    kd = params[3]

    mm = filter.MovingAverage(window)
    error_init = 0
    ki_l = 0
    time = 0.03
    rospy.sleep(time)

    while not rospy.is_shutdown():
	#ret, image = cap.read()

        hough_drive()
        left_cnd,right_cnd=sliding_drive(image)
        #print(1)
        mm.add_sample(error)
        err = mm.get_wmm()
        ki_l += err*time 

        angle = err * kp + ki * ki_l + ((err - error_init)/time) * kd
        error_init = err
        steer_angle = -angle * 0.4
        if left_cnd ==0 or right_cnd==0:
            drive(steer_angle,30)
        else:
            drive(steer_angle,params[0])
        #print(steer_angle)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #plot_moving_average()

if __name__ == '__main__':
    	
    params = [50, 0.5, 0.0001, 0.00005] # speed, p i d
    cann = [60,100] # low, high
    angle_list = []
    angle_pid = []
    angle_p = []    
    time_list = []
    error=0
    
    start()
