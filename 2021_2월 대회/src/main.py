#!/usr/bin/env python

import cv2, time
from pyzbar import pyzbar
from ros_module import *
from dqn_drive import *
from parking2 import *
from ultrasonic_drive import *
from yolo_drive import *
from go_backward import *

rm = ROS("team_name")

angle = 0
speed = 20
mode = ""
ar_qr_check = False
check_turn=False
start = 0


obj = {
    "DQNQR":DQN(),
    "PARKQR":PARK(),
    "ULTRASONICQR":ULTRASONIC(),
    "YOLOQR":YOLO(),
    "BACKWARDQR":BACKWARD()
}


while rm.get_shutdown():
    #print(rm.get_ultrasonic_data())
    camera_image = rm.get_camera_image_data()
    
    #cv2.imshow('frame', camera_image) 
    if not camera_image.size == (640*480*3):
        continue
       
    qrcodes = pyzbar.decode(camera_image.copy())
    #print(qrcodes)   
    for qtcode in qrcodes:
        qr_string = qtcode.data
        print(qr_string)
        #mode = qr_string

        ##### QR mode #######
        if qr_string=="1 algorithm drive_avoid_obstacle":
            mode="ULTRASONICQR"

        elif qr_string =="2 algorithm drive_turn_back":
            mode="BACKWARDQR"

        elif qr_string =="dqn dqn_drive_start" or qr_string=="3 dqn dqn_drive_start":
            mode="DQNQR"
            #mode="ULTRASONICQR"
        elif qr_string =="5 yolo bottle" or qr_string=="4 dqn dqn_drive_end":
            mode="YOLOQR"
            #mode="ULTRASONICQR"


        elif qr_string =="9 ar parking" or qr_string == "7 algorithm drive_to_parking_lot_1":
            mode="PARKQR"

    print("mode : "+str(mode))

    ##### Modes #####

    if mode== "":
        if start==0:
            mode="ULTRASONICQR"
            start=1
        else:
            rm.set_motor(0,0)
        
        continue

    #obj[mode].set_motor(angle, speed)
    


    
    if mode == "ULTRASONICQR":
        obj_ult = obj[mode]
        obj_ult.set_data([rm.get_ultrasonic_data()])
        
        angle,speed = obj_ult.get_motor()
        if obj_ult.flag_back:
            for _ in range(30):
                angle, speed = obj_ult.get_motor()
                rm.set_motor(angle,speed)
                time.sleep(0.1)
            obj_ult.flag_back = False
            obj_ult.flag = 0
        else:
            rm.set_motor(angle, speed)
            #print(angle, speed)

    if mode == "BACKWARDQR":
        for j in range(2):

            for i in range(35):
                ult_data = rm.get_ultrasonic_data()
                print("back")
                if ult_data['BR'] < 30 or ult_data['BM']<30 or ult_data['BL']<30:
                    continue
                obj[mode].set_data(False)

                angle, speed = obj[mode].get_motor()
                rm.set_motor(angle, speed)
                time.sleep(0.1)

            for i in range(35):
                print("front")
                
                ult_data = rm.get_ultrasonic_data()
                if ult_data['FR'] < 30 or ult_data['FM']<30 or ult_data['FL']<30: 
                    continue
                obj[mode].set_data(True)
                angle, speed = obj[mode].get_motor()
                rm.set_motor(angle, speed)
                time.sleep(0.1)
        
        #mode=="dqn dqn_drive_start"
        mode="ULTRASONICQR"
        rm.set_motor(0,20)
    
    if mode == "DQNQR":
        obj[mode].set_data([
            rm.ultrasonic_data["L"],
            rm.ultrasonic_data["FL"],
            rm.ultrasonic_data["FM"],
            rm.ultrasonic_data["FR"],
            rm.ultrasonic_data["R"],
            rm.ultrasonic_data["BR"],
            rm.ultrasonic_data["BM"],
            rm.ultrasonic_data["BL"]])
        angle, speed = obj[mode].get_motor()
        rm.set_motor(angle,speed)
    
    if mode == "YOLOQR":
        label = "bottle"
        obj["YOLOQR"].set_data([label, rm.bounding_boxes, camera_image])

        angle, speed = obj[mode].get_motor()
        rm.set_motor(angle, speed)
        time.sleep(0.05)    

    

    if mode == "PARKQR":
        obj_ar = obj[mode]
        
        ar_data = rm.get_ar_tags_datas()
        if not ar_qr_check:
            for _ in range(15):
                rm.set_motor(50,-20)
                time.sleep(0.1)
            #for _ in range(5):
            #    rm.set_motor(0,0)
            '''    
            for _ in range(30):
                rm.set_motor(-30,20)
                time.sleep(0.1)
            '''
            ar_qr_check = True
            
        obj_ar.set_data(ar_data)
        if obj_ar.is_back:
            cycle = obj_ar.cycle
            angle = obj_ar.angle
            for _ in range(cycle):
                #rm.set_motor(obj_ar.angle,obj_ar.speed)
               
                rm.set_motor(-angle,-20)
                time.sleep(0.1)
            for _ in range(cycle//3):
                #rm.set_motor(-obj_ar.angle,obj_ar.speed)
                rm.set_motor(angle,-20)
                time.sleep(0.1)
        else:
            rm.set_motor(obj_ar.angle,obj_ar.speed)
            time.sleep(0.1)
        #print(obj_ar.is_back, obj_ar.arData["DZ"],obj_ar.pitch)
    #angle, speed = obj[mode].get_motor() 
    #rm.set_motor(angle, speed)
    
    
    
