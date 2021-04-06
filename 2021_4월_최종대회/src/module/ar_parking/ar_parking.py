# !/usr/bin/env python
#-*- coding: utf-8 -*-

from module.infos.ultrasonic_info import UltrasonicInfo
from module.infos.motor_info import MotorInfo

class ParkingBehavior():
    def __init__(self):
        self.ar_info1 = None
        self.ar_info2 = None

        self.behaviors = [
            self._detect(),
            self._stage_1_1(),
            self._stage_1_2(),
            self._stage_1_3(),
            self._stage_2(),
            self._stage_3()]
        self.behavior_idx = 0


    def get_motor_info(self, ultrasonic_info, ar_info1, ar_info2):
        self.ultrasonic_info = ultrasonic_info
        self.ar_info1 = ar_info1
        self.ar_info2 = ar_info2

        for _, angle, speed in self._play():
            return MotorInfo(angle, speed)


    def _play(self):
        if len(self.behaviors) == self.behavior_idx:
            yield True, 0, 0

        behavior = self.behaviors[self.behavior_idx]
        while True:
            is_done, angle, speed = behavior.next()
            if is_done:
                self.behavior_idx += 1
                break
            yield is_done, angle, speed
        yield True, angle, speed
    
            
    def _stage_1_1(self):
        # 초기 주차 방향 및 자리를 맞춰주는 부분(1단계: 직진)
        # substage1 for going forward for distance1 mm
        print("stage_1_1")
        speed = 20
        distance = 400
        cycle = distance // speed

        for _ in range(cycle-1):
            angle = 0
            speed = 20
            yield False, angle, speed

        # angle = 0
        # speed = 0
        yield True, angle, speed


    def _stage_1_2(self):
        # 초기 주차 방향 및 자리를 맞춰주는 부분(2단계: 왼쪽 꺽기)
        # substage2 for going left for distance2 mm
        print("stage_1_2")
        speed = 20
        distance = 360
        cycle = distance // speed

        for _ in range(cycle-1):
            angle = -50
            speed = 20
            yield False, angle, speed

        # angle = 0
        # speed = 0
        yield True, angle, speed


    def _stage_1_3(self):
        # 초기 주차 방향 및 자리를 맞춰주는 부분(3단계: 후진)
        # substage3 for going backward for distance3 mm
        print("stage_1_3")
        speed = -20
        distance = 600
        cycle = distance // abs(speed)

        for _ in range(cycle-1):
            angle = 5
            speed = -20
            yield False, angle, speed

        # angle = 0
        # speed = 0
        yield True, angle, speed
        

    def _stage_2(self):
        print("stage_2")
        # 초음파 센서를 이용하여 최대 후진(장애물 고려)
        # stage 2 for going backward
        backward_info = UltrasonicInfo()
        value1 = 60
        backward_info.back_middle = value1+8
        backward_info.back_right = value1

        forward_info = UltrasonicInfo()
        value2 = 15+3
        forward_info.back_left = value2
        forward_info.back_middle = value2+10
        forward_info.back_right = value2

        while True:
            print("main ultrasonic", self.ultrasonic_info)
            if self.ultrasonic_info.back_middle < backward_info.back_middle    \
            or self.ultrasonic_info.back_right < backward_info.back_right:
                if self.ultrasonic_info.back_left < forward_info.back_left       \
                or self.ultrasonic_info.back_middle < forward_info.back_middle   \
                or self.ultrasonic_info.back_right < forward_info.back_right:
                    angle = 0
                    speed = 0
                    yield True, angle, speed

                angle = -30
                speed = -17
                yield False, -50, -17
                
            # exception
            elif self.ultrasonic_info.back_left < forward_info.back_left  \
            or self.ultrasonic_info.back_middle < forward_info.back_middle:
                    angle = 0
                    speed = 0
                    yield True, angle, speed
            else:
                angle = 5
                speed = -20
                yield False, angle, speed



    def _stage_3(self):
        print("stage_3")
        # 전방 AR만 이용해서 자리맞추면서 주차
        # stage 3 for heading AR tag
        value_for_ar = 52

        while True:
            if self.ar_info2.dz > value_for_ar:
                print("stage 3 ", self.ar_info2.pitch, self.ar_info2.dx, self.ar_info2.dz)
                angle = self.ar_info2.dx #+ 9
                speed = 17
                yield False, angle, speed
            # exception
            elif self.ar_info2.dz == 0:
                angle = 0
                speed = -10
                yield False, angle, speed
            else:
                print("Parked")
                angle = 0
                speed = 0
                yield True, angle, speed
                

    def _detect(self):
        print("detect")
        # weights for angle
        k1 = 1.0
        k2 = 2.0

        while True:
            if self.ar_info1 or self.ar_info2:
                if self.ar_info1:
                    print("=======================")
                    print("AR1 found")
                    print("pitch : " + str(round(self.ar_info1.pitch, 1)))
                    print(" x : " + str(self.ar_info1.dx))
                    print(" z : " + str(self.ar_info1.dz))
                if self.ar_info2:
                    print("=======================")
                    print("AR2 found")
                    print("pitch : " + str(round(self.ar_info2.pitch, 1)))
                    print(" x : " + str(self.ar_info2.dx))
                    print(" z : " + str(self.ar_info2.dz))

                if self.ar_info1:
                    angle = 0
                    speed = 23
                    yield False, angle, speed
                
                # 처음 2번 AR 감지되고 나서, 초기값 세팅
                if self.ar_info2:
                    # exception
                    if self.ar_info2.dz == 0:
                        angle = 0
                        speed = 23
                        yield False, angle, speed

                    # go to stage 1
                    if 0 < self.ar_info2.dz < 70:
                        print("===Stage1===")
                        angle = 0
                        speed = 0
                        yield True, angle, speed

                    """중요"""
                    # self.angle = k1 * math.atan((max(self.xs) - 50) / max(self.zs)) + self.pitch * k2
                    angle = (self.ar_info2.dx - 65) * k1 + self.ar_info2.pitch * k2  # better working
                    speed = 23
                    yield False, angle - 5, speed

            angle = 0
            speed = 23
            yield False, angle, speed


if __name__ == "__main__":
    parking = ParkingBehavior()
    for is_done, angle, speed in parking._play():
        print (is_done, angle, speed)

    # for is_done, angle, speed in stage1.play():
    #     print(is_done, angle, speed)
