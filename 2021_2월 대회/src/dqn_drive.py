#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import *
from dqn import *
from dqn import *
from filter import *


def rint(point):
    return np.rint(point).astype(np.int32)


class DQN:
    angle = 0
    speed = 0

    
    def __init__(self):
        self.skip_frame = 2
        self.stack_frame = 10
        # self.skip_frame = 1
        # self.stack_frame = 5

        self.model = NN(8, self.stack_frame, 6)
        self.target_model = NN(8, self.stack_frame, 6)
        self.agent = DQNAgent(self.model, self.target_model, skip_frame=self.skip_frame, stack_frame=self.stack_frame)
        
        self.agent.model_load(420)

        self.DRIVE = 1      # 차량 주행 기어
        self.REVERSE = 2    # 차량 후진 기어
        self.BREAK = 3      # 차량 정지 기어

        self.filter = Filter(f_cut=5000, freq=20000, num_data=8)


    def set_motor(self, angle, speed):
        self.angle = angle
        self.speed = speed


    def set_data(self, data):
        if len(self.agent.observation_set) == 0:
            self.agent.reset(data)
        new_data = rint(np.clip(self.filter.LPF(np.array(data)), 0, 200))

        state = self.agent.skip_stack_frame(new_data)
        action = self.agent.get_action(state)

        # 조향각 조정
        if action % 3 == 0:
            steering_deg = -50
        elif action % 3 == 1:
            steering_deg = 0
        elif action % 3 == 2:
            steering_deg = 50
        
        # 기어 조정
        if action // 3 == 0:
            gear = self.DRIVE
            speed = 20
            print("DRIVE", steering_deg, data, new_data)
        elif action // 3 == 1:
            gear = self.REVERSE
            speed = -20
            print("REVERSE", steering_deg, data, new_data)

        self.set_motor(steering_deg, speed)


    def get_motor(self):
        return self.angle, self.speed

