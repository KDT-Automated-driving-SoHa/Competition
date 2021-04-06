#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import cv2
import torch
import rospkg

from module.exposure.model import End2End


class AutoExposure():

    def __init__(self, episode_count=1700, batch_count=42):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.camera_device = "/dev/videoCAM"
        self.net = self.load_model(episode_count, batch_count)
        self.prev_value = 0


    def load_model(self, episode_count, batch_count):
        script_dir = os.path.dirname(__file__)
        save_dir = os.path.join(script_dir, "save")
        pth_path = os.path.join(save_dir, "main_model_{:06d}_{:06}.pth".format(episode_count, batch_count))

        model = End2End().to(self.device)
        with open(pth_path, "rb") as f:
            LoadBuffer = io.BytesIO(f.read())
            model.load_state_dict(torch.load(LoadBuffer, map_location=self.device))
        return model


    def get_exposure(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame = cv2.resize(frame, dsize=(200, 112))

        frame = frame[46:, :]

        frame = frame.transpose((2, 0, 1)) / 255.0
        t_frame = torch.FloatTensor([frame]).to(self.device)

        value = int(self.net(t_frame).tolist()[0][0])
        return value


    def set_exposure(self, value):
        if type(value) is int and value != self.prev_value:
            command = "v4l2-ctl -d {:s} -c exposure_absolute={:d}".format(self.camera_device, value)
            os.system(command)
            self.prev_value = value


    def change_exposure(self, frame):
        value = self.get_exposure(frame)

        result_ex = 46

        if value < 30:
            result_ex = 50
        if value >= 60:
            result_ex = 35

        self.set_exposure(result_ex)
