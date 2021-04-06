#!/usr/bin/env python
#-*- coding: utf-8 -*-

# https://medium.com/@inmoonlight/pytorch%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D%ED%95%98%EA%B8%B0-cnn-62a9326111ae

import torch 
import torch.nn as nn


use_cuda = torch.cuda.is_available()


class End2End(nn.Module):
    
    def __init__(self):
        super(End2End, self).__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
        conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1)) 
        conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)) 
        
        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(),
            conv2, nn.ReLU(),
            conv3, nn.ReLU(),
            conv4, nn.ReLU(),
            conv5, nn.ReLU(),
        )
        
        fc1 = nn.Linear(1152, 100)
        fc2 = nn.Linear(100, 50)
        fc3 = nn.Linear(50, 10)
        fc4 = nn.Linear(10, 1)

        self.fc_module = nn.Sequential(
            fc1, nn.ReLU(),
            fc2, nn.ReLU(),
            fc3, nn.ReLU(),
            fc4
        )
        

    def forward(self, x):
        x = self.conv_module(x)
        x = torch.flatten(x, start_dim=1)
        y = self.fc_module(x)

        return y
