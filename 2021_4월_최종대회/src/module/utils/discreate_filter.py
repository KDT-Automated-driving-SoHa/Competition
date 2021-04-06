#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, math
import scipy.signal as sig
import numpy as np

class DiscreateFilter():
    def __init__(self, f_cut, freq, num_of_signal=1): 
        """
        f_cut: 차단 주파수
        freq: 샘플링 주파수
        """
        self.f_cut = f_cut

        # 데이터 초기화
        self.prev_lpf = np.array([ 0 for _ in range(num_of_signal) ])

        # dt: 함수를 모사하기 위한, 차분 샘플링 주기
        self.dt = 1.0 / freq
        self.tau = 1 / (2*np.pi*f_cut)


    def get_lpf(self, signal):
        self.prev_lpf = (self.tau*self.prev_lpf + self.dt*signal)/(self.tau + self.dt)
        return self.prev_lpf
