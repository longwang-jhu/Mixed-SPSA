#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:11:27 2020

@author: Long Wang
"""

import numpy as np

class PressureVesselDesign:
    def __init__(self):
        pass

    def get_loss(self, theta):
        x_1 = theta[0] * 0.0625
        x_2 = theta[1] * 0.0625
        x_3 = theta[2]
        x_4 = theta[3]

        L = 0.6224 * x_1 * x_3 * x_4 + 1.7781 * x_2 * x_3**2 \
            + 3.1661 * x_1**2 * x_4 + 19.84 * x_1 **2 *x_3
        return float(L)

    def get_ineq_constraint(self, theta): # g(theta) <= 0
        x_1 = theta[0] * 0.0625
        x_2 = theta[1] * 0.0625
        x_3 = theta[2]
        x_4 = theta[3]

        g_1 = -x_1 + 0.0193 * x_3
        g_2 = -x_2 + 0.00954 * x_3
        g_3 = (-np.pi * x_3**2 * x_4 - 4/3*np.pi * x_3**3 + 1296000) / 12960 # normalization

        return g_1, g_2, g_3

if __name__ == "__main__":
    # 0.8125 / 0.0625 = 13
    # 1.1250 / 0.0625 = 18
    # 0.4375 / 0.0625 = 7

    pressure_vessel_design = PressureVesselDesign()
    # theta = [1.125/0.0625, 0.625/0.0625, 58.291, 43.690]
    # theta = [1.125/0.0625, 0.625/0.0625, 47.700, 117.701]

    # WOA: 6059.7410
    theta = np.array([13, 7, 42.0984455958549, 176.6365958424394])
    theta = np.array([12, 6, 38.79363018, 200])
    theta = np.array([13, 7, 41.5960, 185.9336])

    # theta_0 = [13, 7, 42.0983, 176.6366]
    # theta = [5.,6.,47.54745887, 149.74276268]

    print(pressure_vessel_design.get_loss(theta))
    print(np.around(pressure_vessel_design.get_ineq_constraint(theta), decimals=5))
