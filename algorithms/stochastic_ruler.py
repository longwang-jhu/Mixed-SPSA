#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 01:41:35 2020

@author: longwang
"""

import numpy as np

class StochasticRuler:
    def __init__(self, M_multi=1, meas_num=1, rep_num=1,
                 d=0, theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=False, record_loss_flag=False,
                 seed=1):

        # M_k: np.ceil(M_multiplier * np.log(iter_idx + 2)) number of measurements to accept a point
        # d: the first d components are integers

        self.seed = seed
        np.random.seed(self.seed)

        self.M_multi = M_multi
        self.rep_num = rep_num

        self.d = d
        self.theta_0 = theta_0

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

        self.meas_num = meas_num
        M_total = 0
        M_ks = []
        iter_idx = 0
        while M_total < self.meas_num:
            M_k = int(np.ceil(self.M_multi * np.log(iter_idx + 2)))
            M_ks.append(M_k)
            M_total += M_k
            iter_idx += 1
        self.M_ks = M_ks
        self.iter_num = iter_idx

        self.p = theta_0.shape[0] # shape = (p,)
        self.loss_0 = loss_true(theta_0)
        if self.record_theta_flag:
            self.theta_ks = np.zeros((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_ks = np.zeros((self.iter_num, self.rep_num))

    def project(self, theta=None):
        # project the first d components to the nearest integer
        proj_theta = theta.copy()
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        return proj_theta

    def train(self):
        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx + 1, "/", self.rep_num)
            theta_k = self.theta_0.copy()
            for iter_idx in range(self.iter_num):
                theta_k_new = self.project(np.random.rand(self.p) * self.theta_0)
                M_k = self.M_ks[iter_idx]
                accept_flag = True
                for i in range(M_k):
                    loss_new = self.loss_noisy(theta_k_new)
                    if loss_new > self.loss_0 * np.random.rand(1):
                        accept_flag = False
                        break
                if accept_flag:
                    theta_k = theta_k_new

                # record result
                if self.record_theta_flag:
                    self.theta_ks[:,iter_idx,rep_idx] = self.project(theta_k)
                if self.record_loss_flag:
                    self.loss_ks[iter_idx,rep_idx] = self.loss_true(self.project(theta_k))