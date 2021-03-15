#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:07:04 2020

@author: Long Wang
"""

import numpy as np

class RandomSearch:
    def __init__(self, sigma=0.1,
                 iter_num=100, rep_num=1,
                 d=0, theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=False, record_loss_flag=False,
                 seed=1):

        # sigma: new candidate point theta + normal(0, simga^2)
        # d: the first d components are integers

        self.seed = seed
        np.random.seed(self.seed)

        self.sigma = sigma

        self.iter_num = iter_num
        self.rep_num = rep_num

        self.d = d
        self.theta_0 = theta_0

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

        self.p = theta_0.shape[0] # shape = (p,)
        if self.record_theta_flag:
            self.theta_ks = np.zeros((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_ks = np.zeros((self.iter_num, self.rep_num))

    def project(self, theta=None):
        # project the first d components to the nearest integer
        proj_theta = theta.copy()
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        return proj_theta

    def get_delta_all(self):
        self.delta_all = np.random.normal(0, self.sigma, (self.p, self.iter_num, self.rep_num))

    def train(self):
        self.get_delta_all()
        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx + 1, "/", self.rep_num)
            theta_k = self.theta_0.copy()
            loss_best = np.Inf
            for iter_idx in range(self.iter_num):
                theta_k_new = self.project(theta_k + self.delta_all[:, iter_idx, rep_idx])
                loss_new = self.loss_noisy(theta_k_new)
                if loss_new < loss_best:
                    theta_k = theta_k_new
                    loss_best = loss_new

                # record result
                if self.record_theta_flag:
                    self.theta_ks[:,iter_idx,rep_idx] = self.project(theta_k)
                if self.record_loss_flag:
                    self.loss_ks[iter_idx,rep_idx] = self.loss_true(self.project(theta_k))