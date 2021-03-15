#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:58:35 2020

@author: Long Wang
"""

import numpy as np

class MSPSA:
    def __init__(self, a=0, c=0.1, A=0, alpha=0.602, gamma=0.101,
                 iter_num=1, rep_num=1, dir_num=1,
                 d=0, theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=False, record_loss_flag=False,
                 seed=1):

        # step size: a_k = a / (k+1+A) ** alpha
        # perturbation size: c_k = c / (k+1) ** gamma
        # dir_num: number of dirions per iteration
        # d: the first d components are integers

        self.seed = seed
        np.random.seed(self.seed)

        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.dir_num = dir_num

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

    def pi(self, theta=None):
        # project the first d components to floor(x) + 0.5
        pi_theta = theta.copy()
        pi_theta[:self.d] = np.floor(pi_theta[:self.d]) + 0.5
        return pi_theta

    def project(self, theta=None):
        proj_theta = theta.copy()
        # project the first d components to the nearest integer
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        return proj_theta

    def get_delta_all(self):
        self.delta_all = np.round(np.random.rand(self.p, self.dir_num, self.iter_num, self.rep_num)) * 2 - 1

    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        C_k = np.concatenate((np.repeat(0.5, self.d), np.repeat(c_k, self.p - self.d)))
        grad_k = np.zeros(self.p)
        for dir_idx in range(self.dir_num):
            delta_k = self.delta_all[:, dir_idx, iter_idx, rep_idx]
            loss_plus = self.loss_noisy(self.pi(theta_k) + C_k * delta_k)
            loss_minus = self.loss_noisy(self.pi(theta_k) - C_k * delta_k)
            grad_k += (loss_plus - loss_minus) / (2 * C_k * delta_k)
        return grad_k / self.dir_num

    def train(self):
        self.get_delta_all()
        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx + 1, "/", self.rep_num)
            theta_k = self.theta_0.copy()
            for iter_idx in range(self.iter_num):
                a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
                g_k = self.get_grad_est(iter_idx, rep_idx, theta_k)
                theta_k -= a_k * g_k

                # record result
                if self.record_theta_flag:
                    self.theta_ks[:,iter_idx,rep_idx] = self.project(theta_k)
                if self.record_loss_flag:
                    self.loss_ks[iter_idx,rep_idx] = self.loss_true(self.project(theta_k))