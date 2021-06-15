#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 01:41:35 2020

@author: Long Wang
"""

import numpy as np

class StochasticRuler:
    def __init__(self, M_multiplier=1,
                 iter_num=100, rep_num=1,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 d=0, lb=None, ub=None, loss_ineq_constraint=None,
                 record_theta_flag=True, record_loss_flag=True,
                 seed=1):

        # M_k: np.ceil(M_multiplier * np.log(iter_idx + 2)) number of measurements to accept a point
        # d: the first d components are integers

        self.seed = seed
        np.random.seed(self.seed)

        self.M_multiplier = M_multiplier

        self.iter_num = iter_num
        self.rep_num = rep_num

        self.theta_0 = theta_0
        self.p = theta_0.shape[0] # shape = (p,)
        self.loss_0 = loss_true(theta_0)

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        # constraint
        self.d = d
        self.lb = lb if lb != None else -np.ones(self.p) * np.Inf
        self.ub = ub if ub != None else np.ones(self.p) * np.Inf
        self.loss_ineq_constraint = loss_ineq_constraint

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

    def project(self, theta):
        proj_theta = theta.copy()
        proj_theta = np.clip(proj_theta, self.lb, self.ub)
        # project the first d components to the nearest integer
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        return proj_theta

    def train(self):
        if self.record_theta_flag:
            self.theta_k_all = np.empty((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_k_all = np.empty((self.iter_num, self.rep_num))

        for rep_idx in range(self.rep_num):
            print("algo: Stochastic Ruler; rep_idx:", rep_idx+1, "/", self.rep_num)
            theta = self.theta_0

            for iter_idx in range(self.iter_num):
                if self.loss_ineq_constraint == None:
                    theta_new = np.random.rand(self.p)
                    for i in range(self.p):
                        theta_new[i] = theta_new[i] * (self.ub[i] - self.lb[i]) + self.lb[i]
                    theta_new = self.project(theta_new)
                else:
                    constraint_flag = False
                    while constraint_flag == False:
                        theta_new = np.random.rand(self.p)
                        for i in range(self.p):
                            theta_new[i] = theta_new[i] * (self.ub[i] - self.lb[i]) + self.lb[i]
                        theta_new = self.project(theta_new)
                        ineq_constraint_value = self.loss_ineq_constraint(theta_new)
                        constraint_flag = (sum(np.array(ineq_constraint_value) <= 0) == len(ineq_constraint_value))

                M_k = int(np.ceil(self.M_multiplier * np.log(iter_idx + 2)))
                accept_flag = True
                for i in range(M_k):
                    loss_new = self.loss_noisy(theta_new)
                    if loss_new > np.random.rand(1) * (self.loss_0 - 6000) + 6000:
                        accept_flag = False
                        break

                if accept_flag:
                    theta = theta_new

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = self.project(theta)
                if self.record_loss_flag:
                    self.loss_k_all[iter_idx,rep_idx] = self.loss_true(self.project(theta))