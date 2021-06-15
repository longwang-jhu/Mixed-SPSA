#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:58:35 2020

@author: Long Wang
"""

import numpy as np

class MSPSA:
    def __init__(self, a=0, c=0.1, A=0, alpha=0.602, gamma=0.101,
                 iter_num=100, rep_num=1, direct_num=1,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 d=0, lb=None, ub=None, loss_ineq_constraint=None, Lagrangian_multiplier=0,
                 record_theta_flag=True, record_loss_flag=True,
                 seed=1):

        # step size: a_k = a / (k+1+A) ** alpha
        # perturbation size: c_k = c / (k+1) ** gamma
        # direct_num: number of directions per iteration
        # d: the first d components are integers
        # lb/ub: lower/upper bound of the variable List[p]

        self.seed = seed
        np.random.seed(self.seed)

        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
        self.p = theta_0.shape[0] # shape = (p,)
        # initialize perturbation vector: uniform{-1, 1}
        self.delta_all = np.round(np.random.rand(self.p, direct_num, iter_num, rep_num)) * 2 - 1

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        # constraint
        self.d = d
        self.lb = lb if lb != None else -np.ones(self.p) * np.Inf
        self.ub = ub if ub != None else np.ones(self.p) * np.Inf
        self.loss_ineq_constraint = loss_ineq_constraint
        self.Lagrangian_multiplier = Lagrangian_multiplier

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

    def pi(self, theta):
        # project the first d components to floor(x) + 0.5
        pi_theta = theta.copy()
        pi_theta[:self.d] = np.floor(pi_theta[:self.d]) + 0.5
        return pi_theta

    def project(self, theta):
        # project the first d components to the nearest integer
        proj_theta = theta.copy()
        proj_theta = np.clip(proj_theta, self.lb, self.ub)
        proj_theta[:self.d] = np.round(proj_theta[:self.d])
        return proj_theta

    def get_grad_est(self, theta, iter_idx=0, rep_idx=0):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        C_k = np.concatenate((np.repeat(0.5, self.d), np.repeat(c_k, self.p-self.d)))

        grad_all = np.empty((self.p, self.direct_num))
        for direct_idx in range(self.direct_num):
            delta = self.delta_all[:, direct_idx, iter_idx, rep_idx]
            loss_plus = self.loss_noisy(self.pi(theta) + C_k * delta)
            loss_minus = self.loss_noisy(self.pi(theta) - C_k * delta)

            if self.loss_ineq_constraint != None:
                ineq_cons_value_plus = self.loss_ineq_constraint(self.pi(theta) + C_k * delta)
                loss_plus += self.Lagrangian_multiplier * np.log(iter_idx+2) * np.sum(np.maximum(ineq_cons_value_plus, 0))
                ineq_cons_value_minus = self.loss_ineq_constraint(self.pi(theta) - C_k * delta)
                loss_minus += self.Lagrangian_multiplier * np.log(iter_idx+2) * np.sum(np.maximum(ineq_cons_value_minus, 0))

            grad_all[:,direct_idx] = (loss_plus - loss_minus) / (2 * C_k * delta)

        grad = np.average(grad_all, axis=1)
        return grad

    def get_new_est(self, theta, grad, iter_idx=0):
        a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
        return theta - a_k * grad

    def train(self):
        if self.record_theta_flag:
            self.theta_k_all = np.empty((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_k_all = np.empty((self.iter_num, self.rep_num))

        for rep_idx in range(self.rep_num):
            print("algo: MSPSA; rep_idx:", rep_idx+1, "/", self.rep_num)
            theta = self.theta_0

            for iter_idx in range(self.iter_num):
                grad = self.get_grad_est(theta, iter_idx, rep_idx)
                theta = self.get_new_est(theta, grad, iter_idx)

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = self.project(theta)
                if self.record_loss_flag:
                    self.loss_k_all[iter_idx,rep_idx] = self.loss_true(self.project(theta))