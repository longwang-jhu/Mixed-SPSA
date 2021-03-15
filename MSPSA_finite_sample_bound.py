#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 2021-03-14
@author: Long Wang
"""
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from objectives.skewed_quartic import SkewedQuartic
from algorithms.mspsa import MSPSA

def loss_noisy(theta):
    return loss_obj.loss(theta) + np.random.normal(0,1) # normal(mu, sigma)

def loss_true(theta):
    return loss_obj.loss(theta)

def get_theta_error(theta_ks, theta_0, theta_star):
    theta_error = np.linalg.norm(theta_ks - theta_star[:,None,None], axis=0)
    theta_error = np.mean(theta_error, axis=1)
    theta_error = np.concatenate(([np.linalg.norm(theta_0 - theta_star)], theta_error))
    return theta_error

today = date.today()

d = 5; p = 10
loss_obj = SkewedQuartic(p)
theta_star = np.zeros(p)
loss_star = loss_true(theta_star)

### algorithm parameters ###
theta_0 = np.ones(p) * 1
loss_0 = loss_true(theta_0)
a = 0.1; c = 0.5; A = 100;
alpha = 0.7; gamma = 0.167
iter_num = 5000; rep_num = 20

MSPSA_solver = MSPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                     iter_num=iter_num, rep_num=rep_num,
                     d=d, theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                     record_theta_flag=True, record_loss_flag=False)
MSPSA_solver.train()
MSPSA_theta_error = get_theta_error(MSPSA_solver.theta_ks, theta_0, theta_star) ** 2

### finite sample bound ###
kappa_0 = 1; kappa_1 = 1; kappa_2 = 1
sigma2_epsilon = 2
lamb = 1.1
B_H = 0.8; B_T = 0.8
sigma2_L = 15

P_0 = 1
P_1 = np.exp((2*lamb-1) * a / (1-alpha) * ((1+A) ** (1-alpha) - (1+1+A) ** (1-alpha)))

a_0 = a / (1+A) ** alpha
a_1 = a / (1+A+1) ** alpha
c_0 = c
c_1 = c / (1+1) ** gamma

U_norm2 = (B_H * (p-d)**2 * kappa_0**2)**2 * d + (1/6*B_T*(
    ((p-d)**3 - (p-d-1)**3)*kappa_0**2 + (p-d-1)**3*kappa_0**3*kappa_1))**2 * (p-d)

MSPSA_theta_error_bound = np.zeros(iter_num)
theta_0_error = np.linalg.norm(theta_0 - theta_star)
MSPSA_theta_error_bound_k = theta_0_error ** 2

for iter_idx in range(iter_num):
    a_k = a / (1 + A + iter_idx) ** alpha
    c_k = c / (1 + iter_idx) ** gamma

    MSPSA_theta_error_bound_k = (1-(2*lamb-1)*a_k) * MSPSA_theta_error_bound_k \
            + U_norm2 * a_k * c_k**4 + (sigma2_L+sigma2_epsilon) * a_k**2 * (d + (p-d)) * kappa_2 / (4 * c_k**2)
    MSPSA_theta_error_bound[iter_idx] = MSPSA_theta_error_bound_k

MSPSA_theta_error_bound = np.concatenate(([theta_0_error**2], MSPSA_theta_error_bound))

### plot ###
matplotlib.rcParams.update({"font.size": 12})
linewidth = 2

plot_theta = plt.figure()
plt.grid()
# plt.yscale('log')
plt.title(r'Mean-Squared Error for $\hat{\mathbf{\theta}}_k$')
plt.xlabel("Number of Iterations")
plt.ylabel("Mean Squared Error")

plt.plot(MSPSA_theta_error, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_theta_error_bound, linewidth=linewidth, linestyle="--", color="black")

plt.legend(["MSPSA", "Finite-Sample Upper Bound"])
plt.show()
plt.close()
plot_theta.savefig("figures/skewed-quartic-MSPSA-bound-" + str(today) + ".pdf", bbox_inches='tight')