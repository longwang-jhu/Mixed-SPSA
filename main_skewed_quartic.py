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
from algorithms.random_search import RandomSearch
from algorithms.stochastic_ruler import StochasticRuler

def loss_noisy(theta):
    return loss_obj.loss(theta) + np.random.normal(0,5) # normal(mu, sigma)

def loss_true(theta):
    return loss_obj.loss(theta)

def get_norm_loss_error(loss_ks, loss_0, loss_star, multi=1):
    loss_error = (np.mean(loss_ks, axis=1) - loss_star) / (loss_0 - loss_star)
    loss_error = np.concatenate(([1], np.repeat(loss_error, multi)))
    return loss_error

def get_norm_theta_error(theta_ks, theta_0, theta_star, multi=1):
    theta_error = np.linalg.norm(theta_ks - theta_star[:,None,None], axis=0)
    theta_error = np.mean(theta_error, axis=1) / np.linalg.norm(theta_0 - theta_star)
    theta_error = np.concatenate(([1], np.repeat(theta_error, multi)))
    return theta_error

today = date.today()

d = 50; p = 100
loss_obj = SkewedQuartic(p)
theta_star = np.zeros(p)
loss_star = loss_true(theta_star)

### algorithm parameters ###
theta_0 = np.ones(p) * 5
loss_0 = loss_true(theta_0)
meas_num = 5000
rep_num = 20

### MSPSA ###
print('Running MSPSA')
MSPSA_solver = MSPSA(a=0.1, c=0.5, A=500, alpha=0.7, gamma=0.167,
                     iter_num=int(meas_num/2), rep_num=rep_num,
                     d=d, theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                     record_theta_flag=True, record_loss_flag=True)
MSPSA_solver.train()
MSPSA_loss_error = get_norm_loss_error(MSPSA_solver.loss_ks, loss_0, loss_star, multi=2)
MSPSA_theta_error = get_norm_theta_error(MSPSA_solver.theta_ks, theta_0, theta_star, multi=2)
# plt.plot(MSPSA_loss_error)

### Random Search ###
print('Running Random Search')
RS_solver = RandomSearch(sigma=0.1,
                         iter_num=meas_num, rep_num=rep_num,
                         d=d, theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                         record_theta_flag=True, record_loss_flag=True)
RS_solver.train()
RS_loss_error = get_norm_loss_error(RS_solver.loss_ks, loss_0, loss_star, multi=1)
RS_theta_error = get_norm_theta_error(RS_solver.theta_ks, theta_0, theta_star, multi=1)
# plt.plot(RS_loss_error)

### Stochastic Ruler ###
print('Running Stochastic Ruler')
M_multiplier = 0.5
SR_solver = StochasticRuler(M_multi=0.5, meas_num=meas_num, rep_num=rep_num,
                            d=d, theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                            record_theta_flag=True, record_loss_flag=True)
SR_solver.train()
SR_M_ks = SR_solver.M_ks
SR_loss_error = get_norm_loss_error(SR_solver.loss_ks, loss_0, loss_star, multi=SR_M_ks)[:meas_num+1]
SR_theta_error = get_norm_theta_error(SR_solver.theta_ks, theta_0, theta_star, multi=SR_M_ks)[:meas_num+1]
plt.plot(SR_loss_error)

### Plot ###
matplotlib.rcParams.update({"font.size": 12})
linewidth = 2

# plot theta
plot_theta = plt.figure()
plt.grid()
plt.title(r'Normalized Mean-Squared Error for $\hat{\mathbf{\theta}}_k$')
plt.xlabel("Number of Loss Function Measurements")
plt.ylabel("Normalized Mean-Squared Error")
plt.ylim(0, 1)

plt.plot(RS_theta_error**2, linewidth=linewidth, linestyle=":", color="black")
plt.plot(SR_theta_error**2, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_theta_error**2, linewidth=linewidth, linestyle="--", color="black")

plt.legend(["Local Random Search", "Stochastic Ruler", "MSPSA"])
plt.close()
plot_theta.savefig("figures/skewed-quartic-theta-error-" + str(today) + ".pdf", bbox_inches='tight')

# plot loss
plot_loss = plt.figure()
plt.grid()
plt.title("Normalized Error for Loss")
plt.xlabel("Number of Loss Function Measurements")
plt.ylabel("Normalized Error")
plt.ylim(0, 1)

plt.plot(RS_loss_error, linewidth=linewidth, linestyle=":", color="black")
plt.plot(SR_loss_error, linewidth=linewidth, linestyle="-", color="black")
plt.plot(MSPSA_loss_error, linewidth=linewidth, linestyle="--", color="black")

plt.legend(["Local Random Search", "Stochastic Ruler", "MSPSA"])
plt.close()
plot_loss.savefig("figures/skewed-quartic-loss-error-" + str(today) + ".pdf", bbox_inches='tight')