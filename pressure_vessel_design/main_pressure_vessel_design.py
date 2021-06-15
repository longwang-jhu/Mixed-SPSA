"""
Created on Fri Aug  7 17:25:45 2020
@author: Long Wang
"""

import numpy as np
from objectives.pressure_vessel_design import PressureVesselDesign
from algorithms.mspsa_constraint import MSPSA
from algorithms.random_search_constraint import RandomSearch
from algorithms.stochastic_ruler_constraint import StochasticRuler

str_today = str(np.datetime64('today'))

### normalize result ###
def loss_noisy(theta):
    return loss_obj.get_loss(theta) + np.random.normal(0,100,1)

def loss_true(theta):
    return loss_obj.get_loss(theta)

def loss_ineq_constraint(theta):
    return loss_obj.get_ineq_constraint(theta)

def show_result(solver, file_name):
    loss_Ks = solver.loss_k_all[-1, :]
    loss_K = np.median(loss_Ks) # get median terminal loss
    loss_K_norm_error = (loss_K - loss_star) / (loss_0 - loss_star)
    
    idx = np.where(loss_Ks == loss_K)[0][0]
    theta_K = solver.theta_k_all[:, -1, idx]

    print("g:", np.around(loss_obj.get_ineq_constraint(theta_K), decimals=4))
    print("theta_K:", theta_K)
    print("loss_K:", loss_K)
    print("loss_K_norm_error:", loss_K_norm_error)

    np.savez("data/" + file_name + "-" + str_today,
              theta_K=theta_K, loss_K=loss_K, loss_K_norm_error=loss_K_norm_error)

p = 4; d = 2
lb = [1, 1, 10, 10]; ub = [99, 99, 200, 200]
loss_obj = PressureVesselDesign()

# 0.8125 / 0.0625 = 13; 0.4375 / 0.0625 = 7
theta_star = np.array([13, 7, 42.098445, 176.636596])
loss_star = loss_true(theta_star)

# 1.125 / 0.0625 = 18; 0.625 / 0.0625 = 10
theta_0 = np.array([18, 10, 50, 150])
loss_0 = loss_true(theta_0)
np.savez("data/pressure-vessel-design",
         theta_star=theta_star, loss_star=loss_star,
         theta_0=theta_0, loss_0=loss_0)

### algorithm parameters ###
meas_num = 20000; rep_num = 25

### MSPSA ###
MSPSA_solver = MSPSA(
    a=0.0005 * np.array([1,1,10,10]),
    c=1, A=100, alpha=0.7, gamma=0.1667,
    iter_num=int(meas_num/2), rep_num=rep_num,
    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
    d=d, lb=lb, ub=ub, loss_ineq_constraint=loss_ineq_constraint, Lagrangian_multiplier=1000,
    seed=1)
MSPSA_solver.train()
show_result(MSPSA_solver, "pressure-vessel-design-MSPSA")

### Random Search ###
RS_solver = RandomSearch(
    sigma=np.sqrt(0.025),
    iter_num=meas_num, rep_num=rep_num,
    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
    d=d, lb=lb, ub=ub, loss_ineq_constraint=loss_ineq_constraint,
    seed=1)
RS_solver.train()
show_result(RS_solver, "pressure-vessel-design-RS")

### Stochastic Ruler ###
M_multiplier = 1
SR_meas_num = 0
SR_iter_num = 0
SR_iter_seq = []
while SR_meas_num < meas_num:
    M_k = int(np.ceil(M_multiplier * np.log(SR_iter_num + 2)))
    SR_meas_num += M_k
    SR_iter_seq.append(M_k)
    SR_iter_num += 1

SR_solver = StochasticRuler(
    M_multiplier=M_multiplier,
    iter_num=SR_iter_num, rep_num=rep_num,
    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
    d=d, lb=lb, ub=ub, loss_ineq_constraint=loss_ineq_constraint,
    seed=1)
SR_solver.train()
show_result(SR_solver, "pressure-vessel-design-SR")