#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 4 21:32:45 2020

@author: Long Wang
"""

import numpy as np

class SkewedQuartic:
    def __init__(self, p):
        self.p = p
        self.B = np.triu(np.ones((p,p))) / p

    def loss(self, theta):
        L = theta.T.dot(self.B.T.dot(self.B)).dot(theta) \
            + 0.1 * np.sum(self.B.dot(theta) ** 3) \
            + 0.01 * np.sum(self.B.dot(theta) ** 4)
        return float(L)

    def grad(self, theta):
        g = self.B.T.dot(
            2 * self.B.dot(theta)
            + 0.3 * np.sum(self.B.dot(theta) ** 2)
            + 0.04 * np.sum(self.B.dot(theta) ** 3))
        return g

    def Hess(self, theta):
        H = self.B.T.dot(
            np.diag(2 + 0.6 * self.B.dot(theta)
                    + 0.12 * np.sum(self.B.dot(theta) ** 2))).dot(self.B)
        return H

if __name__ == "__main__":
    p = 5
    skewed_quartic = SkewedQuartic(p)
    theta_0 = np.ones(p)
    print(skewed_quartic.loss(theta_0))
    print(skewed_quartic.grad(theta_0))
    print(skewed_quartic.Hess(theta_0))