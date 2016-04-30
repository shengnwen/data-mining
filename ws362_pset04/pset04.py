""" Problem Set 04 starter code

Please make sure your code runs on Python version 3.5.0

Due date: 2016-03-04 13:00
"""

import numpy as np
import scipy.optimize

def dualFun(alphas, X, y):
    res = -sum(alphas)
    alphas = np.array([alphas])
    y = np.array([y])
    n = X.shape[0]
    k = 0
    res += 1.0 / 2.0 * np.dot(np.dot(alphas, np.dot(np.dot(y.T, y), np.dot(X, X.T))), alphas.T)
    return res

def my_dual_svm(X, y, C=1):
    bnd = ((0, C),) * X.shape[0]
    alphas = np.array([0] * X.shape[0])
    alphas = scipy.optimize.minimize(dualFun,alphas,args = (X,y),method='L-BFGS-B', options={'disp': True},bounds=bnd).x
    betas = np.zeros((X.shape[1]))
    for i in range(X.shape[0]):
        betas += alphas[i] * y[i] * X[i]
    return betas


def my_primal_svm(X, y, lam=1, k=5, T=100):
    n = X.shape[1]
    num = X.shape[0]
    w1 = np.random.normal(0, 1.0, n)
    w1 = w1 * np.linalg.norm(w1) / np.sqrt(lam)
    w_t = w1
    for t in range(1, T + 1):
        A_t_idx = np.random.choice(range(0, num), k)
        A_t_plus_idx = A_t_idx[(np.dot(w_t, X[A_t_idx].T) < 1)]
        eta_t = 1.0 / (t * lam)
        w_t_half = (1 - eta_t * lam) * w_t + eta_t / k * np.sum((np.dot(X[A_t_plus_idx].T, y[A_t_plus_idx])))
        w_t_half_norm = np.linalg.norm(w_t_half)
        tmp = None
        if w_t_half_norm == 0.0:
            tmp = 1
        else:
            tmp = 1.0/np.sqrt(lam)/np.linalg.norm(w_t_half)
        # print("2:",np.linalg.norm(w_t_half))
        w_t_plus_1 = min(1, tmp) * w_t_half
        w_t = w_t_plus_1
    # print("here",w_t)
    return w_t