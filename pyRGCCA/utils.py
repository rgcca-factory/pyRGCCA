import numpy as np

from numpy.linalg import norm

def l1_norm(f, x):
    return np.sum(np.abs(y=f**2, x=x))

def scalar(f1, f2, grid):
    return np.trapz(f1 * f2, x=grid)

def soft_thresholding(x, l):
    u = [0 if np.abs(x[j]) <= l else np.sign(x[j])*(np.abs(x[j]) - l) for j in range(len(x))]
    return np.array(u)

def binary_search(a, b, f, x):
    m = (a+b)/2
    while abs(f(m) - x) > 1e-8:
        if f(m) > x:
            a = m
            m = (m+b)/2
        else:
            b = m
            m = (a+m)/2
    return m

def proj_l1_l2(a, s):
    if np.linalg.norm(a, ord=1) / np.linalg.norm(a) <= s: return soft_thresholding(a, 0) / norm(soft_thresholding(a, 0))
    a_sorted = np.flip(np.sort(np.abs(a)))
    psi = lambda l : norm(soft_thresholding(a_sorted, l), ord=1) / norm(soft_thresholding(a_sorted, l))
    i = len(a) - 1
    while i >= 0 and psi(a_sorted[i]) > s: i -= 1
    if i == (len(a) - 1):
        up = 0
    else:
        up = a_sorted[i+1]
    lambda_ = binary_search(up, a_sorted[i], psi, s)
    return soft_thresholding(a, lambda_) / norm(soft_thresholding(a, lambda_))
