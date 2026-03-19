from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.optimize import minimize, least_squares

import scivis

# %% Model Parameters
l1 = 0.193
l2 = 0.292
l3 = 0.220
l4 = 0.271
l5 = 0.430
l6 = 0.430

m1 = 2.294
m2 = 1.941
m3 = 1.943
m4 = 2.732
m5 = 0.774
m6 = 0.774
mbeam1 = 0.3
mbeam2 = 0.421
macc = 0.073
g = 9.81

b1 = 0.0291
h1 = 0.0011
b2 = 0.0350
h2 = 0.0015
E = 2e11
I1 = (b1 * h1**3) / 12
I2 = (b2 * h2**3) / 12
k1 = 4 * (12 * E * I1) / l1**3
k2 = 4 * (12 * E * I1) / l2**3
k3 = 2 * (12 * E * I1) / l3**3
k4 = 2 * (12 * E * I1) / l4**3
k5 = (3 * E * I2) / l5**3
k6 = (3 * E * I2) / l6**3
N_DOF = 6

# %% Calculate modal parameters of the mathematical model
def system_matrices(m, k):
    m1, m2, m3, m4, m5, m6 = m
    k1, k2, k3, k4, k5, k6 = k

    M = np.array([[m1, 0, 0, 0, 0, 0],
                   [m2, m2, 0, 0, 0, 0],
                   [m3, m3, m3, 0, 0, 0],
                   [m4, m4, m4, m4, 0, 0],
                   [m5, m5, m5, m5, m5, 0],
                   [m6, m6, m6, m6, 0, m6]])

    K = -np.array([[-k1, k2, 0, 0, 0, 0],
                   [0, -k2, k3, 0, 0, 0],
                   [0, 0, -k3, k4, 0, 0],
                   [0, 0, 0, -k4, k5, k6],
                   [0, 0, 0, 0, -k5, 0],
                   [0, 0, 0, 0, 0, -k6]])

    return M, K

def modal_analysis(M, K):
    n_dof = M.shape[0]

    zero_mat = np.zeros((n_dof, n_dof))
    A= np.block([[M, zero_mat],
                 [zero_mat, M]])
    B= np.block([[zero_mat, K],
                 [-M, zero_mat]])

    eigval, eigvec= eig(-B, A)
    eigfreqs= np.abs(eigval) / (2 * np.pi)  # Eigenfrequencies in [Hz]

    # Sort by eigenfrequency
    idx_sorted = np.argsort(eigfreqs)
    eigfreqs= eigfreqs[idx_sorted]
    eigval= eigval[idx_sorted]
    eigvec= eigvec[:, idx_sorted]

    # Normalize eigenvectors
    idx = N_DOF + np.argmax(np.abs(eigvec[n_dof:]), axis=0)
    scale = eigvec[idx, np.arange(eigvec.shape[1])]
    eigvec_norm = np.round(eigvec/ scale, 4)

    # Extract modal matrices shape matrix
    modeshapes= eigvec_norm[n_dof:, ::2].real

    return eigfreqs[::2], modeshapes

M, K = system_matrices(m=[m1, m2, m3, m4, m5, m6], k=[k1, k2, k3, k4, k5, k6])
eigfreqs, modeshapes = modal_analysis(M, K)

# %% Compare modal parameters to experimental values

eigfreqs_exp = np.array([1.2849, 1.812, 2.264, 4.473, 6.744, 8.591])

# %% Adjust model parameters with least squares fit to the eigenfrequencies
def state_func(x):
    m1, m2, m3, m4, m5, m6, k1, k2, k3, k4, k5, k6 = x

    M, K = system_matrices(m=[m1, m2, m3, m4, m5, m6],
                           k=[k1, k2, k3, k4, k5, k6])

    eigfreqs, modeshapes = modal_analysis(M, K)

    domega = eigfreqs_exp - eigfreqs

    return domega

lbounds = np.array([vi*0.85 for vi in [m1, m2, m3, m4, m5, m6,
                                       k1, k2, k3, k4, k5, k6]])
ubounds = np.array([vi*1.15 for vi in [m1, m2, m3, m4, m5, m6,
                                       k1, k2, k3, k4, k5, k6]])

res = least_squares(fun=state_func,
                    x0=[m1, m2, m3, m4, m5, m6, k1, k2, k3, k4, k5, k6],
                    bounds=(lbounds, ubounds))

m_adjusted = res.x[:6]
k_adjusted = res.x[6:]

M_adjusted, K_adjusted = system_matrices(m=m_adjusted, k=k_adjusted)
eigfreqs_adjusted, modeshapes_adjusted = modal_analysis(M_adjusted, K_adjusted)

domega_adjusted = eigfreqs_exp - eigfreqs_adjusted
