import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scivis

# %% User settings
T = 20  # Total simulation time

plot_lines = True
# %% Parameters
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

# %% Integrator settings
deltaT = 0.0001
n_int = int(np.ceil(T / deltaT)) + 1
t_int = np.arange(0, T + deltaT, deltaT)  # time vector

# %% Equations of motion
# State vector layout (13 elements):
#   s[0]  = w1,   s[1]  = w1d
#   s[2]  = w2,   s[3]  = w2d
#   s[4]  = w3,   s[5]  = w3d
#   s[6]  = w4,   s[7]  = w4d
#   s[8]  = w5,   s[9]  = w5d
#   s[10] = w6,   s[11] = w6d
#   s[12] = theta (included for completeness; thetad/thetadd are prescribed = 0)

def derivatives(s, t):
    """Return ds/dt for the full state vector at time t."""
    w1_  = s[0];  w1d_  = s[1]
    w2_  = s[2];  w2d_  = s[3]
    w3_  = s[4];  w3d_  = s[5]
    w4_  = s[6];  w4d_  = s[7]
    w5_  = s[8];  w5d_  = s[9]
    w6_  = s[10]; w6d_  = s[11]
    th_  = s[12]

    # Prescribed rotation: constant speed (omega = 0 here), so thetad = thetadd = 0
    thetad_  = 0.0
    thetadd_ = 0.0

    sin_th = np.sin(th_)
    cos_th = np.cos(th_)

    w1dd_ = 48 * E * I1 * (w2_ * l1 ** 3 - w1_ * l2 ** 3) / l1 ** 3 / l2 ** 3 / m1

    w2dd_ = -48 * E * (l1 ** 3 * l3 ** 3 * (m1 + m2) * w2_ - l2 ** 3 * (l1 ** 3 * w3_ * m1 + 2 * l3 ** 3 * w1_ * m2) / 2) * I1 / m2 / l2 ** 3 / l3 ** 3 / l1 ** 3 / m1

    w3dd_ = -24 * E * (l2 ** 3 * l4 ** 3 * (m2 + m3) * w3_ - l3 ** 3 * (l2 ** 3 * w4_ * m2 + 2 * l4 ** 3 * w2_ * m3)) * I1 / m3 / l3 ** 3 / l4 ** 3 / l2 ** 3 / m2

    w4dd_ = (m3 * sin_th * l3 ** 3 * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5_ - m6 * w6_) * thetadd_ - m3 * sin_th * l3 ** 3 * l4 ** 3 * l5 ** 3 * l6 ** 3 * (l5 * m5 + l6 * m6) * thetad_ ** 2 + 2 * m3 * sin_th * l3 ** 3 * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad_ + 24 * E * I1 * l5 ** 3 * l6 ** 3 * (-w4_ * l3 ** 3 + w3_ * l4 ** 3) * (m5 - m6) * sin_th ** 2 + m3 * g * cos_th * l3 ** 3 * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 - m6) * sin_th - 24 * (-m3 * I2 * l3 ** 3 * l4 ** 3 * (w6_ * l5 ** 3 + w5_ * l6 ** 3) * cos_th / 8 + l5 ** 3 * I1 * (l3 ** 3 * (m3 + m4) * w4_ - m4 * w3_ * l4 ** 3) * l6 ** 3) * E) / l6 ** 3 / m3 / l4 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l5 ** 3 / l3 ** 3

    w5dd_ = (-m5 * l5 ** 3 * (-l5 * (m5 - m6) * sin_th ** 2 + sin_th * (m5 * w5_ - m6 * w6_) * cos_th - l5 * m4) * l6 ** 3 * l4 ** 3 * thetadd_ + m5 * l5 ** 3 * (w5_ * (m5 - m6) * sin_th ** 2 + sin_th * (l5 * m5 + l6 * m6) * cos_th + w5_ * m4) * l6 ** 3 * l4 ** 3 * thetad_ ** 2 - 2 * m5 * cos_th * sin_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad_ + 3 * E * I2 * l4 ** 3 * w5_ * sin_th ** 2 * l6 ** 3 * m6 - m5 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m4 + m5 - m6) * sin_th + 24 * (-m5 * w6_ * cos_th ** 2 * I2 * l4 ** 3 * l5 ** 3 / 8 + m5 * w4_ * cos_th * I1 * l5 ** 3 * l6 ** 3 - w5_ * I2 * l4 ** 3 * l6 ** 3 * (m5 + m4) / 8) * E) / m5 / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3

    w6dd_ = (-m6 * l5 ** 3 * l6 ** 3 * (l6 * (m5 - m6) * sin_th ** 2 + sin_th * (m5 * w5_ - m6 * w6_) * cos_th + l6 * m4) * l4 ** 3 * thetadd_ + m6 * l5 ** 3 * l6 ** 3 * (w6_ * (m5 - m6) * sin_th ** 2 + sin_th * (l5 * m5 + l6 * m6) * cos_th + w6_ * m4) * l4 ** 3 * thetad_ ** 2 - 2 * m6 * cos_th * sin_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad_ - 3 * E * I2 * l4 ** 3 * l5 ** 3 * w6_ * sin_th ** 2 * m5 - m6 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m4 + m5 - m6) * sin_th + 24 * E * (-m6 * I2 * l4 ** 3 * (2 * w6_ * l5 ** 3 + w5_ * l6 ** 3) * cos_th ** 2 / 8 + m6 * w4_ * cos_th * I1 * l5 ** 3 * l6 ** 3 + w6_ * I2 * l4 ** 3 * l5 ** 3 * (-m4 + m6) / 8)) / m6 / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3


    return np.array([
        w1d_,  w1dd_,
        w2d_,  w2dd_,
        w3d_,  w3dd_,
        w4d_,  w4dd_,
        w5d_,  w5dd_,
        w6d_,  w6dd_,
        thetad_          # dtheta/dt
        ])


# %% Allocate output arrays
w1   = np.zeros(n_int)
w2   = np.zeros(n_int)
w3   = np.zeros(n_int)
w4   = np.zeros(n_int)
w5   = np.zeros(n_int)
w6   = np.zeros(n_int)

w1d  = np.zeros(n_int)
w2d  = np.zeros(n_int)
w3d  = np.zeros(n_int)
w4d  = np.zeros(n_int)
w5d  = np.zeros(n_int)
w6d  = np.zeros(n_int)

w1dd = np.zeros(n_int)
w2dd = np.zeros(n_int)
w3dd = np.zeros(n_int)
w4dd = np.zeros(n_int)
w5dd = np.zeros(n_int)
w6dd = np.zeros(n_int)

theta  = np.zeros(n_int)
thetad = np.zeros(n_int)
thetadd = np.zeros(n_int)

rIoa = np.zeros((n_int, 3))
rIab = np.zeros((n_int, 3))
rIbc = np.zeros((n_int, 3))
rIcd = np.zeros((n_int, 3))
rB5de = np.zeros((n_int, 3))
rB5df = np.zeros((n_int, 3))

# rIob = np.zeros((n_int, 3))
# rIoc = np.zeros((n_int, 3))
# rIod = np.zeros((n_int, 3))
# rIoe = np.zeros((n_int, 3))
# rIof = np.zeros((n_int, 3))

R_I1x = np.zeros(n_int)
R_I2x = np.zeros(n_int)
R_I3x = np.zeros(n_int)
R_I4x = np.zeros(n_int)
R_B55x = np.zeros(n_int)
R_B56x = np.zeros(n_int)

R_I1y = np.zeros(n_int)
R_I2y = np.zeros(n_int)
R_I3y = np.zeros(n_int)
R_I4y = np.zeros(n_int)
R_B55y = np.zeros(n_int)
R_B56y = np.zeros(n_int)

R_I1 = np.zeros((n_int, 3))
R_I2 = np.zeros((n_int, 3))
R_I3 = np.zeros((n_int, 3))
R_I4 = np.zeros((n_int, 3))
R_B55 = np.zeros((n_int, 3))
R_B56 = np.zeros((n_int, 3))

T45 = np.zeros((n_int, 3, 3))
T54 = np.zeros((n_int, 3, 3))

# %% Initial conditions
theta[0] = np.pi / 2
T45[0] = np.array(
    [[np.cos(theta[0]),  np.sin(theta[0]), 0],
     [-np.sin(theta[0]), np.cos(theta[0]), 0],
     [0, 0, 1]]
)
T54[0] = T45[0].T

# Position vectors
# rIab = np.array([0, l2, 0])
# rIbc = np.array([0, l3, 0])
# rIcd = np.array([0, l4, 0])
# rB5de = np.array([0, l5, 0])
# rB5df = np.array([0, -l6, 0])

# Get other position vectors in inertial reference frame
# rIde = np.dot(T45[0], rB5de)
# rIdf = np.dot(T45[0], rB5df)

# Creating the position vectors describing masses trajectories
# rIoa[0] = np.array([0, l1, 0])
# rIob[0] = rIoa[0] + rIab
# rIoc[0] = rIob[0] + rIbc
# rIod[0] = rIoc[0] + rIcd
# rIoe[0] = rIod[0] + rIde
# rIof[0] = rIoe[0] + rIdf

# Pack initial state: [w1, w1d, w2, w2d, w3, w3d, w4, w4d, w5, w5d, w6, w6d, theta]
state = np.array([
    w1[0], w1d[0],
    w2[0], w2d[0],
    w3[0], w3d[0],
    w4[0], w4d[0],
    w5[0], w5d[0],
    w6[0], w6d[0],
    theta[0]
])

# %% Time loop — RK4 integration
for i in range(1, n_int):
    t = t_int[i - 1]

    k1 = derivatives(state,                  t)
    k2 = derivatives(state + 0.5*deltaT*k1,  t + 0.5*deltaT)
    k3 = derivatives(state + 0.5*deltaT*k2,  t + 0.5*deltaT)
    k4 = derivatives(state +     deltaT*k3,  t +     deltaT)

    state = state + (deltaT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Unpack state
    w1[i]  = state[0];  w1d[i]  = state[1]
    w2[i]  = state[2];  w2d[i]  = state[3]
    w3[i]  = state[4];  w3d[i]  = state[5]
    w4[i]  = state[6];  w4d[i]  = state[7]
    w5[i]  = state[8];  w5d[i]  = state[9]
    w6[i]  = state[10]; w6d[i]  = state[11]
    theta[i] = state[12]

    sin_th = np.sin(theta[i])
    cos_th = np.cos(theta[i])

    # Recover accelerations from the final derivative evaluation (k4 stage)
    # (stored for convenience; same call as k1 at the new state would give true accel)
    d = derivatives(state, t + deltaT)
    w1dd[i]  = d[1];  w2dd[i]  = d[3]
    w3dd[i]  = d[5];  w4dd[i]  = d[7]
    w5dd[i]  = d[9];  w6dd[i]  = d[11]

    # Update transformation matrix
    T45[i] = np.array(
        [[cos_th,  sin_th, 0],
         [-sin_th, cos_th, 0],
         [0, 0, 1]]
    )
    T54[i] = T45[i].T

# =============================================================================
#     # Reaction forces
#     R_I1x[i] = 48 * E * I1 * w1[i] / l1**3
#     R_I2x[i] = 48 * E * I1 * w2[i] / l2**3
#     R_I3x[i] = 24 * E * I1 * w3[i] / l3**3
#     R_I4x[i] = 24 * E * I1 * w4[i] / l4**3
#     R_B55x[i] = 3 * E * I2 * w5[i] / l5**3
#     R_B56x[i] = -m6 * g * sin_th - 3 * E * I2 * w6[i] / l6**3
#
#     R_I1y[i] = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5[i] - m6 * w6[i]) * thetadd[i] - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad[i] ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad[i] + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 - m6) * (m2 + m3 + m1) * sin_th ** 2 - 3 * (-8 * w4[i] * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th + I2 * l4 ** 3 * (w6[i] * l5 ** 3 + w5[i] * l6 ** 3) * (m4 + m5 - m6)) * E * sin_th + m4 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m2 + m3 + m4 + m5 - m6 + m1)) / (sin_th ** 2 * (m5 - m6) + m4) / l5 ** 3 / l6 ** 3 / l4 ** 3
#     R_I2y[i] = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5[i] - m6 * w6[i]) * thetadd[i] - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad[i] ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad[i] + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 - m6) * (m2 + m3) * sin_th ** 2 - 3 * (-8 * w4[i] * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th + I2 * l4 ** 3 * (w6[i] * l5 ** 3 + w5[i] * l6 ** 3) * (m4 + m5 - m6)) * E * sin_th + m4 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m3 + m4 + m5 - m6 + m2)) / (sin_th ** 2 * (m5 - m6) + m4) / l5 ** 3 / l6 ** 3 / l4 ** 3
#     R_I3y[i] = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5[i] - m6 * w6[i]) * thetadd[i] - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad[i] ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad[i] + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * m3 * (m5 - m6) * sin_th ** 2 - 3 * (-8 * w4[i] * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th + I2 * l4 ** 3 * (w6[i] * l5 ** 3 + w5[i] * l6 ** 3) * (m4 + m5 - m6)) * E * sin_th + m4 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m3 + m4 + m5 - m6)) / (sin_th ** 2 * (m5 - m6) + m4) / l5 ** 3 / l6 ** 3 / l4 ** 3
#     R_I4y[i] = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5[i] - m6 * w6[i]) * thetadd[i] - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad[i] ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad[i] + 24 * (w4[i] * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th - I2 * l4 ** 3 * (w6[i] * l5 ** 3 + w5[i] * l6 ** 3) * (m4 + m5 - m6) / 8) * E * sin_th + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * m4 * (m4 + m5 - m6)) / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3
#     R_B55y[i] = 24 * m5 * (-l5 ** 3 * l6 ** 3 * (m6 * (w5[i] - w6[i]) * sin_th ** 2 - w5[i] * m4) * l4 ** 3 * thetadd[i] / 24 + (m6 * (l5 + l6) * sin_th ** 2 - l5 * m4) * l5 ** 3 * l6 ** 3 * l4 ** 3 * thetad[i] ** 2 / 24 - ((sin_th ** 2 * m6 - m4) * w5d[i-1] - m6 * sin_th ** 2 * w6d[i-1]) * l5 ** 3 * l6 ** 3 * l4 ** 3 * thetad[i] / 12 + E * (-I2 * l4 ** 3 * (w6[i] * l5 ** 3 + w5[i] * l6 ** 3) * cos_th / 8 + I1 * w4[i] * l5 ** 3 * l6 ** 3) * sin_th + l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * g * m4 / 24) / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3
#     R_B56y[i] = 24 * m6 * (-(m5 * (w5[i] - w6[i]) * sin_th ** 2 - w6[i] * m4) * l5 ** 3 * l6 ** 3 * l4 ** 3 * thetadd[i] / 24 + l5 ** 3 * (m5 * (l5 + l6) * sin_th ** 2 + l6 * m4) * l6 ** 3 * l4 ** 3 * thetad[i] ** 2 / 24 - l5 ** 3 * ((-sin_th ** 2 * m5 - m4) * w6d[i-1] + m5 * sin_th ** 2 * w5d[i-1]) * l6 ** 3 * l4 ** 3 * thetad[i] / 12 + E * (-I2 * l4 ** 3 * (w6[i] * l5 ** 3 + w5[i] * l6 ** 3) * cos_th / 8 + I1 * w4[i] * l5 ** 3 * l6 ** 3) * sin_th + l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * g * m4 / 24) / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3
#
#     R_I1[i, 0] = R_I2x[i] - R_I1x[i]
#     R_I2[i, 0] = R_I3x[i] - R_I2x[i]
#     R_I3[i, 0] = R_I4x[i] - R_I3x[i]
#     R_I4[i, 0] = R_B55x[i] * cos_th + R_B56x[i] * cos_th - R_I4x[i] + R_B55y[i] * sin_th - R_B56y[i] * sin_th
#     R_B55[i, 0] = -sin_th * m5 * g - R_B55x[i]
#     R_B56[i, 0] = -sin_th * m6 * g - R_B56x[i]
#
#     R_I1[i, 1] = -m1 * g + R_I1y[i] - R_I2y[i]
#     R_I2[i, 1] = -m2 * g + R_I2y[i] - R_I3y[i]
#     R_I3[i, 1] = -m3 * g + R_I3y[i] - R_I4y[i]
#     R_I4[i, 1] = -m4 * g + R_I4y[i] + 3 * sin_th * E * I2 / l5 ** 3 * w5[i] - cos_th * R_B55y[i] + 3 * sin_th * E * I2 / l6 ** 3 * w6[i] + cos_th * R_B56y[i]
#     R_B55[i, 1] = -cos_th * m5 * g
#     R_B56[i, 1] = -cos_th * m6 * g
# =============================================================================

# %% Post processing
sin_th = np.sin(theta)
cos_th = np.cos(theta)

# Reaction forces
R_I1x = 48 * E * I1 * w1 / l1**3
R_I2x = 48 * E * I1 * w2 / l2**3
R_I3x = 24 * E * I1 * w3 / l3**3
R_I4x = 24 * E * I1 * w4 / l4**3
R_B55x = 3 * E * I2 * w5 / l5**3
R_B56x = -m6 * g * sin_th - 3 * E * I2 * w6 / l6**3

R_I1y = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5 - m6 * w6) * thetadd - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 - m6) * (m2 + m3 + m1) * sin_th ** 2 - 3 * (-8 * w4 * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th + I2 * l4 ** 3 * (w6 * l5 ** 3 + w5 * l6 ** 3) * (m4 + m5 - m6)) * E * sin_th + m4 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m2 + m3 + m4 + m5 - m6 + m1)) / (sin_th ** 2 * (m5 - m6) + m4) / l5 ** 3 / l6 ** 3 / l4 ** 3
R_I2y = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5 - m6 * w6) * thetadd - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 - m6) * (m2 + m3) * sin_th ** 2 - 3 * (-8 * w4 * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th + I2 * l4 ** 3 * (w6 * l5 ** 3 + w5 * l6 ** 3) * (m4 + m5 - m6)) * E * sin_th + m4 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m3 + m4 + m5 - m6 + m2)) / (sin_th ** 2 * (m5 - m6) + m4) / l5 ** 3 / l6 ** 3 / l4 ** 3
R_I3y = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5 - m6 * w6) * thetadd - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * m3 * (m5 - m6) * sin_th ** 2 - 3 * (-8 * w4 * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th + I2 * l4 ** 3 * (w6 * l5 ** 3 + w5 * l6 ** 3) * (m4 + m5 - m6)) * E * sin_th + m4 * g * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m3 + m4 + m5 - m6)) / (sin_th ** 2 * (m5 - m6) + m4) / l5 ** 3 / l6 ** 3 / l4 ** 3
R_I4y = (m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5 - m6 * w6) * thetadd - l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * m4 * (l5 * m5 + l6 * m6) * thetad ** 2 + 2 * m4 * cos_th * l4 ** 3 * l5 ** 3 * l6 ** 3 * (m5 * w5d[i-1] - m6 * w6d[i-1]) * thetad + 24 * (w4 * I1 * l5 ** 3 * l6 ** 3 * (m5 - m6) * cos_th - I2 * l4 ** 3 * (w6 * l5 ** 3 + w5 * l6 ** 3) * (m4 + m5 - m6) / 8) * E * sin_th + g * l4 ** 3 * l5 ** 3 * l6 ** 3 * m4 * (m4 + m5 - m6)) / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3
R_B55y = 24 * m5 * (-l5 ** 3 * l6 ** 3 * (m6 * (w5 - w6) * sin_th ** 2 - w5 * m4) * l4 ** 3 * thetadd / 24 + (m6 * (l5 + l6) * sin_th ** 2 - l5 * m4) * l5 ** 3 * l6 ** 3 * l4 ** 3 * thetad ** 2 / 24 - ((sin_th ** 2 * m6 - m4) * w5d[i-1] - m6 * sin_th ** 2 * w6d[i-1]) * l5 ** 3 * l6 ** 3 * l4 ** 3 * thetad / 12 + E * (-I2 * l4 ** 3 * (w6 * l5 ** 3 + w5 * l6 ** 3) * cos_th / 8 + I1 * w4 * l5 ** 3 * l6 ** 3) * sin_th + l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * g * m4 / 24) / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3
R_B56y = 24 * m6 * (-(m5 * (w5 - w6) * sin_th ** 2 - w6 * m4) * l5 ** 3 * l6 ** 3 * l4 ** 3 * thetadd / 24 + l5 ** 3 * (m5 * (l5 + l6) * sin_th ** 2 + l6 * m4) * l6 ** 3 * l4 ** 3 * thetad ** 2 / 24 - l5 ** 3 * ((-sin_th ** 2 * m5 - m4) * w6d[i-1] + m5 * sin_th ** 2 * w5d[i-1]) * l6 ** 3 * l4 ** 3 * thetad / 12 + E * (-I2 * l4 ** 3 * (w6 * l5 ** 3 + w5 * l6 ** 3) * cos_th / 8 + I1 * w4 * l5 ** 3 * l6 ** 3) * sin_th + l4 ** 3 * l5 ** 3 * l6 ** 3 * cos_th * g * m4 / 24) / l5 ** 3 / (sin_th ** 2 * (m5 - m6) + m4) / l6 ** 3 / l4 ** 3

R_I1[:, 0] = R_I2x - R_I1x
R_I2[:, 0] = R_I3x - R_I2x
R_I3[:, 0] = R_I4x - R_I3x
R_I4[:, 0] = R_B55x * cos_th + R_B56x * cos_th - R_I4x + R_B55y * sin_th - R_B56y * sin_th
R_B55[:, 0] = -sin_th * m5 * g - R_B55x
R_B56[:, 0] = -sin_th * m6 * g - R_B56x

R_I1[:, 1] = -m1 * g + R_I1y - R_I2y
R_I2[:, 1] = -m2 * g + R_I2y - R_I3y
R_I3[:, 1] = -m3 * g + R_I3y - R_I4y
R_I4[:, 1] = -m4 * g + R_I4y + 3 * sin_th * E * I2 / l5 ** 3 * w5 - cos_th * R_B55y + 3 * sin_th * E * I2 / l6 ** 3 * w6 + cos_th * R_B56y
R_B55[:, 1] = -cos_th * m5 * g
R_B56[:, 1] = -cos_th * m6 * g


# Relative position vectors
rIoa[:, 0] = w1[i]
rIoa[:, 1] = l1

rIab[:, 0] = w2[i]
rIab[:, 1] = l2

rIbc[:, 0] = w3[i]
rIbc[:, 1] = l3

rIcd[:, 0] = w4[i]
rIcd[:, 1] = l4

rB5de[:, 0] = w5[i]
rB5de[:, 1] = l5

rB5df[:, 0] = w6[i]
rB5df[:, 1] = -l6

# creating the position vectors describing masses trajectories
rIob = rIoa + rIab
rIoc = rIoa + rIab + rIbc
rIod = rIoa + rIab + rIbc + rIcd
rIoe = rIoa + rIab + rIbc + rIcd + np.einsum("nij,nj->ni", T45, rB5de)
rIof = rIoa + rIab + rIbc + rIcd + np.einsum("nij,nj->ni", T45, rB5df)

R_I5 = np.einsum("nij,nj->ni", T45, R_B55)
R_I6 = np.einsum("nij,nj->ni", T45, R_B56)

# %% Plots
if plot_lines:
    rcparams = scivis.rcparams._prepare_rcparams()

    with mpl.rc_context(rcparams):
        # Plot beam deflections over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([w1, w2, w3, w4, w5, w6], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(6)], show_legend=False,
            cmap="jet", linestyles = "-"
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()

        # Plot reaction forces in global x-direction over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([R_I1[:, 0], R_I2[:, 0], R_I3[:, 0],
                             R_I4[:, 0], R_I5[:, 0], R_I6[:, 0]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(6)], show_legend=False,
            cmap="plasma", linestyles = "-"
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()

        # Plot reaction forces in global y-direction over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([R_I1[:, 1], R_I2[:, 1], R_I3[:, 1],
                             R_I4[:, 1], R_I5[:, 1], R_I6[:, 1]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(6)], show_legend=False,
            cmap="plasma", linestyles = "-"
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()

        # Plot reaction forces in local y-direction for blade nodes over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([R_B55[:, 0], R_B56[:, 0]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(4, 6)], show_legend=False,
            cmap="plasma", linestyles = "-"
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()
