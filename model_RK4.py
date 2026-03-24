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

PI1y = m1 * g
PI2y = m2 * g
PI3y = m3 * g
PI4y = m4 * g
PI5y = m5 * g
PI6y = m6 * g

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

    w1dd_ = (-k1 * w1_ + k2 * w2_) / m1
    w2dd_ = (-k2 * (m1 + m2) * w2_ + w1_ * k1 * m2 + k3 * w3_ * m1) / m2 / m1
    w3dd_ = (-k3 * (m2 + m3) * w3_ + w2_ * k2 * m3 + w4_ * k4 * m2) / m3 / m2
    w4dd_ = (m3 * sin_th * (m5 * w5_ + m6 * w6_) * thetadd_ - m3 * sin_th * (l5 * m5 - l6 * m6) * thetad_ ** 2 + 2 * m3 * sin_th * (m5 * w4d_ + m6 * w5d_) * thetad_ + (k3 * w3_ - k4 * w4_) * (m5 + m6) * sin_th ** 2 + cos_th * m3 * (PI5y + PI6y) * sin_th + m3 * (k5 * w5_ + k6 * w6_) * cos_th - k4 * (m3 + m4) * w4_ + w3_ * k3 * m4) / m3 / (sin_th ** 2 * (m5 + m6) + m4)
    w5dd_ = (-m5 * (-l5 * (m5 + m6) * sin_th ** 2 + sin_th * (m5 * w5_ + m6 * w6_) * cos_th - l5 * m4) * thetadd_ + m5 * (w5_ * (m5 + m6) * sin_th ** 2 + sin_th * (l5 * m5 - l6 * m6) * cos_th + w5_ * m4) * thetad_ ** 2 - 2 * cos_th * sin_th * m5 * (m5 * w4d_ + m6 * w5d_) * thetad_ - sin_th ** 3 * PI5y * m6 - w5_ * sin_th ** 2 * k5 * m6 + (-PI6y * cos_th ** 2 * m5 - PI5y * (m4 + m5)) * sin_th - w6_ * cos_th ** 2 * k6 * m5 + w4_ * cos_th * k4 * m5 - w5_ * k5 * (m4 + m5)) / m5 / (sin_th ** 2 * (m5 + m6) + m4)
    w6dd_ = (-m6 * (l6 * (m5 + m6) * sin_th ** 2 + sin_th * (m5 * w5_ + m6 * w6_) * cos_th + l6 * m4) * thetadd_ + m6 * (w6_ * (m5 + m6) * sin_th ** 2 + sin_th * (l5 * m5 - l6 * m6) * cos_th + w6_ * m4) * thetad_ ** 2 - 2 * cos_th * sin_th * m6 * (m5 * w4d_ + m6 * w5d_) * thetad_ - sin_th ** 3 * PI6y * m5 - w6_ * sin_th ** 2 * k6 * m5 + (-PI5y * cos_th ** 2 * m6 - PI6y * (m4 + m6)) * sin_th - cos_th ** 2 * w5_ * k5 * m6 + w4_ * cos_th * k4 * m6 - w6_ * k6 * (m4 + m6)) / (sin_th ** 2 * (m5 + m6) + m4) / m6

    try:
        np.array([
            w1d_,  w1dd_,
            w2d_,  w2dd_,
            w3d_,  w3dd_,
            w4d_,  w4dd_,
            w5d_,  w5dd_,
            w6d_,  w6dd_,
            thetad_          # dtheta/dt
            ])
    except:
        pass


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

F_I1 = np.zeros((n_int, 3))
F_I2 = np.zeros((n_int, 3))
F_I3 = np.zeros((n_int, 3))
F_I4 = np.zeros((n_int, 3))
F_B55 = np.zeros((n_int, 3))
F_B56 = np.zeros((n_int, 3))

T45 = np.zeros((n_int, 3, 3))
T54 = np.zeros((n_int, 3, 3))

# %% Initial conditions
theta[0] = np.pi / 2
T45[0] = np.array(
    [[np.cos(theta[0]),  -np.sin(theta[0]), 0],
     [np.sin(theta[0]), np.cos(theta[0]), 0],
     [0, 0, 1]]
)
T54[0] = T45[0].T

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

    d1 = derivatives(state,                  t)
    d2 = derivatives(state + 0.5*deltaT*d1,  t + 0.5*deltaT)
    d3 = derivatives(state + 0.5*deltaT*d2,  t + 0.5*deltaT)
    d4 = derivatives(state +     deltaT*d3,  t +     deltaT)

    state = state + (deltaT / 6.0) * (d1 + 2*d2 + 2*d3 + d4)

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
        [[cos_th,  -sin_th, 0],
         [sin_th, cos_th, 0],
         [0, 0, 1]]
    )
    T54[i] = T45[i].T

# %% Post processing
sin_th = np.sin(theta)
cos_th = np.cos(theta)

# Reaction forces
R_I1x = k1 * w1
R_I2x = k2 * w2
R_I3x = k3 * w3
R_I4x = k4 * w4
R_B55x = k5 * w5
R_B56x = k6 * w6

R_I1y = (cos_th * m4 * (m5 * w5 + m6 * w6) * thetadd - cos_th * m4 * (l5 * m5 - l6 * m6) * thetad ** 2 + 2 * cos_th * m4 * (m5 * w4d + m6 * w5d) * thetad + (m5 + m6) * (PI1y + PI2y + PI3y + PI4y) * sin_th ** 2 + (w4 * k4 * (m5 + m6) * cos_th - (m5 + m6 + m4) * (k5 * w5 + k6 * w6)) * sin_th + m4 * ((PI5y + PI6y) * cos_th ** 2 + PI1y + PI2y + PI3y + PI4y)) / (sin_th ** 2 * (m5 + m6) + m4)
R_I2y = (cos_th * m4 * (m5 * w5 + m6 * w6) * thetadd - cos_th * m4 * (l5 * m5 - l6 * m6) * thetad ** 2 + 2 * cos_th * m4 * (m5 * w4d + m6 * w5d) * thetad + (m5 + m6) * (PI2y + PI3y + PI4y) * sin_th ** 2 + (w4 * k4 * (m5 + m6) * cos_th - (m5 + m6 + m4) * (k5 * w5 + k6 * w6)) * sin_th + m4 * ((PI5y + PI6y) * cos_th ** 2 + PI2y + PI3y + PI4y)) / (sin_th ** 2 * (m5 + m6) + m4)
R_I3y = (cos_th * m4 * (m5 * w5 + m6 * w6) * thetadd - cos_th * m4 * (l5 * m5 - l6 * m6) * thetad ** 2 + 2 * cos_th * m4 * (m5 * w4d + m6 * w5d) * thetad + (m5 + m6) * (PI3y + PI4y) * sin_th ** 2 + (w4 * k4 * (m5 + m6) * cos_th - (m5 + m6 + m4) * (k5 * w5 + k6 * w6)) * sin_th + ((PI5y + PI6y) * cos_th ** 2 + PI3y + PI4y) * m4) / (sin_th ** 2 * (m5 + m6) + m4)
R_I4y = (cos_th * m4 * (m5 * w5 + m6 * w6) * thetadd - cos_th * m4 * (l5 * m5 - l6 * m6) * thetad ** 2 + 2 * cos_th * m4 * (m5 * w4d + m6 * w5d) * thetad + PI4y * (m5 + m6) * sin_th ** 2 + (w4 * k4 * (m5 + m6) * cos_th - (m5 + m6 + m4) * (k5 * w5 + k6 * w6)) * sin_th + m4 * ((PI5y + PI6y) * cos_th ** 2 + PI4y)) / (sin_th ** 2 * (m5 + m6) + m4)
R_B55y = (m5 * (m6 * (w5 - w6) * sin_th ** 2 + w5 * m4) * thetadd - m5 * (m6 * (l5 + l6) * sin_th ** 2 + l5 * m4) * thetad ** 2 + 2 * m5 * ((sin_th ** 2 * m6 + m4) * w4d - sin_th ** 2 * w5d * m6) * thetad + cos_th * (PI5y * m6 - PI6y * m5) * sin_th ** 2 + m5 * ((-k5 * w5 - k6 * w6) * cos_th + k4 * w4) * sin_th + cos_th * PI5y * m4) / (sin_th ** 2 * (m5 + m6) + m4)
R_B56y = (m6 * (m5 * (w5 - w6) * sin_th ** 2 - w6 * m4) * thetadd - m6 * (m5 * (l5 + l6) * sin_th ** 2 + l6 * m4) * thetad ** 2 + 2 * m6 * ((-m5 * sin_th ** 2 - m4) * w5d + sin_th ** 2 * w4d * m5) * thetad + cos_th * (PI5y * m6 - PI6y * m5) * sin_th ** 2 - m6 * ((-k5 * w5 - k6 * w6) * cos_th + k4 * w4) * sin_th - cos_th * PI6y * m4) / (sin_th ** 2 * (m5 + m6) + m4)

F_I1[:, 0] = R_I2x - R_I1x
F_I2[:, 0] = R_I3x - R_I2x
F_I3[:, 0] = R_I4x - R_I3x
F_I4[:, 0] = R_B55x * cos_th + R_B56x * cos_th - R_I4x + R_B55y * sin_th - R_B56y * sin_th
F_B55[:, 0] = -sin_th * m5 * g - R_B55x
F_B56[:, 0] = -sin_th * m6 * g - R_B56x

F_I1[:, 1] = -PI1y + R_I1y - R_I2y
F_I2[:, 1] = -PI2y + R_I2y - R_I3y
F_I3[:, 1] = -PI3y + R_I3y - R_I4y
F_I4[:, 1] = -PI4y + R_I4y + k5 * w5 * sin_th - cos_th * R_B55y + k6 * w6 * sin_th + cos_th * R_B56y
F_B55[:, 1] = +cos_th * PI5y
F_B56[:, 1] = -cos_th * PI6y

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

F_I5 = np.einsum("nij,nj->ni", T45, F_B55)
F_I6 = np.einsum("nij,nj->ni", T45, F_B56)

# %% Plots
if plot_lines:
    rcparams = scivis.rcparams._prepare_rcparams()
    colors = ["#045893", "#db6100", "#108010", "#b40c0d", "#74499c", "#c158a0"]

    with mpl.rc_context(rcparams):
        # Plot beam deflections over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([w1, w2, w3, w4, w5, w6], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(6)], show_legend=False,
            colors=colors, linestyles = "-", linewidths=1.6
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("Beam deflections over time")
        plt.show()

        # Plot net forces in global x-direction over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([F_I1[:, 0], F_I2[:, 0], F_I3[:, 0],
                             F_I4[:, 0], F_I5[:, 0], F_I6[:, 0]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(6)], show_legend=False,
            colors=colors, linestyles = "-", linewidths=1.6
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("Net forces in global x-direction over time")
        plt.show()

        # Plot net forces in global y-direction over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([F_I1[:, 1], F_I2[:, 1], F_I3[:, 1],
                             F_I4[:, 1], F_I5[:, 1], F_I6[:, 1]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(6)], show_legend=False,
            colors=colors, linestyles = "-", linewidths=1.6
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("Net forces in global y-direction over time")
        plt.show()

        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([F_I1[:, 1], F_I2[:, 1], F_I3[:, 1],
                             F_I4[:, 1], F_I5[:, 1], F_I6[:, 1]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(6)], show_legend=False,
            colors=colors, linestyles = "-", linewidths=1.6
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("Net forces in global y-direction over time")
        plt.show()

        # Plot net forces in local x-direction for blade nodes over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([F_B55[:, 0], F_B56[:, 0]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(4, 6)], show_legend=False,
            colors=colors[:2], linestyles = "-", linewidths=1.6
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("Net forces in local x-direction for blade nodes over time")
        plt.show()

        # Plot net forces in local y-direction for blade nodes over time
        fig, ax, _ = scivis.plot_line(
            t_int, np.stack([F_B55[:, 1], F_B56[:, 1]], axis=0),
            plt_labels=[f"Mass{i+1:d}" for i in range(4, 6)], show_legend=False,
            colors=colors[:2], linestyles = "-", linewidths=1.6
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_title("Net forces in local y-direction for blade nodes over time")
        plt.show()
