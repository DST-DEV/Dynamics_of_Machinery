import numpy as np
import math

# Parameters
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

# Forces Matrix
PI1 = np.array([0, -m1 * g, 0])
PI2 = np.array([0, -m2 * g, 0])
PI3 = np.array([0, -m3 * g, 0])
PI4 = np.array([0, -m4 * g, 0])
PI5 = np.array([0, -m5 * g, 0])
PI6 = np.array([0, -m6 * g, 0])


# Integrator settings
deltaT = 0.0001
n_int = int(100000)

# Initial conditions
omega = np.pi  # constant rotational speed (rad/s) — adjust as needed
theta = np.zeros(n_int)
theta[0] = np.pi / 2
thetad = np.zeros(n_int)
thetad[0] = omega
thetadd = np.zeros(n_int)  # zero: constant speed means no angular acceleration

# Position vectors
rIoa = np.array([0, l1, 0])
rIab = np.array([0, l2, 0])
rIbc = np.array([0, l3, 0])
rIcd = np.array([0, l4, 0])
rB5de = np.array([0, l5, 0])
rB5df = np.array([0, -l6, 0])


# Transformation matrix
T45 = np.array(
    [
        [np.cos(theta[0]), np.sin(theta[0]), 0],
        [-np.sin(theta[0]), np.cos(theta[0]), 0],
        [0, 0, 1],
    ]
)
T54 = T45.T

# Get other position vectors in inertial reference frame
rIde = np.dot(T45, rB5de)
rIdf = np.dot(T45, rB5df)

# Creating the position vectors describing masses trajectories
rIob = rIoa + rIab
rIoc = rIob + rIbc
rIod = rIoc + rIcd
rIoe = rIod + rIde
rIof = rIoe + rIdf

# creating the position, speed and acceleration vectors describing masses trajectories over time
Ia = np.zeros((n_int, 3))
Ia[0] = rIoa
dIa = np.zeros((n_int, 3))  # speed of mass 1
ddIa = np.zeros((n_int, 3))  # acceleration of mass 1

Ib = np.zeros((n_int, 3))
Ib[0] = rIob
dIb = np.zeros((n_int, 3))  # speed of mass 2
ddIb = np.zeros((n_int, 3))  # acceleration of mass 2

Ic = np.zeros((n_int, 3))
Ic[0] = rIoc
dIc = np.zeros((n_int, 3))  # speed of mass 3
ddIc = np.zeros((n_int, 3))  # acceleration of mass 3

Id = np.zeros((n_int, 3))
Id[0] = rIod
dId = np.zeros((n_int, 3))  # speed of mass 4
ddId = np.zeros((n_int, 3))  # acceleration of mass 4

Ie = np.zeros((n_int, 3))
Ie[0] = rIoe
dIe = np.zeros((n_int, 3))  # speed of mass 5
ddIe = np.zeros((n_int, 3))  # acceleration of mass 5

If = np.zeros((n_int, 3))
If[0] = rIof
dIf = np.zeros((n_int, 3))  # speed of mass 6
ddIf = np.zeros((n_int, 3))  # acceleration of mass 6

# create the position, velocities and acceleration in their own coordinate system
w1 = np.zeros(n_int)
w2 = np.zeros(n_int)
w3 = np.zeros(n_int)
w4 = np.zeros(n_int)
w5 = np.zeros(n_int)
w6 = np.zeros(n_int)

w1d = np.zeros(n_int)
w2d = np.zeros(n_int)
w3d = np.zeros(n_int)
w4d = np.zeros(n_int)
w5d = np.zeros(n_int)
w6d = np.zeros(n_int)

w1dd = np.zeros(n_int)
w2dd = np.zeros(n_int)
w3dd = np.zeros(n_int)
w4dd = np.zeros(n_int)
w5dd = np.zeros(n_int)
w6dd = np.zeros(n_int)

# Reaction forces over time

F_I1 = np.zeros((n_int, 3))
I_T2 = np.zeros((n_int, 3))
I_T3 = np.zeros((n_int, 3))
I_T4 = np.zeros((n_int, 3))
I_T5 = np.zeros((n_int, 3))
I_T6 = np.zeros((n_int, 3))

t_int = np.zeros(n_int)  # time vector
for i in range(1, n_int):

    t_int[i - 1] = (i - 2) * deltaT

    # accelerations
    w1dd[i] = 48 * E * I1 * (l1**3 * w2[i - 1] - l2**3 * w1[i - 1]) / l1**3 / l2**3 / m1

    w2dd[i] = (
        -48
        * E
        * (
            l1**3 * l3**3 * (m1 + m2) * w2[i - 1]
            - l2**3 * (l1**3 * w3[i - 1] * m1 + 2 * l3**3 * w1[i - 1] * m2) / 2
        )
        * I1
        / m2
        / l2**3
        / l3**3
        / l1**3
        / m1
    )

    w3dd[i] = (
        -24
        * E
        * (
            l2**3 * l4**3 * (m2 + m3) * w3[i - 1]
            - l3**3 * (l2**3 * w4[i - 1] * m2 + 2 * l4**3 * w2[i - 1] * m3)
        )
        * I1
        / m3
        / l3**3
        / l4**3
        / l2**3
        / m2
    )

    w4dd[i] = (
        (
            m3
            * np.sin(theta[i - 1])
            * l3**3
            * l4**3
            * l5**3
            * l6**3
            * (
                m5 * w5[i - 1]
                + 2 * m5 * w5d[i - 1]
                + m6 * w6[i - 1]
                + 2 * m6 * w6d[i - 1]
            )
            * thetadd[i - 1]
            + 24
            * E
            * I1
            * l5**3
            * l6**3
            * (-(l3**3) * w4[i - 1] + l4**3 * w3[i - 1])
            * (m5 + m6)
            * np.sin(theta[i - 1]) ** 2
            + m3
            * g
            * np.cos(theta[i - 1])
            * l3**3
            * l4**3
            * l5**3
            * l6**3
            * (m5 + m6)
            * np.sin(theta[i - 1])
            - 24
            * E
            * (
                -m3
                * I2
                * l3**3
                * l4**3
                * (l5**3 * w6[i - 1] + l6**3 * w5[i - 1])
                * np.cos(theta[i - 1])
                / 8
                + (w4[i - 1] * (m3 + m4) * l3**3 - m4 * w3[i - 1] * l4**3)
                * I1
                * l5**3
                * l6**3
            )
        )
        / m3
        / l3**3
        / l5**3
        / l4**3
        / l6**3
        / ((m5 + m6) * np.sin(theta[i - 1]) ** 2 + m4)
    )

    w5dd[i] = (
        (
            -(
                2 * m5 * np.cos(theta[i - 1]) * np.sin(theta[i - 1]) * w5d[i - 1]
                + 2 * m6 * np.cos(theta[i - 1]) * np.sin(theta[i - 1]) * w6d[i - 1]
                - l5 * (m5 + m6) * np.sin(theta[i - 1]) ** 2
                + np.cos(theta[i - 1])
                * (m5 * w5[i - 1] + m6 * w6[i - 1])
                * np.sin(theta[i - 1])
                - l5 * m4
            )
            * l6**3
            * l5**3
            * m5
            * l4**3
            * thetadd[i - 1]
            - 3 * E * I2 * l4**3 * np.sin(theta[i - 1]) ** 2 * w5[i - 1] * l6**3 * m6
            - m5 * g * l4**3 * l5**3 * l6**3 * (m4 + m5 + m6) * np.sin(theta[i - 1])
            + 24
            * E
            * (
                -m5 * w6[i - 1] * np.cos(theta[i - 1]) ** 2 * I2 * l4**3 * l5**3 / 8
                + m5 * w4[i - 1] * np.cos(theta[i - 1]) * I1 * l5**3 * l6**3
                - w5[i - 1] * I2 * l4**3 * l6**3 * (m5 + m4) / 8
            )
        )
        / l6**3
        / ((m5 + m6) * np.sin(theta[i - 1]) ** 2 + m4)
        / l5**3
        / m5
        / l4**3
    )

    w6dd[i] = (
        (
            -(l6**3)
            * l5**3
            * l4**3
            * (
                2 * m5 * np.cos(theta[i - 1]) * np.sin(theta[i - 1]) * w5d[i - 1]
                + 2 * m6 * np.cos(theta[i - 1]) * np.sin(theta[i - 1]) * w6d[i - 1]
                + l6 * (m5 + m6) * np.sin(theta[i - 1]) ** 2
                + np.cos(theta[i - 1])
                * (m5 * w5[i - 1] + m6 * w6[i - 1])
                * np.sin(theta[i - 1])
                + l6 * m4
            )
            * m6
            * thetadd[i - 1]
            - 3 * E * I2 * l4**3 * l5**3 * np.sin(theta[i - 1]) ** 2 * m5 * w6[i - 1]
            - m6 * g * l4**3 * l5**3 * l6**3 * (m4 + m5 + m6) * np.sin(theta[i - 1])
            + 24
            * (
                -m6 * w5[i - 1] * np.cos(theta[i - 1]) ** 2 * I2 * l4**3 * l6**3 / 8
                + m6 * w4[i - 1] * np.cos(theta[i - 1]) * I1 * l5**3 * l6**3
                - w6[i - 1] * I2 * l4**3 * l5**3 * (m6 + m4) / 8
            )
            * E
        )
        / l6**3
        / ((m5 + m6) * np.sin(theta[i - 1]) ** 2 + m4)
        / l5**3
        / l4**3
        / m6
    )

    # velocities (Euler forward: v[i] = v[i-1] + a[i]*dt)
    w1d[i] = w1d[i - 1] + w1dd[i] * deltaT
    w2d[i] = w2d[i - 1] + w2dd[i] * deltaT
    w3d[i] = w3d[i - 1] + w3dd[i] * deltaT
    w4d[i] = w4d[i - 1] + w4dd[i] * deltaT
    w5d[i] = w5d[i - 1] + w5dd[i] * deltaT
    w6d[i] = w6d[i - 1] + w6dd[i] * deltaT

    # displacements (Euler forward: x[i] = x[i-1] + v[i-1]*dt)
    w1[i] = w1[i - 1] + w1d[i - 1] * deltaT
    w2[i] = w2[i - 1] + w2d[i - 1] * deltaT
    w3[i] = w3[i - 1] + w3d[i - 1] * deltaT
    w4[i] = w4[i - 1] + w4d[i - 1] * deltaT
    w5[i] = w5[i - 1] + w5d[i - 1] * deltaT
    w6[i] = w6[i - 1] + w6d[i - 1] * deltaT

    # reaction forces
    R_I1x = 48 * E * I1 * w1[i] / l1**3
    R_I2x = 48 * E * I1 * w2[i] / l2**3
    R_I3x = 24 * E * I1 * w3[i] / l3**3
    R_I4x = 24 * E * I1 * w4[i] / l4**3
    R_I5x = 3 * E * I2 * w5[i] / l5**3
    R_I6x = 3 * E * I2 * w6[i] / l6**3

    # update theta (Euler forward)
    thetad[i] = thetad[i - 1] + thetadd[i - 1] * deltaT
    theta[i] = theta[i - 1] + thetad[i - 1] * deltaT

    # update transformation matrix
    T45 = np.array(
        [
            [np.cos(theta[i]), np.sin(theta[i]), 0],
            [-np.sin(theta[i]), np.cos(theta[i]), 0],
            [0, 0, 1],
        ]
    )
    T54 = T45.T

    # position vectors
    rIoa = np.array([w1[i], l1, 0])
    rIab = np.array([w2[i], l2, 0])
    rIbc = np.array([w3[i], l3, 0])
    rIcd = np.array([w4[i], l4, 0])
    rB5de = np.array([w5[i], l5, 0])
    rB5df = np.array([w6[i], -l6, 0])

    # creating the position vectors describing masses trajectories
    Ia[i] = rIoa
    Ib[i] = rIoa + rIab
    Ic[i] = rIoa + rIab + rIbc
    Id[i] = rIoa + rIab + rIbc + rIcd
    Ie[i] = rIoa + rIab + rIbc + rIcd + T45.T @ rB5de
    If[i] = rIoa + rIab + rIbc + rIcd + T45.T @ rB5df

    # creating the speed and acceleration vectors describing masses trajectories over time
    dIa[i] = (Ia[i] - Ia[i - 1]) / deltaT
    ddIa[i] = (dIa[i] - dIa[i - 1]) / deltaT
    dIb[i] = (Ib[i] - Ib[i - 1]) / deltaT
    ddIb[i] = (dIb[i] - dIb[i - 1]) / deltaT
    dIc[i] = (Ic[i] - Ic[i - 1]) / deltaT
    ddIc[i] = (dIc[i] - dIc[i - 1]) / deltaT
    dId[i] = (Id[i] - Id[i - 1]) / deltaT
    ddId[i] = (dId[i] - dId[i - 1]) / deltaT
    dIe[i] = (Ie[i] - Ie[i - 1]) / deltaT
    ddIe[i] = (dIe[i] - dIe[i - 1]) / deltaT
    dIf[i] = (If[i] - If[i - 1]) / deltaT
    ddIf[i] = (dIf[i] - dIf[i - 1]) / deltaT

# ── Animation ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Subsample to at most 500 frames for smooth playback
step = max(1, n_int // 500)
frames = np.arange(0, n_int, step)
t_frames = t_int[frames]

origin = np.zeros(3)

fig, ax = plt.subplots(figsize=(6, 8))
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Mass positions over time")

# Determine axis limits from the data
all_x = np.concatenate(
    [
        Ia[frames, 0],
        Ib[frames, 0],
        Ic[frames, 0],
        Id[frames, 0],
        Ie[frames, 0],
        If[frames, 0],
    ]
)
all_y = np.concatenate(
    [
        Ia[frames, 1],
        Ib[frames, 1],
        Ic[frames, 1],
        Id[frames, 1],
        Ie[frames, 1],
        If[frames, 1],
    ]
)
margin = 0.05
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(-margin, all_y.max() + margin)

# Line representing the chain O-a-b-c-d with branches d-e and d-f
(chain_line,) = ax.plot([], [], "b-o", lw=1.5, ms=6, label="chain")
(branch_e,) = ax.plot([], [], "g-o", lw=1.5, ms=6, label="mass e")
(branch_f,) = ax.plot([], [], "r-o", lw=1.5, ms=6, label="mass f")
time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
ax.legend(loc="upper right")
ax.plot(*origin[:2], "ks", ms=8)  # fixed origin


def init():
    chain_line.set_data([], [])
    branch_e.set_data([], [])
    branch_f.set_data([], [])
    time_text.set_text("")
    return chain_line, branch_e, branch_f, time_text


def update(frame):
    pts = np.array([origin, Ia[frame], Ib[frame], Ic[frame], Id[frame]])
    chain_line.set_data(pts[:, 0], pts[:, 1])
    branch_e.set_data([Id[frame, 0], Ie[frame, 0]], [Id[frame, 1], Ie[frame, 1]])
    branch_f.set_data([Id[frame, 0], If[frame, 0]], [Id[frame, 1], If[frame, 1]])
    time_text.set_text(f"t = {t_int[frame]:.4f} s")
    return chain_line, branch_e, branch_f, time_text


ani = animation.FuncAnimation(
    fig, update, frames=frames, init_func=init, interval=20, blit=True
)
plt.tight_layout()
plt.show()
