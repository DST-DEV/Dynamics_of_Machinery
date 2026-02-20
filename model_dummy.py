import numpy as np

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

# Integrator settings
deltaT = 1e-5
n_int = int(8e6)

# Initial conditions
theta = np.zeros(n_int)
theta[0] = np.pi / 2
thetad = np.zeros(n_int)

x1 = np.zeros(n_int)
x1d = np.zeros(n_int)
x2 = np.zeros(n_int)
x2d = np.zeros(n_int)
x3 = np.zeros(n_int)
x3d = np.zeros(n_int)
x4 = np.zeros(n_int)
x4d = np.zeros(n_int)
x5 = np.zeros(n_int)
x5d = np.zeros(n_int)
x6 = np.zeros(n_int)
x6d = np.zeros(n_int)

# Position vectors
Iroa = np.array([x1[0], l1, 0])
Irab = np.array([x2[0], l2, 0])
Irbc = np.array([x3[0], l3, 0])
Ircd = np.array([x4[0], l4, 0])
B5rde = np.array([x5[0], l5, 0])
B5rdf = np.array([x6[0], -l6, 0])

# Transformation matrix
Ttheta = np.array([
    [np.cos(theta[0]), np.sin(theta[0]), 0],
    [-np.sin(theta[0]), np.cos(theta[0]), 0],
    [0, 0, 1]
    ])

# Creating the position vectors describing masses trajectories
Irob = Iroa + Irab
Iroc = Iroa + Irab + Irbc
Irod = Iroa + Irab + Irbc + Ircd
Iroe = Iroa + Irab + Irbc + Ircd + np.dot(Ttheta.T, B5rde)
Irof = Iroa + Irab + Irbc + Ircd + np.dot(Ttheta.T, B5rdf)

rx_a = np.zeros(n_int)
ry_a = np.zeros(n_int)
rz_a = np.zeros(n_int)
rx_a[0] = Iroa[0]
ry_a[0] = Iroa[1]
rz_a[0] = Iroa[2]

rx_b = np.zeros(n_int)
ry_b = np.zeros(n_int)
rz_b = np.zeros(n_int)
rx_b[0] = Irob[0]
ry_b[0] = Irob[1]
rz_b[0] = Irob[2]

rx_c = np.zeros(n_int)
ry_c = np.zeros(n_int)
rz_c = np.zeros(n_int)
rx_c[0] = Iroc[0]
ry_c[0] = Iroc[1]
rz_c[0] = Iroc[2]

rx_d = np.zeros(n_int)
ry_d = np.zeros(n_int)
rz_d = np.zeros(n_int)
rx_d[0] = Irod[0]
ry_d[0] = Irod[1]
rz_d[0] = Irod[2]

rx_e = np.zeros(n_int)
ry_e = np.zeros(n_int)
rz_e = np.zeros(n_int)
rx_e[0] = Iroe[0]
ry_e[0] = Iroe[1]
rz_e[0] = Iroe[2]

rx_f = np.zeros(n_int)
ry_f = np.zeros(n_int)
rz_f = np.zeros(n_int)
rx_f[0] = Irof[0]
ry_f[0] = Irof[1]
rz_f[0] = Irof[2]

# Reaction forces
React_y1 = np.zeros(n_int)
React_y2 = np.zeros(n_int)
React_y3 = np.zeros(n_int)
React_y4 = np.zeros(n_int)
React_y5 = np.zeros(n_int)
React_y6 = np.zeros(n_int)

React_y1[0] = k5 * m5 * x5[0]

React_y2[0] = k5 * m5 * x5[0] * np.sin(theta[0])

React_y3[0] = 4 * g * m4**2 + 4 * g * m3 * m4

React_y4[0] = 4 * g * m4**2 + 8 * m4 * thetad[0] * x5[0] * np.cos(theta[0])


x1dd = np.zeros(n_int)
x2dd = np.zeros(n_int)
x3dd = np.zeros(n_int)
x4dd = np.zeros(n_int)
x5dd = np.zeros(n_int)
x6dd = np.zeros(n_int)
t_int = np.zeros(n_int)

for i in range(1, n_int):
    t_int[i-1] = (i-2)*deltaT

    # accelerations
    x1dd[i-1] = (-k1*x1[i-1] + k2*x2[i-1]) / m1

    x2dd[i-1] = (k1*m2*x1[i-1] - k2*m1*x2[i-1] - k2*m2*x2[i-1] + k3*m1*x3[i-1]) / (m1*m2)

    x3dd[i-1] = (k2*m3*x2[i-1] - k3*m2*x3[i-1] - k3*m3*x3[i-1] + k4*m2*x4[i-1]) / (m2*m3)

    x4dd[i-1] = 2*k4*m3*x4[i-1] - 2*k3*m4*x3[i-1] + 2*k4*m4*x4[i-1]

    x5dd[i-1] = m5 * thetad[i-1]**2 * x5[i-1]

    x6dd[i-1] = -m5 * thetad[i-1]**2 * x6[i-1]

    # velocities
    x1d[i] = x1d[i-1] + x1dd[i-1]*deltaT
    x2d[i] = x2d[i-1] + x2dd[i-1]*deltaT
    x3d[i] = x3d[i-1] + x3dd[i-1]*deltaT
    x4d[i] = x4d[i-1] + x4dd[i-1]*deltaT
    x5d[i] = x5d[i-1] + x5dd[i-1]*deltaT
    x6d[i] = x6d[i-1] + x6dd[i-1]*deltaT

    # displacements
    theta[i] = theta[i-1] + thetad[i-1]*deltaT
    x1[i] = x1[i-1] + x1d[i-1]*deltaT
    x2[i] = x2[i-1] + x2d[i-1]*deltaT
    x3[i] = x3[i-1] + x3d[i-1]*deltaT
    x4[i] = x4[i-1] + x4d[i-1]*deltaT
    x5[i] = x5[i-1] + x5d[i-1]*deltaT
    x6[i] = x6[i-1] + x6d[i-1]*deltaT

    # reaction forces
    React_y1[i] = k5*m5*x5[i]*np.sin(theta[i])

    React_y2[i] = k5*m5*x5[i]*np.sin(theta[i])

    React_y3[i] = 4*g*m4**2

    React_y4[i] = 4*g*m4**2

    React_y5[i] = 2*m4*thetad[i]*x5[i]

    React_y6[i] = 2*m4*thetad[i]*x6[i]

    # transformation matrix
    Ttheta = np.array([
        [np.cos(theta[i]), np.sin(theta[i]), 0],
        [-np.sin(theta[i]), np.cos(theta[i]), 0],
        [0, 0, 1]
        ])

    # position vectors
    Iroa = np.array([x1[i], l1, 0])
    Irab = np.array([x2[i], l2, 0])
    Irbc = np.array([x3[i], l3, 0])
    Ircd = np.array([x4[i], l4, 0])
    B5rde = np.array([x5[i], l5, 0])
    B5rdf = np.array([x6[i], -l6, 0])

    # creating the position vectors describing masses trajectories
    Irob = Iroa + Irab
    Iroc = Iroa + Irab + Irbc
    Irod = Iroa + Irab + Irbc + Ircd
    Iroe = Iroa + Irab + Irbc + Ircd + Ttheta.T * B5rde
    Irof = Iroa + Irab + Irbc + Ircd + Ttheta.T * B5rdf

    rx_a[i] = Iroa[0]
    ry_a[i] = Iroa[1]
    rz_a[i] = Iroa[2]

    rx_b[i] = Irob[0]
    ry_b[i] = Irob[1]
    rz_b[i] = Irob[2]

    rx_c[i] = Iroc[0]
    ry_c[i] = Iroc[1]
    rz_c[i] = Iroc[2]

    rx_d[i] = Irod[0]
    ry_d[i] = Irod[1]
    rz_d[i] = Irod[2]

    rx_e[i] = Iroe[0]
    ry_e[i] = Iroe[1]
    rz_e[i] = Iroe[2]

    rx_f[i] = Irof[0]
    ry_f[i] = Irof[1]
    rz_f[i] = Irof[2]

print("Reaction forces:")
print(f"React_y1: {React_y1}")
print(f"React_y2: {React_y2}")
print(f"React_y3: {React_y3}")
print(f"React_y4: {React_y4}")
