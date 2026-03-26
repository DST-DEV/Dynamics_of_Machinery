import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

import scivis

# %% User settings
T = 48  # Total simulation time

plot_lines = True
# %% Parameters
l1 = 0.193
l2 = 0.292
l3 = 0.220
l4 = 0.271
l5 = 0.430
l6 = 0.430

m1 = 2.41
m2 = 1.84395
m3 = 1.84585
m4 = 2.5954
m5 = 0.7353
m6 = 0.7353
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
# k1 = 4 * (12 * E * I1) / l1**3
# k2 = 4 * (12 * E * I1) / l2**3
# k3 = 2 * (12 * E * I1) / l3**3
# k4 = 2 * (12 * E * I1) / l4**3
# k5 = (3 * E * I2) / l5**3
# k6 = (3 * E * I2) / l6**3

k1 = 4653.30
k2 = 1431.16
k3 = 286.21
k4 = 895.20
k5 = 85.43
k6 = 85.43

PI1y = m1 * g
PI2y = m2 * g
PI3y = m3 * g
PI4y = m4 * g
PI5y = m5 * g
PI6y = m6 * g

# %% Integrator settings
deltaT = 0.0005
n_int = int(np.ceil(T / deltaT)) + 1
t_int = np.arange(0, T + deltaT, deltaT)  # time vector


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
    thetad_  = np.pi / 24
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
    [[np.cos(theta[0]), -np.sin(theta[0]), 0],
     [np.sin(theta[0]), np.cos(theta[0]), 0],
     [0, 0, 1]]
)
T54[0] = T45[0].T

# Pack initial state: [w1, w1d, w2, w2d, w3, w3d, w4, w4d, w5, w5d, w6, w6d, theta]
w5_stat = -PI5y*np.sin(theta[0])/k5
w6_stat = -PI6y*np.sin(theta[0])/k6

w5[0] = -PI5y*np.sin(theta[0])/k5
w6[0] = -PI6y*np.sin(theta[0])/k6

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

# Sum of forces on the bodies
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
rIoa[:, 0] = w1
rIoa[:, 1] = l1

rIab[:, 0] = w2
rIab[:, 1] = l2

rIbc[:, 0] = w3
rIbc[:, 1] = l3

rIcd[:, 0] = w4
rIcd[:, 1] = l4

rB5de[:, 0] = w5
rB5de[:, 1] = l5

rB5df[:, 0] = w6
rB5df[:, 1] = -l6

# creating the position vectors describing masses trajectories
rIob = rIoa + rIab
rIoc = rIoa + rIab + rIbc
rIod = rIoa + rIab + rIbc + rIcd
rIoe = rIoa + rIab + rIbc + rIcd + np.einsum("nij,nj->ni", T45, rB5de)
rIof = rIoa + rIab + rIbc + rIcd + np.einsum("nij,nj->ni", T45, rB5df)

F_I5 = np.einsum("nij,nj->ni", T45, F_B55)
F_I6 = np.einsum("nij,nj->ni", T45, F_B56)

def compute_stresses(w, l, h, b, I, R_y, R_x, E, n_elements=1, is_blade=False):
    """
    n_elements: Anzahl der parallelen Stäbe (z.B. 4 für Beam 1 & 2)
    """
    A_single = b * h
    
    # 1. Bending Stress: Hängt nur von Krümmung/Verformung ab. 
    # n_elements spielt hier keine Rolle, da jeder Stab die gleiche Krümmung erfährt.
    factor_b = 1.5 if is_blade else 3.0
    sigma_b = (factor_b * E * w * h / (l**2)) / 1e6
    
    # 2. Normal Stress: Kraft teilt sich auf n Stäbe auf
    sigma_n = ((R_y / n_elements) / A_single) / 1e6
    
    # 3. Max Shear Stress: Kraft teilt sich auf n Stäbe auf
    tau_max = (1.5 * (R_x / n_elements) / A_single) / 1e6
    
    # 4. Von Mises
    sigma_vm = np.sqrt((sigma_b + sigma_n)**2 + 3 * tau_max**2)
    
    return sigma_b, sigma_n, tau_max, sigma_vm

# Aufrufe mit n_elements
S1_b, S1_n, S1_t, S1_vm = compute_stresses(w1, l1, h1, b1, I1, R_I1y, R_I1x, E, n_elements=4, is_blade=False)
S2_b, S2_n, S2_t, S2_vm = compute_stresses(w2, l2, h1, b1, I1, R_I2y, R_I2x, E, n_elements=4, is_blade=False)
S3_b, S3_n, S3_t, S3_vm = compute_stresses(w3, l3, h1, b1, I1, R_I3y, R_I3x, E, n_elements=2, is_blade=False)
S4_b, S4_n, S4_t, S4_vm = compute_stresses(w4, l4, h1, b1, I1, R_I4y, R_I4x, E, n_elements=2, is_blade=False)
S5_b, S5_n, S5_t, S5_vm = compute_stresses(w5, l5, h2, b2, I2, R_B55y, R_B55x, E, n_elements=1, is_blade=True)
S6_b, S6_n, S6_t, S6_vm = compute_stresses(w6, l6, h2, b2, I2, R_B56y, R_B56x, E, n_elements=1, is_blade=True)

# %% Standard Matplotlib Plotting
stress_types = [
    ([S1_b, S2_b, S3_b, S4_b, S5_b, S6_b], "Bending Stress", "MPa"),
    ([S1_n, S2_n, S3_n, S4_n, S5_n, S6_n], "Normal Stress (Axial)", "MPa"),
    ([S1_t, S2_t, S3_t, S4_t, S5_t, S6_t], "Shear Stress", "MPa"),
    ([S1_vm, S2_vm, S3_vm, S4_vm, S5_vm, S6_vm], "Von Mises Stress", "MPa")
]


# %% Fatigue Estimation (R10 / Equivalent Stress)
def calculate_r10(stress_data, m_exponent=3, n_ref=1e7, simulation_time=T):

    delta_sigma = np.max(stress_data) - np.min(stress_data)
    f_rot = (np.pi/24) / (2 * np.pi) 
    n_cycles_sim = f_rot * simulation_time
    r10 = delta_sigma * (n_cycles_sim / n_ref)**(1/m_exponent)
    return r10

# Wähle m=3 für Stahlkomponenten (Turm)
m_steel = 3

r10_beam1 = calculate_r10(S1_vm, m_exponent=m_steel)
r10_beam5 = calculate_r10(S5_vm, m_exponent=m_steel)

print(f"--- Fatigue Estimation (R10) ---")
print(f"Beam 1 (Tower Base) R10: {r10_beam1:.4f} MPa")
print(f"Beam 5 (Blade) R10:      {r10_beam5:.4f} MPa")

# Convert theta from radians to degrees and wrap to [0, 360]
azimuth_deg = np.degrees(theta) % 360

# Create sorting indices to keep plot lines continuous
sort_idx = np.argsort(azimuth_deg)
azimuth_plot = azimuth_deg[sort_idx]
# %% Azimuthal Stress Plotting
for data_list, title, unit in stress_types:
    plt.figure(figsize=(10, 5))
    for i, data in enumerate(data_list):
        # Apply the sorting index to the data as well
        plt.plot(azimuth_plot, data[sort_idx], label=f"Beam {i+1}")
    
    #if "Von Mises" in title:
    #    plt.axhline(250, color='r', linestyle='--', label='Yield Limit (250 MPa)')
        
    #plt.title(f"{title} vs Azimuth")
    plt.xlabel("Azimuth Angle [deg]")
    plt.ylabel(f"Stress [{unit}]")
    plt.xticks(np.arange(0, 361, 45))  # Professional 45-degree increments
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# %% Azimuthal Deflection Plot
plt.figure(figsize=(10, 5))
mass_labels = ["Mass 1", "Mass 2", "Mass 3", "Mass 4", "Mass 5 (Blade)", "Mass 6 (Blade)"]
deflections = [w1, w2, w3, w4, w5, w6]

for label, w_data in zip(mass_labels, deflections):
    plt.plot(azimuth_plot, w_data[sort_idx]*1000, label=label)

plt.title("Beam Deflections vs Azimuth")
plt.xlabel("Azimuth Angle [deg]")
plt.ylabel("Deflection [mm]")
plt.xticks(np.arange(0, 361, 45))
plt.legend()
plt.grid(True)
plt.show()

# %% Updated Animation with Runtime Display
fig_anim, ax_anim = plt.subplots(figsize=(6, 8))
ax_anim.set_aspect('equal')
ax_anim.set_xlim(-1, 1)
ax_anim.set_ylim(0, 2)
ax_anim.grid(True)
ax_anim.set_title("Wind Turbine Digital Twin Animation")

# Text object for the timer
time_template = 'Time = %.2fs'
time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes, 
                         weight='bold', fontsize=12)

# Line object representing the structure
line, = ax_anim.plot([], [], 'o-', lw=2, color='navy', markersize=6)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    # Mapping points: Origin -> A -> B -> C -> D -> E (Blade 1) -> D -> F (Blade 2)
    # Using the pre-calculated global coordinates
    xs = [0, rIoa[frame, 0], rIob[frame, 0], rIoc[frame, 0], 
          rIod[frame, 0], rIoe[frame, 0], rIod[frame, 0], rIof[frame, 0]]
    ys = [0, rIoa[frame, 1], rIob[frame, 1], rIoc[frame, 1], 
          rIod[frame, 1], rIoe[frame, 1], rIod[frame, 1], rIof[frame, 1]]
    
    line.set_data(xs, ys)
    time_text.set_text(time_template % (t_int[frame]))
    return line, time_text

# Adjust 'step' to control playback speed vs smoothness
step = 40 
frames = range(0, n_int, step)

ani = FuncAnimation(fig_anim, update, frames=frames,
                    init_func=init, blit=True, interval=25)

plt.show()

# %% Combined Stress and Deflection Analysis
lstyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 5))]

# Added a 5th data category for deflections
plot_data = stress_types + [
    ([w1*1000, w2*1000, w3*1000, w4*1000, w5*1000, w6*1000], "Beam Deflections", "mm")
]

import matplotlib.gridspec as gridspec

# Linestyles definieren
lstyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 5))]

# Figur mit 3 Zeilen und 2 Spalten erstellen
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(3, 2, figure=fig)

# --- ZEILE 1: Deflections (Split Tower vs. Blade) ---
ax_t_def = fig.add_subplot(gs[0, 0])
ax_b_def = fig.add_subplot(gs[0, 1])

# Tower Deflections (Beams 1-4)
tower_defs = [w1*1000, w2*1000, w3*1000, w4*1000]
for j, data in enumerate(tower_defs):
    ax_t_def.plot(azimuth_plot, data[sort_idx], linestyle=lstyles[j], label=f"Beam {j+1}")
ax_t_def.set_title("Tower Deflections", fontweight='bold')
ax_t_def.set_ylabel("Deflection [mm]")

# Blade Deflections (Beams 5-6)
blade_defs = [w5*1000, w6*1000]
for j, data in enumerate(blade_defs):
    ax_b_def.plot(azimuth_plot, data[sort_idx], linestyle=lstyles[j+4], label=f"Beam {j+5} (Blade)")
ax_b_def.set_title("Blade Deflections", fontweight='bold')

for ax in [ax_t_def, ax_b_def]:
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(np.arange(0, 361, 45))

# --- ZEILE 2 & 3: Stresses ---
stress_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), 
               fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

handles, labels = [], []
for i, (ax, (data_list, title, unit)) in enumerate(zip(stress_axes, stress_types)):
    for j, data in enumerate(data_list):
        line, = ax.plot(azimuth_plot, data[sort_idx], linestyle=lstyles[j], linewidth=1.5)
        if i == 0: # Legendendaten nur einmal sammeln
            handles.append(line)
            labels.append(f"Beam {j+1}" + (" (Blade)" if j >= 4 else ""))
    
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(f"[{unit}]")
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, linestyle='--', alpha=0.4)
    if i >= 2: ax.set_xlabel("Azimuth Angle [$^{\circ}$]")

# Globale Legende ganz oben
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 0.98), 
           ncol=6, fontsize='small', frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("stress_analysis_distinguishable.pdf", bbox_inches='tight')
plt.show()

# Define the element names for the rows
elements = ["Beam 1 (Tower Base)", "Beam 2", "Beam 3", "Beam 4", "Beam 5 (Blade)", "Beam 6 (Blade)"]

# Extract Max values from your existing stress variables
data = {
    "Bending [MPa]": [np.max(S1_b), np.max(S2_b), np.max(S3_b), np.max(S4_b), np.max(S5_b), np.max(S6_b)],
    "Normal [MPa]":  [np.max(S1_n), np.max(S2_n), np.max(S3_n), np.max(S4_n), np.max(S5_n), np.max(S6_n)],
    "Shear [MPa]":   [np.max(S1_t), np.max(S2_t), np.max(S3_t), np.max(S4_t), np.max(S5_t), np.max(S6_t)],
    "Von Mises [MPa]": [np.max(S1_vm), np.max(S2_vm), np.max(S3_vm), np.max(S4_vm), np.max(S5_vm), np.max(S6_vm)]
}

# Create DataFrame
df_max_stresses = pd.DataFrame(data, index=elements)

# Print formatted table
print("\n" + "="*30)
print("MAXIMUM STRESS SUMMARY")
print("="*30)
print(df_max_stresses.round(4).to_string())
print("="*30)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Setup Time mask for 0-10 seconds
# Assuming 't_int' is your simulation time array
mask = (t_int >= 0) & (t_int <= 10)
t_plot = t_int[mask]

# 2. Define Linestyles
lstyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 5))]
labels = [f"Beam {i+1}" for i in range(6)]
stress_data_all = [
    ([S1_b, S2_b, S3_b, S4_b, S5_b, S6_b], "Bending Stress", "MPa"),
    ([S1_n, S2_n, S3_n, S4_n, S5_n, S6_n], "Normal Stress (Axial)", "MPa"),
    ([S1_t, S2_t, S3_t, S4_t, S5_t, S6_t], "Shear Stress", "MPa"),
    ([S1_vm, S2_vm, S3_vm, S4_vm, S5_vm, S6_vm], "Von Mises Stress", "MPa")
]
deflections = [w1, w2, w3, w4, w5, w6]

# 3. Create Figure with GridSpec (3 rows total: 1 for deflection, 2 for stress)
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

# --- Top Plot: Deflection (1 wide) ---
ax_def = fig.add_subplot(gs[0, :])
handles = []
for i, w_data in enumerate(deflections):
    line, = ax_def.plot(t_plot, w_data[mask] * 1000, label=labels[i], 
                        linestyle=lstyles[i], linewidth=1.5)
    handles.append(line)

ax_def.set_title("Beam Deflections over Time", fontweight='bold')
ax_def.set_ylabel("Deflection [mm]")
ax_def.grid(True, alpha=0.3)

# --- Bottom 4 Plots: Stresses (2x2 grid) ---
stress_axes = [
    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
]

for ax, (data_list, title, unit) in zip(stress_axes, stress_data_all):
    for i, data in enumerate(data_list):
        ax.plot(t_plot, data[mask], linestyle=lstyles[i], linewidth=1.2)
    
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel(f"[{unit}]")
    ax.grid(True, alpha=0.3)

# Set common X-label for the bottom row
stress_axes[2].set_xlabel("Time [s]")
stress_axes[3].set_xlabel("Time [s]")

# Global legend at the very top
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
           ncol=6, fontsize='small', frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("turbine_time_response_122.pdf", bbox_inches='tight')
plt.show()