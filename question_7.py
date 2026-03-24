from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model Parameters
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

# Calculate modal parameters of the mathematical model

M = np.array(
    [
        [m1, 0, 0, 0, 0, 0],
        [m2, m2, 0, 0, 0, 0],
        [m3, m3, m3, 0, 0, 0],
        [m4, m4, m4, m4, 0, 0],
        [m5, m5, m5, m5, m5, 0],
        [m6, m6, m6, m6, 0, m6],
    ]
)

K = -np.array(
    [
        [-k1, k2, 0, 0, 0, 0],
        [0, -k2, k3, 0, 0, 0],
        [0, 0, -k3, k4, 0, 0],
        [0, 0, 0, -k4, k5, k6],
        [0, 0, 0, 0, -k5, 0],
        [0, 0, 0, 0, 0, -k6],
    ]
)


# Damping matrix calculation
def modal_analysis(M, K):
    """Calculate eigenfrequencies and mode shapes."""
    from scipy.linalg import eig

    n_dof = M.shape[0]

    zero_mat = np.zeros((n_dof, n_dof))
    A = np.block([[M, zero_mat], [zero_mat, M]])
    B = np.block([[zero_mat, K], [-M, zero_mat]])

    eigval, eigvec = eig(-B, A)
    eigfreqs = np.abs(eigval) / (2 * np.pi)  # Eigenfrequencies in [Hz]

    # Sort by eigenfrequency
    idx_sorted = np.argsort(eigfreqs)
    eigfreqs = eigfreqs[idx_sorted]
    eigval = eigval[idx_sorted]
    eigvec = eigvec[:, idx_sorted]

    # Normalize eigenvectors
    idx = N_DOF + np.argmax(np.abs(eigvec[n_dof:]), axis=0)
    scale = eigvec[idx, np.arange(eigvec.shape[1])]
    eigvec_norm = np.round(eigvec / scale, 4)

    # Extract modal matrices shape matrix
    modeshapes = eigvec_norm[n_dof:, ::2].real

    return eigfreqs[::2], modeshapes


def fit_damping(xi, eifreqs):
    """Fit Rayleigh damping parameters (alpha, beta) from damping ratios."""
    b = np.asarray(xi).reshape((-1, 1))

    A_alpha = 1 / (2 * eifreqs * (2 * np.pi)).reshape((-1, 1))
    A_beta = (eifreqs * (2 * np.pi) / 2).reshape((-1, 1))
    A_rayleigh = np.hstack([A_alpha, A_beta])

    sol_damping, *_ = np.linalg.lstsq(A_rayleigh, b, rcond=None)
    alpha, beta = sol_damping.flatten()

    return alpha, beta


# Calculate eigenfrequencies from model
eigfreqs_model, modeshapes = modal_analysis(M, K)

# Load experimental data
res_exp = pd.read_csv(Path(__file__).parent / "EMA_global_comparison_summary.csv")
eigfreqs_exp = res_exp["fn_freq_Hz"].values
xi_exp_ls = res_exp["zeta_ls_ref_acc"].values
xi_exp_logdec = res_exp["zeta_time_logdec"].values
xi_exp_hpp = res_exp["zeta_hpp_ref_acc"].values

# %% USER SETTINGS: Choose damping calculation method
# =========================================================================
# DAMPING_METHOD: Choose which experimental damping ratios to use
#   - 'ls': Use zeta_ls_ref_acc (least squares method from frequency domain)
#   - 'logdec': Use zeta_time_logdec (logarithmic decrement from time domain)
#   - 'hpp': Use zeta_hpp_ref_acc (half-power point method)
#   - 'constant': Use constant damping ratio of 0.003 for all modes
#   - 'mixed': Use PER_MODE_DAMPING_METHODS to specify method for each mode
DAMPING_METHOD = "mixed"

# PER_MODE_DAMPING_METHODS: Specify which method to use for each mode (only used if DAMPING_METHOD = 'mixed')
#   Each element can be: 'ls', 'logdec', 'hpp', or a float value for constant
#   Example: ['logdec', 'logdec', 'logdec', 'ls', 'ls', 'ls']
#   Example: ['logdec', 'logdec', 'logdec', 0.005, 0.005, 0.005]
PER_MODE_DAMPING_METHODS = ['logdec', 'logdec', 'logdec', 'ls', 'ls', 'ls']

# FREQUENCY_SOURCE: Choose which frequencies to use for fitting Rayleigh damping
#   - 'model': Use eigenfrequencies calculated from M and K matrices
#   - 'experimental': Use experimental eigenfrequencies from EMA
FREQUENCY_SOURCE = "experimental"
# =========================================================================

# Select damping ratios based on chosen method
if DAMPING_METHOD == "ls":
    xi_chosen = xi_exp_ls
    print(f"Using least-squares damping ratios (zeta_ls_ref_acc) for all modes:")
elif DAMPING_METHOD == "logdec":
    xi_chosen = xi_exp_logdec
    print(f"Using logarithmic decrement damping ratios (zeta_time_logdec) for all modes:")
elif DAMPING_METHOD == "hpp":
    xi_chosen = xi_exp_hpp
    print(f"Using half-power point damping ratios (zeta_hpp_ref_acc) for all modes:")
elif DAMPING_METHOD == "constant":
    xi_chosen = np.full(N_DOF, 0.003)
    print(f"Using constant damping ratio: 0.003 for all modes")
elif DAMPING_METHOD == "mixed":
    xi_chosen = np.zeros(N_DOF)
    method_names = []

    if len(PER_MODE_DAMPING_METHODS) != N_DOF:
        raise ValueError(f"PER_MODE_DAMPING_METHODS must have {N_DOF} elements, got {len(PER_MODE_DAMPING_METHODS)}")

    for i, method in enumerate(PER_MODE_DAMPING_METHODS):
        if method == 'ls':
            xi_chosen[i] = xi_exp_ls[i]
            method_names.append('ls')
        elif method == 'logdec':
            xi_chosen[i] = xi_exp_logdec[i]
            method_names.append('logdec')
        elif method == 'hpp':
            xi_chosen[i] = xi_exp_hpp[i]
            method_names.append('hpp')
        elif isinstance(method, (int, float)):
            xi_chosen[i] = float(method)
            method_names.append(f'{method}')
        else:
            raise ValueError(f"Invalid method '{method}' for mode {i+1}. Use 'ls', 'logdec', 'hpp', or a numeric value.")

    print(f"Using mixed damping methods per mode:")
    for i in range(N_DOF):
        print(f"  Mode {i+1}: {method_names[i]:8s} → ζ = {xi_chosen[i]:.6f}")
else:
    raise ValueError(
        f"Invalid DAMPING_METHOD: {DAMPING_METHOD}. Choose 'ls', 'logdec', 'hpp', 'constant', or 'mixed'."
    )

if DAMPING_METHOD != "mixed":
    print(f"  ζ = {xi_chosen}")

# Select frequencies for fitting
if FREQUENCY_SOURCE == "model":
    eigfreqs_fit = eigfreqs_model
    print(f"Using model eigenfrequencies: {eigfreqs_fit} Hz")
elif FREQUENCY_SOURCE == "experimental":
    eigfreqs_fit = eigfreqs_exp
    print(f"Using experimental eigenfrequencies: {eigfreqs_fit} Hz")
else:
    raise ValueError(
        f"Invalid FREQUENCY_SOURCE: {FREQUENCY_SOURCE}. Choose 'model' or 'experimental'."
    )

# Fit Rayleigh damping parameters
alpha, beta = fit_damping(xi=xi_chosen, eifreqs=eigfreqs_fit)

# Calculate damping matrix
D = alpha * M + beta * K

print(f"\nRayleigh damping coefficients:")
print(f"  α (mass-proportional) = {alpha:.6e}")
print(f"  β (stiffness-proportional) = {beta:.6e}")

# Calculate resulting damping ratios at model frequencies
xi_resulting = (
    alpha / (2 * eigfreqs_model * 2 * np.pi) + beta * (eigfreqs_model * 2 * np.pi) / 2
)
print(f"\nResulting damping ratios at model frequencies:")
print(f"  ζ = {xi_resulting}\n")


# FRF -- FREQUENCY RESPONSE FUNCTION
N = 4000  # number of points for plotting
N_factor = 500

# Pre-allocate arrays
w = np.zeros(N)
x11 = np.zeros(N, dtype=complex)
x21 = np.zeros(N, dtype=complex)
v11 = np.zeros(N, dtype=complex)
v21 = np.zeros(N, dtype=complex)
a11 = np.zeros(N, dtype=complex)
a21 = np.zeros(N, dtype=complex)

phase11 = np.zeros(N)
phase21 = np.zeros(N)
phase11a = np.zeros(N)
phase21a = np.zeros(N)

# Given Excitation Function acting on point 1 (force on first DOF only)
F_excitation_1 = np.zeros((N_DOF, N))  # Force vector for all DOFs
F_excitation_1[0, :] = 1.0  # [N] force amplitude on first DOF

# Calculation of the displacement, velocity and acceleration responses (Excitation on point 1)
for i in range(N):
    w[i] = 2 * np.pi * i / N_factor
    j = 1j

    # Dynamical Stiffness Matrix
    AA = (-M * w[i] ** 2 + K) + D * w[i] * j
    x = np.linalg.inv(AA).dot(F_excitation_1[:, i])

    # Receptance
    x11[i] = x[0]
    x21[i] = x[1]
    phase11[i] = np.arccos(np.real(x11[i]) / np.abs(x11[i]))
    phase21[i] = np.arccos(np.real(x21[i]) / np.abs(x21[i]))

    # Mobility
    v11[i] = j * w[i] * x[0]
    v21[i] = j * w[i] * x[1]

    # Inertance
    a11[i] = -w[i] ** 2 * x[0]
    a21[i] = -w[i] ** 2 * x[1]
    phase11a[i] = np.arccos(np.real(a11[i]) / np.abs(a11[i]))
    phase21a[i] = np.arccos(np.real(a21[i]) / np.abs(a21[i]))


# Given Excitation Function acting on point 2 (force on second DOF only)
F_excitation_2 = np.zeros((N_DOF, N))  # Force vector for all DOFs
F_excitation_2[1, :] = 1.0  # [N] force amplitude on second DOF

# Pre-allocate arrays for excitation 2
x12 = np.zeros(N, dtype=complex)
x22 = np.zeros(N, dtype=complex)
a12 = np.zeros(N, dtype=complex)
a22 = np.zeros(N, dtype=complex)

phase12 = np.zeros(N)
phase22 = np.zeros(N)
phase12a = np.zeros(N)
phase22a = np.zeros(N)

# Calculation of responses (Excitation on point 2)
for i in range(N):
    j = 1j
    AA = (-M * w[i] ** 2 + K) + D * w[i] * j
    x = np.linalg.inv(AA).dot(F_excitation_2[:, i])

    # Receptance
    x12[i] = x[0]
    x22[i] = x[1]
    phase12[i] = np.arccos(np.real(x12[i]) / np.abs(x12[i]))
    phase22[i] = np.arccos(np.real(x22[i]) / np.abs(x22[i]))

    # Inertance
    a12[i] = -w[i] ** 2 * x[0]
    a22[i] = -w[i] ** 2 * x[1]
    phase12a[i] = np.arccos(np.real(a12[i]) / np.abs(a12[i]))
    phase22a[i] = np.arccos(np.real(a22[i]) / np.abs(a22[i]))


# Frequency in Hz
freq_hz = w / (2 * np.pi)

# Plotting the results - RECEPTANCE
plt.figure(1, figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(freq_hz, np.abs(x11), "r", linewidth=1.5)
plt.title("Excitation on Point 1 and Response of Point 1")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_1(\omega)||    [m/N]")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(freq_hz, 180 / np.pi * phase11, "r", linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.axis([0, 8, 0, 180])
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freq_hz, np.abs(x21), "b-.", linewidth=1.5)
plt.title("Excitation on Point 1 and Response of Point 2")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_2(\omega)||    [m/N]")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(freq_hz, 180 / np.pi * phase21, "b-.", linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.axis([0, 8, 0, 180])
plt.grid(True)
plt.tight_layout()


plt.figure(2, figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(freq_hz, np.abs(x12), "r", linewidth=1.5)
plt.title("Excitation on Point 2 and Response of Point 1")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_1(\omega)||    [m/N]")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(freq_hz, 180 / np.pi * phase12, "r", linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.axis([0, 8, 0, 180])
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freq_hz, np.abs(x22), "b-.", linewidth=1.5)
plt.title("Excitation on Point 2 and Response of Point 2")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_2(\omega)||    [m/N]")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(freq_hz, 180 / np.pi * phase22, "b-.", linewidth=1.5)
plt.plot(
    freq_hz,
    -90 * np.ones_like(w),
    "k--",
    freq_hz,
    -270 * np.ones_like(w),
    "k--",
    linewidth=0.5,
)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.axis([0, 8, 0, 180])
plt.grid(True)
plt.tight_layout()


plt.figure(3, figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(freq_hz, np.abs(x11), "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, np.abs(x21), "b-.", label="point 2", linewidth=1.5)
plt.title("RECEPTANCE - Excitation on Point 1")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_i(\omega)||    [m/N]")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(freq_hz, 180 / np.pi * phase11, "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, 180 / np.pi * phase21, "b-.", label="point 2", linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.legend()
plt.axis([0, 8, 0, 180])
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freq_hz, np.abs(x12), "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, np.abs(x22), "b-.", label="point 2", linewidth=1.5)
plt.title("RECEPTANCE - Excitation on Point 2")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_i(\omega)||    [m/N]")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(freq_hz, 180 / np.pi * phase12, "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, 180 / np.pi * phase22, "b-.", label="point 2", linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.legend()
plt.axis([0, 8, 0, 180])
plt.grid(True)
plt.tight_layout()


plt.figure(4)
plt.plot(np.real(x11), np.imag(x11), "r", label="point 1", linewidth=1.5)
plt.plot(np.real(x21), np.imag(x21), "b-.", label="point 2", linewidth=1.5)
plt.title("FRF - Excitation on Point 1")
plt.xlabel(r"Real($y_i(\omega)/f_1(\omega)$)  (i=1,2)     [m/N]")
plt.ylabel(r"Imag($y_i(\omega)/f_1(\omega)$)  (i=1,2)     [m/N]")
plt.legend()
plt.grid(True)


fig5 = plt.figure(5)
ax = fig5.add_subplot(111, projection="3d")
ax.plot(freq_hz, np.real(x11), np.imag(x11), "r", label="point 1", linewidth=0.5)
ax.plot(freq_hz, np.real(x21), np.imag(x21), "b-.", label="point 2", linewidth=0.5)

# Projections
min_imag_x21 = np.min(np.imag(x21))
max_real_x21 = np.max(np.real(x21))

ax.plot(
    freq_hz,
    np.real(x11),
    2 * min_imag_x21 * np.ones_like(np.imag(x11)),
    "r-",
    linewidth=1.5,
)
ax.plot(
    freq_hz,
    np.real(x21),
    2 * min_imag_x21 * np.ones_like(np.imag(x21)),
    "b-.",
    linewidth=1.5,
)

ax.plot(
    freq_hz,
    2 * max_real_x21 * np.ones_like(np.real(x11)),
    np.imag(x11),
    "r-",
    linewidth=1.5,
)
ax.plot(
    freq_hz,
    2 * max_real_x21 * np.ones_like(np.real(x11)),
    np.imag(x21),
    "b-.",
    linewidth=1.5,
)

ax.set_title("FRF - Excitation on Point 1")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"Real($y_i(\omega)/f_1(\omega)$)    (i=1,2)      [m/N]")
ax.set_zlabel(r"Imag($y_i(\omega)/f_1(\omega)$)    (i=1,2)      [m/N]")
ax.legend()
ax.grid(True)


# Plotting the results - INERTANCE
plt.figure(6, figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(freq_hz, np.abs(a11), "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, np.abs(a21), "b-.", label="point 2", linewidth=1.5)
plt.title(" INERTANCE - Excitation on Point 1")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_i(\omega)||    [(m/s^2)/N]")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(freq_hz, 180 / np.pi * phase11a, "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, 180 / np.pi * phase21a, "b-.", label="point 2", linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.legend()
plt.axis([0, 8, 0, 180])
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freq_hz, np.abs(a12), "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, np.abs(a22), "b-.", label="point 2", linewidth=1.5)
plt.title("INERTANCE - Excitation on Point 2")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"||y_i(\omega)||    [(m/s^2)/N]")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(freq_hz, 180 / np.pi * phase12a, "r", label="point 1", linewidth=1.5)
plt.plot(freq_hz, 180 / np.pi * phase22a, "b-.", label="point 2", linewidth=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase        [ $^o$]")
plt.legend()
plt.axis([0, 8, 0, 180])
plt.grid(True)
plt.tight_layout()

plt.show()
