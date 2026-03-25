#########################################################
# 41514 DYNAMICS OF MACHINERY                           #
# MEK - DEPARTMENT OF MECHANICAL ENGINEERING            #
# DTU - TECHNICAL UNIVERSITY OF DENMARK                 #
#                                                       #
#  QUESTION 7: THEORETICAL VS EXPERIMENTAL FRF COMPARISON#
#                                                       #
#  - Calculate theoretical FRFs for all 6 DOFs         #
#  - Load experimental FRFs from EMA analysis          #
#  - Compare and plot both on same figure              #
#  - Justify discrepancies                             #
#########################################################

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, lfilter, csd as _csd, welch as _welch

# ═══════════════════════════════════════════════════════
# PART 1: THEORETICAL MODEL - Calculate FRFs
# ═══════════════════════════════════════════════════════

# Model Parameters (same as question_7.py)
l1 = 0.193
l2 = 0.292
l3 = 0.220
l4 = 0.271
l5 = 0.430
l6 = 0.430

# m1 = 2.294
# m2 = 1.941
# m3 = 1.943
# m4 = 2.732
# m5 = 0.774
# m6 = 0.774
m1 = 2.4087
m2 = 1.8440
m3 = 1.8459
m4 = 2.5954
m5 = 0.7353
m6 = 0.7353
g = 9.81
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
k1 = 4695.4364
k2 = 1430.8602
k3 = 1275.9561
k4 = 895.1968
k5 = 85.4288
k6 = 85.4288
N_DOF = 6

# Mass matrix
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

# Stiffness matrix
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


# Damping matrix calculation functions
def modal_analysis(M, K):
    """Calculate eigenfrequencies and mode shapes."""
    from scipy.linalg import eig

    n_dof = M.shape[0]

    zero_mat = np.zeros((n_dof, n_dof))
    A = np.block([[M, zero_mat], [zero_mat, M]])
    B = np.block([[zero_mat, K], [-M, zero_mat]])

    eigval, eigvec = eig(-B, A)
    eigfreqs = np.abs(eigval) / (2 * np.pi)

    idx_sorted = np.argsort(eigfreqs)
    eigfreqs = eigfreqs[idx_sorted]
    eigval = eigval[idx_sorted]
    eigvec = eigvec[:, idx_sorted]

    idx = N_DOF + np.argmax(np.abs(eigvec[n_dof:]), axis=0)
    scale = eigvec[idx, np.arange(eigvec.shape[1])]
    eigvec_norm = np.round(eigvec / scale, 4)

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

# Load experimental data for damping
res_exp = pd.read_csv(Path(__file__).parent / "EMA_global_comparison_summary.csv")
eigfreqs_exp = res_exp["fn_freq_Hz"].values
xi_exp_ls = res_exp["zeta_ls_ref_acc"].values
xi_exp_logdec = res_exp["zeta_time_logdec"].values

# Use custom damping methods
DAMPING_METHOD = "custom"
FREQUENCY_SOURCE = "experimental"

xi_chosen = np.array([0.00553, 0.00419, 0.00353, 0.00292, 0.00334, 0.00379])

eigfreqs_fit = eigfreqs_exp if FREQUENCY_SOURCE == "experimental" else eigfreqs_model

alpha = 0.07682
beta = 0.000115

print("\n" + "=" * 70)
print("  DAMPING RATIOS USED FOR RAYLEIGH DAMPING MATRIX")
print("=" * 70)
print("Using previously adjusted D matrix from model validation:")
print(f"\nFrequency source: {FREQUENCY_SOURCE}")

# Load damping matrix
D = np.load(Path(__file__).parent / "D_adjusted.npy")

print("\n" + "=" * 70)
print("  THEORETICAL MODEL PARAMETERS")
print("=" * 70)
print(f"Rayleigh damping coefficients:")
print(f"  alpha (mass-proportional) = {alpha:.6e}")
print(f"  beta (stiffness-proportional) = {beta:.6e}")
print(f"\nTheoretical natural frequencies [Hz]:")
for i, fn in enumerate(eigfreqs_model):
    print(f"  Mode {i+1}: {fn:.4f} Hz")

print("\n" + "=" * 70)
print("  MODE SHAPES (normalized)")
print("=" * 70)
print("Mass/DOF  |  Mode 1   Mode 2   Mode 3   Mode 4   Mode 5   Mode 6")
print("-" * 70)
for dof in range(N_DOF):
    row = f"Mass {dof+1}   | "
    for mode in range(N_DOF):
        row += f"{modeshapes[dof, mode]:8.4f} "
    print(row)
print("\nNOTE: If Mode 2 has near-zero amplitude at Mass 1,")
print("      it will not be excited by force on Mass 1!")

# ═══════════════════════════════════════════════════════
# CALCULATE THEORETICAL FRFs - ALL 6 DOFs
# Excitation on mass 1 (DOF 1), measure all 6 accelerations
# ═══════════════════════════════════════════════════════

N_freq = 6000  # number of frequency points
N_factor = 500  # frequency scaling factor

# Frequency vector
w = np.zeros(N_freq)
for i in range(N_freq):
    w[i] = 2 * np.pi * i / N_factor

freq_hz = w / (2 * np.pi)

# Pre-allocate arrays for all 6 DOFs
# Inertance FRFs: acceleration / force [m/s²/N]
H_theoretical = np.zeros((N_DOF, N_freq), dtype=complex)

# Excitation on DOF 1 (mass 1 - the lowest mass)
F_excitation = np.zeros((N_DOF, N_freq))
F_excitation[0, :] = 1.0  # 1 N force on first DOF

print("\nCalculating theoretical FRFs...")
for i in range(N_freq):
    j = 1j
    # Dynamic stiffness matrix
    AA = (-M * w[i] ** 2 + K) + D * w[i] * j

    # Solve for displacement: AA * x = F
    x = np.linalg.inv(AA).dot(F_excitation[:, i])

    # Convert displacement to acceleration (inertance)
    # a = -ω² * x
    for dof in range(N_DOF):
        H_theoretical[dof, i] = -w[i] ** 2 * x[dof]

print("Theoretical FRF calculation complete.")

# ═══════════════════════════════════════════════════════
# PART 2: LOAD EXPERIMENTAL FRFs
# ═══════════════════════════════════════════════════════

print("\nLoading experimental FRF data...")

# Parameters from EMA_frequency_domain.py
BL = 25
Fs = 50
OVLP = 1000
N_ACC = 6

# Load experimental data
file = "data/Downsweep50Hz.txt"
data = pd.read_csv(file, delimiter=";")

x_raw = data["Force"].values

# Butterworth low-pass filter
b_f, a_f = butter(5, 0.99)
x_filt = lfilter(b_f, a_f, x_raw)

# Compute FRF (H2 estimator) for all 6 accelerometers
N_block = round(len(x_filt) / BL)
WIN = N_block
ovlp = min(OVLP, WIN - 1)

H_experimental = np.zeros((N_ACC, WIN // 2 + 1), dtype=complex)

for i in range(N_ACC):
    y_raw = data.iloc[:, i + 1].values
    y_filt = lfilter(b_f, a_f, y_raw)

    _, PXY = _csd(
        x_filt, y_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN
    )
    _, PYX = _csd(
        y_filt, x_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN
    )
    _, PYY = _welch(y_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN)

    H_experimental[i] = PYY / PYX

F_exp = np.fft.rfftfreq(WIN, d=1.0 / Fs)
freq_idx = slice(0, WIN // 2)
F_exp = F_exp[freq_idx]

print("Experimental FRF data loaded.")

# ═══════════════════════════════════════════════════════
# PART 3: PLOT COMPARISON - THEORETICAL VS EXPERIMENTAL
# ═══════════════════════════════════════════════════════

colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
mass_labels = [f"Mass {i+1}" for i in range(N_DOF)]

# Create 1 figure with 6 subplots (3 rows, 2 columns)
fig_frf_all, axes_all = plt.subplots(3, 2, figsize=(24, 16))
axes_flat = axes_all.flatten()

for i in range(N_DOF):
    ax = axes_flat[i]

    # Plot experimental FRF (H2)
    ax.semilogy(
        F_exp,
        np.abs(H_experimental[i, freq_idx]),
        color=colors[i],
        linewidth=1.5,
        label="Experimental (H2)",
        alpha=0.7,
    )

    # Plot theoretical FRF
    ax.semilogy(
        freq_hz,
        np.abs(H_theoretical[i, :]),
        "k--",
        linewidth=1.2,
        label="Theoretical",
        alpha=0.8,
    )

    ax.set_xlim([0.5, 10])
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Frequency [Hz]", fontsize=20)
    ax.set_ylabel("|H| [m/s²/N]", fontsize=20)
    ax.set_title(f"{mass_labels[i]}", fontsize=24, fontweight="bold")
    ax.legend(fontsize=16, loc="best")

    # Increase tick label size
    ax.tick_params(axis="both", which="major", labelsize=18)

    # Add vertical lines at theoretical natural frequencies with frequency labels
    for fn in eigfreqs_model:
        ax.axvline(fn, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        # Add frequency label below the line
        ax.text(
            fn,
            ax.get_ylim()[0] * 1.5,
            f"{fn:.1f} Hz",
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right",
            color="gray",
            fontsize=16,
            alpha=0.8,
        )

fig_frf_all.tight_layout()

# ═══════════════════════════════════════════════════════
# PART 4: OVERLAY PLOT - ALL 6 DOFs ON ONE FIGURE
# ═══════════════════════════════════════════════════════

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top plot: Experimental FRFs
for i in range(N_DOF):
    ax1.semilogy(
        F_exp,
        np.abs(H_experimental[i, freq_idx]),
        color=colors[i],
        linewidth=1.5,
        label=mass_labels[i],
        alpha=0.7,
    )

ax1.set_xlim([0.5, 10])
ax1.grid(True, which="both", alpha=0.3)
ax1.set_ylabel("|H| [m/s²/N]", fontsize=18)
ax1.legend(fontsize=13, ncol=3, loc="upper right")
ax1.tick_params(axis="both", which="major", labelsize=15)

# Set x-axis major ticks to 0.1 Hz
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))

# Add vertical lines at experimental natural frequencies
for fn in eigfreqs_exp:
    ax1.axvline(fn, color="red", linestyle=":", linewidth=0.8, alpha=0.6)

# Bottom plot: Theoretical FRFs
for i in range(N_DOF):
    ax2.semilogy(
        freq_hz,
        np.abs(H_theoretical[i, :]),
        color=colors[i],
        linewidth=1.5,
        label=mass_labels[i],
        alpha=0.7,
    )

ax2.set_xlim([0.5, 10])
ax2.grid(True, which="both", alpha=0.3)
ax2.set_xlabel("Frequency [Hz]", fontsize=18)
ax2.set_ylabel("|H| [m/s²/N]", fontsize=18)
ax2.legend(fontsize=13, ncol=3, loc="upper right")
ax2.tick_params(axis="both", which="major", labelsize=15)

# Set x-axis major ticks to 0.1 Hz
ax2.xaxis.set_major_locator(MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))

# Add vertical lines at theoretical natural frequencies with frequency labels
for fn in eigfreqs_model:
    ax2.axvline(fn, color="blue", linestyle=":", linewidth=0.8, alpha=0.6)
    # Add frequency label below the blue line
    ax2.text(
        fn,
        ax2.get_ylim()[0] * 1.5,
        f"{fn:.1f} Hz",
        rotation=90,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="blue",
        fontsize=13,
        alpha=0.8,
    )

fig2.tight_layout()

# ═══════════════════════════════════════════════════════
# PART 5: QUANTITATIVE COMPARISON
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  COMPARISON: THEORETICAL VS EXPERIMENTAL NATURAL FREQUENCIES")
print("=" * 70)
print(
    f"{'Mode':>6}  {'Theoretical [Hz]':>18}  {'Experimental [Hz]':>18}  {'Error [%]':>12}"
)
print("-" * 70)

for i in range(N_DOF):
    fn_theo = eigfreqs_model[i]
    fn_exp = eigfreqs_exp[i]
    error_pct = abs(fn_theo - fn_exp) / fn_exp * 100
    print(f"{i+1:>6}  {fn_theo:>18.4f}  {fn_exp:>18.4f}  {error_pct:>12.2f}")

# ═══════════════════════════════════════════════════════
# PART 6: FORCE AMPLIFICATION AT NATURAL FREQUENCIES
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  FORCE AMPLIFICATION AT NATURAL FREQUENCIES (THEORETICAL)")
print("=" * 70)
print("  Natural Freq [Hz] and FRF Magnitude [m/s²/N] at each DOF")
print("-" * 70)

# Header
header = f"{'Mode':>6}  {'Freq [Hz]':>10} |"
for dof in range(N_DOF):
    header += f" Mass {dof+1:1d} |"
print(header)
print("-" * 70)

# For each natural frequency (mode)
for mode in range(N_DOF):
    fn = eigfreqs_model[mode]

    # Find closest frequency index in theoretical FRF
    freq_idx_theo = np.argmin(np.abs(freq_hz - fn))

    row = f"{mode+1:>6}  {fn:>10.4f} |"

    # Get FRF magnitude at this frequency for each DOF
    for dof in range(N_DOF):
        mag = np.abs(H_theoretical[dof, freq_idx_theo])
        row += f" {mag:>6.2f} |"

    print(row)

print("\n" + "=" * 70)
print("  FORCE AMPLIFICATION AT NATURAL FREQUENCIES (EXPERIMENTAL)")
print("=" * 70)
print("  Natural Freq [Hz] and FRF Magnitude [m/s²/N] at each DOF")
print("-" * 70)

# Header
header = f"{'Mode':>6}  {'Freq [Hz]':>10} |"
for dof in range(N_DOF):
    header += f" Mass {dof+1:1d} |"
print(header)
print("-" * 70)

# For each natural frequency (mode)
for mode in range(N_DOF):
    fn = eigfreqs_exp[mode]

    # Find closest frequency index in experimental FRF
    freq_idx_exp = np.argmin(np.abs(F_exp - fn))

    row = f"{mode+1:>6}  {fn:>10.4f} |"

    # Get FRF magnitude at this frequency for each DOF
    for dof in range(N_DOF):
        mag = np.abs(H_experimental[dof, freq_idx_exp])
        row += f" {mag:>6.2f} |"

    print(row)

print("\nNOTE: Values show acceleration response magnitude [m/s²/N]")
print("      Higher values = greater amplification at that natural frequency")
print("      Diagonal dominance expected: each mode amplifies specific masses")

# ═══════════════════════════════════════════════════════
# PART 7: EXTRACT EXPERIMENTAL MODE SHAPES FROM FRFs
# ═══════════════════════════════════════════════════════

# Extract experimental mode shapes from FRF peaks
# At each natural frequency, the FRF magnitude gives mode shape amplitude
modeshapes_exp = np.zeros((N_DOF, N_DOF))

for mode in range(N_DOF):
    fn = eigfreqs_exp[mode]
    # Find closest frequency index in experimental FRF
    freq_idx_exp = np.argmin(np.abs(F_exp - fn))

    # Get the imaginary part of H2 at this frequency for all DOFs
    # (imaginary part at resonance gives mode shape with correct sign)
    for dof in range(N_DOF):
        modeshapes_exp[dof, mode] = np.imag(H_experimental[dof, freq_idx_exp])

    # Normalize to match the reference accelerometer sign convention
    ref_acc_idx = 3  # Mass 4 is the reference (index 3)
    ref_sign = np.sign(modeshapes_exp[ref_acc_idx, mode])
    if ref_sign == 0:
        ref_sign = 1.0

    # Normalize to max amplitude = 1 with correct sign
    max_amplitude = np.max(np.abs(modeshapes_exp[:, mode]))
    modeshapes_exp[:, mode] = (modeshapes_exp[:, mode] / max_amplitude) * ref_sign

print("\n" + "=" * 70)
print("  EXPERIMENTAL MODE SHAPES EXTRACTED FROM FRF PEAKS")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# PART 8: PLOT MODE SHAPES - COMPARISON (NORMALIZED TO MAX = 1)
# ═══════════════════════════════════════════════════════

# Normalize theoretical mode shapes so max amplitude = 1 for each mode
modeshapes_normalized = np.zeros_like(modeshapes)
for mode in range(N_DOF):
    max_amplitude = np.max(np.abs(modeshapes[:, mode]))
    modeshapes_normalized[:, mode] = modeshapes[:, mode] / max_amplitude

# Flip sign for modes 2, 3, and 5 (indices 1, 2, 4)
modes_to_flip = [1, 2, 4]  # Mode 2, 3, 5
for mode_idx in modes_to_flip:
    modeshapes_normalized[:, mode_idx] *= -1

# Define physical positions for masses using actual lengths
# y-coordinate represents vertical height, x=0 is the centerline
# Calculate cumulative heights based on beam lengths
h1 = l1  # Mass 1 at height l1 from floor
h2 = l1 + l2  # Mass 2 at height l1 + l2
h3 = l1 + l2 + l3  # Mass 3
h4 = l1 + l2 + l3 + l4  # Mass 4
h5 = l1 + l2 + l3 + l4 - l5  # Mass 5 hangs BELOW mass 4 by length l5
h6 = l1 + l2 + l3 + l4 + l6  # Mass 6 extends ABOVE mass 4 by length l6

mass_coords = {
    1: (0, h1),  # Mass 1
    2: (0, h2),  # Mass 2
    3: (0, h3),  # Mass 3
    4: (0, h4),  # Mass 4 (connection point for 5 and 6)
    5: (0, h5),  # Mass 5 - below mass 4
    6: (0, h6),  # Mass 6 - above mass 4
}

# Define connections (which masses are connected by springs)
connections = [
    (1, 2),  # Spring k1 connects mass 1 to mass 2
    (2, 3),  # Spring k2 connects mass 2 to mass 3
    (3, 4),  # Spring k3 connects mass 3 to mass 4
    (4, 5),  # Spring k4 connects mass 4 to mass 5
    (4, 6),  # Spring k5/k6 connects mass 4 to mass 6
]

# Create figure for mode shapes comparison
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
axes3 = axes3.flatten()

for mode in range(N_DOF):
    ax = axes3[mode]

    # Get equilibrium positions for all masses
    y_coords = [mass_coords[i + 1][1] for i in range(N_DOF)]

    # Mode shape amplitude represents horizontal displacement
    x_exp = modeshapes_exp[:, mode]
    x_theo = modeshapes_normalized[:, mode]

    # Draw the floor at y=0
    floor_width = 0.3  # Width of floor representation
    ax.plot(
        [-floor_width, floor_width],
        [0, 0],
        color="black",
        linewidth=5,
        solid_capstyle="butt",
        zorder=1,
    )
    # Add hatching to represent ground
    ax.fill_between(
        [-floor_width, floor_width],
        0,
        -0.05,
        color="gray",
        alpha=0.3,
        hatch="///",
        edgecolor="black",
        linewidth=1,
        zorder=1,
    )

    # Draw connection from floor to Mass 1 (undeformed structure)
    ax.plot(
        [0, 0],
        [0, mass_coords[1][1]],
        color="lightgray",
        linewidth=3,
        alpha=0.5,
        zorder=1,
    )

    # Draw the undeformed structure (equilibrium position) for the rest
    for conn in connections:
        m1, m2 = conn
        y1, y2 = mass_coords[m1][1], mass_coords[m2][1]
        ax.plot([0, 0], [y1, y2], color="lightgray", linewidth=3, alpha=0.5, zorder=1)

    # Draw vertical reference line
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)

    # Draw connection from floor to Mass 1 (deformed - experimental)
    ax.plot(
        [0, x_exp[0]],
        [0, y_coords[0]],
        color=colors[mode],
        linewidth=2,
        alpha=0.6,
        linestyle="-",
        zorder=2,
    )

    # Draw connection from floor to Mass 1 (deformed - theoretical)
    ax.plot(
        [0, x_theo[0]],
        [0, y_coords[0]],
        color="black",
        linewidth=2,
        alpha=0.7,
        linestyle=":",
        zorder=2,
    )

    # Draw deformed structure for experimental (solid lines)
    for conn in connections:
        m1, m2 = conn
        idx1, idx2 = m1 - 1, m2 - 1
        ax.plot(
            [x_exp[idx1], x_exp[idx2]],
            [y_coords[idx1], y_coords[idx2]],
            color=colors[mode],
            linewidth=2,
            alpha=0.6,
            linestyle="-",
            zorder=2,
        )

    # Draw deformed structure for theoretical (dotted lines)
    for conn in connections:
        m1, m2 = conn
        idx1, idx2 = m1 - 1, m2 - 1
        ax.plot(
            [x_theo[idx1], x_theo[idx2]],
            [y_coords[idx1], y_coords[idx2]],
            color="black",
            linewidth=2,
            alpha=0.7,
            linestyle=":",
            zorder=2,
        )

    # Draw blue lines connecting experimental to digital twin for each mass (shows difference)
    for i in range(N_DOF):
        ax.plot(
            [x_exp[i], x_theo[i]],
            [y_coords[i], y_coords[i]],
            color="blue",
            linewidth=1.5,
            alpha=0.6,
            zorder=2,
        )

    # Plot experimental mode shape (filled circles)
    ax.scatter(
        x_exp,
        y_coords,
        s=200,
        marker="o",
        color=colors[mode],
        edgecolor="k",
        linewidth=2,
        alpha=0.7,
        label="Experimental",
        zorder=3,
    )

    # Plot theoretical mode shape (filled circles with different edge)
    ax.scatter(
        x_theo,
        y_coords,
        s=200,
        marker="o",
        facecolors="white",
        edgecolor="k",
        linewidth=2.5,
        alpha=0.9,
        label="Digital Twin",
        zorder=4,
    )

    ax.grid(True, alpha=0.3, axis="x")

    # Set y-ticks to show mass labels at their heights
    ax.set_yticks(y_coords)
    ax.set_yticklabels([f"M{i+1}" for i in range(N_DOF)], fontsize=15)
    ax.tick_params(axis="x", which="major", labelsize=14)

    ax.set_xlabel("Normalized Amplitude (Displacement)", fontsize=16)
    ax.set_ylabel("Height [m]", fontsize=16)
    ax.set_title(f"Mode {mode+1}", fontsize=18, fontweight="bold")
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-0.08, 1.5])
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)

fig3.tight_layout()

# ═══════════════════════════════════════════════════════
# SAVE FIGURES TO PDF
# ═══════════════════════════════════════════════════════

# Create figures folder if it doesn't exist
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

# Save both figures
fig_frf_all.savefig(
    figures_dir / "FRF_comparison_individual_all.pdf",
    dpi=300,
    bbox_inches="tight",
)

fig2.savefig(figures_dir / "FRF_comparison_overlay.pdf", dpi=300, bbox_inches="tight")
fig3.savefig(figures_dir / "mode_shapes_normalized.pdf", dpi=300, bbox_inches="tight")

print("\n" + "=" * 70)
print("  FIGURES SAVED")
print("=" * 70)
print(f"Figures saved to: {figures_dir}")
print("  - FRF_comparison_individual_all.pdf")
print("  - FRF_comparison_overlay.pdf")
print("  - mode_shapes_normalized.pdf")

plt.show()
