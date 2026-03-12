#########################################################
# 41514 DYNAMICS OF MACHINERY                           #
# MEK - DEPARTMENT OF MECHANICAL ENGINEERING            #
# DTU - TECHNICAL UNIVERSITY OF DENMARK                 #
#                                                       #
#  EMA – PHASE 1 & 2: FREQUENCY DOMAIN ANALYSIS        #
#                                                       #
#  Phase 1: FRF + Coherence for all 6 accelerometers   #
#           Identification of 6 natural frequencies     #
#  Phase 2: Modal parameter extraction per mode         #
#           Half-power bandwidth method (1-DOF model)   #
#           ωn, ξ, m_modal, d_modal, k_modal            #
#           Experimental mode shapes from Im(H1)        #
#########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import (
    butter,
    lfilter,
    csd as _csd,
    welch as _welch,
    coherence as _coherence,
    find_peaks,
    peak_prominences,
)

# ─────────────────────────────────────────────────────
# PARAMETERS (must match acquisition system)
# ─────────────────────────────────────────────────────
BL = 25  # number of blocks
Fs = 50  # sampling frequency [Hz]
OVLP = 1000  # overlap in samples (clipped to WIN-1 if necessary)
N_ACC = 6  # number of accelerometers / DOFs

# ─────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────
file = "data/Downsweep50Hz.txt"
data = pd.read_csv(file, delimiter=";")

x_raw = data["Force"].values

# ─────────────────────────────────────────────────────
# BUTTERWORTH LOW-PASS FILTER  (5th order, Wn = 0.99)
# cut-off ≈ 0.99 × Nyquist = 24.75 Hz
# ─────────────────────────────────────────────────────
b_f, a_f = butter(5, 0.99)
x_filt = lfilter(b_f, a_f, x_raw)

# ─────────────────────────────────────────────────────
# COMPUTE FRF (H1, H2) AND COHERENCE FOR ALL 6 ACC
# ─────────────────────────────────────────────────────
N_block = round(len(x_filt) / BL)  # samples per block
WIN = N_block
ovlp = min(OVLP, WIN - 1)  # safety clip

H1 = np.zeros((N_ACC, WIN // 2 + 1), dtype=complex)
H2 = np.zeros_like(H1)
Coh = np.zeros((N_ACC, WIN // 2 + 1))

for i in range(N_ACC):
    y_raw = data.iloc[:, i + 1].values  # columns 1-6 → acc1-acc6
    y_filt = lfilter(b_f, a_f, y_raw)

    _, PXY = _csd(
        x_filt, y_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN
    )
    _, PYX = _csd(
        y_filt, x_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN
    )
    _, PXX = _welch(x_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN)
    _, PYY = _welch(y_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN)
    F, Cy = _coherence(
        x_filt, y_filt, fs=Fs, window="hann", nperseg=WIN, noverlap=ovlp, nfft=WIN
    )

    H1[i] = PXY / PXX
    H2[i] = PYY / PYX
    Coh[i] = Cy

# Frequency axis limited to 0 – Fs/2
freq_idx = slice(0, WIN // 2)
F_p = F[freq_idx]  # plotting frequency vector

# ═══════════════════════════════════════════════════════
# PHASE 1 – PLOT ALL FRFs AND COHERENCES
# ═══════════════════════════════════════════════════════
colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
acc_names = [f"Acc {i+1}" for i in range(N_ACC)]

fig1, (ax_frf, ax_coh) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for i in range(N_ACC):
    ax_frf.semilogy(
        F_p, np.abs(H2[i, freq_idx]), color=colors[i], linewidth=1.2, label=acc_names[i]
    )
    ax_coh.plot(
        F_p, Coh[i, freq_idx], color=colors[i], linewidth=1.0, label=acc_names[i]
    )

ax_frf.set_xlim([0.5, Fs / 2])
ax_frf.grid(True, which="both")
ax_frf.set_ylabel("|H2| [(m/s²)/N]", fontsize=13)
ax_frf.set_title("Phase 1 – FRF Magnitude (H2) – All 6 Accelerometers", fontsize=13)
ax_frf.legend(fontsize=10)

ax_coh.set_xlim([0.5, Fs / 2])
ax_coh.set_ylim([0, 1.1])
ax_coh.grid(True)
ax_coh.set_ylabel("Coherence γ²(f)", fontsize=13)
ax_coh.set_xlabel("Frequency [Hz]", fontsize=13)
ax_coh.legend(fontsize=10)

fig1.tight_layout()

# ═══════════════════════════════════════════════════════
# PHASE 1 – IDENTIFY 6 NATURAL FREQUENCIES
# Composite FRF = sum of |H2|² across all DOFs to capture
# every mode regardless of which accelerometer catches it.
# ═══════════════════════════════════════════════════════
H2_composite = np.sum(np.abs(H2[:, freq_idx]) ** 2, axis=0)
H2_composite /= H2_composite.max()  # normalise for threshold

df = F_p[1] - F_p[0]  # frequency resolution [Hz]
min_dist = max(3, int(0.3 / df))  # at least 0.3 Hz between peaks

all_peaks, _ = find_peaks(
    H2_composite, distance=min_dist, height=H2_composite.max() * 0.002
)

# Keep the 6 most prominent peaks
proms, _, _ = peak_prominences(H2_composite, all_peaks)
top_idx = np.argsort(proms)[::-1][:N_ACC]
top6_peaks = np.sort(all_peaks[top_idx])  # sorted by frequency

fn_exp = F_p[top6_peaks]  # natural frequencies [Hz]
wn_exp = 2 * np.pi * fn_exp  # [rad/s]

print("\n" + "=" * 58)
print("  PHASE 1 – IDENTIFIED NATURAL FREQUENCIES")
print("=" * 58)
print(f"  {'Mode':>4}  {'fn [Hz]':>10}  {'ωn [rad/s]':>12}")
print("  " + "-" * 30)
for k, (fn, wn) in enumerate(zip(fn_exp, wn_exp), 1):
    print(f"  {k:>4}  {fn:>10.4f}  {wn:>12.4f}")

# ═══════════════════════════════════════════════════════
# PHASE 2 – MODAL PARAMETERS  (half-power bandwidth)
# ═══════════════════════════════════════════════════════


def half_power_bw(F_arr, H_mag, pk_idx):
    """
    Return (f1, f2, fn, xi) using the −3 dB (half-power) bandwidth.
    Linear interpolation is used between frequency bins.
    """
    Hpk = H_mag[pk_idx]
    fn = F_arr[pk_idx]
    half = Hpk / np.sqrt(2.0)

    # Left crossing
    left = pk_idx
    while left > 0 and H_mag[left] > half:
        left -= 1
    if left == 0:
        f1 = F_arr[0]
    else:
        slope = H_mag[left + 1] - H_mag[left]
        f1 = (
            F_arr[left] + (half - H_mag[left]) / slope * df
            if abs(slope) > 1e-30
            else F_arr[left]
        )

    # Right crossing
    right = pk_idx
    while right < len(H_mag) - 1 and H_mag[right] > half:
        right += 1
    if right == len(H_mag) - 1:
        f2 = F_arr[-1]
    else:
        slope = H_mag[right] - H_mag[right - 1]
        f2 = (
            F_arr[right - 1] + (half - H_mag[right - 1]) / slope * df
            if abs(slope) > 1e-30
            else F_arr[right]
        )

    xi = (f2 - f1) / (2.0 * fn)
    return f1, f2, fn, xi


modal_results = []
mode_shapes = np.zeros((N_ACC, N_ACC))  # [dof, mode]

print("\n" + "=" * 78)
print("  PHASE 2 – MODAL PARAMETERS  (half-power bandwidth / 1-DOF model)")
print("=" * 78)
print(
    f"  {'Mode':>4}  {'fn [Hz]':>8}  {'ξ [-]':>9}  "
    f"{'m_modal [kg]':>13}  {'d_modal [Ns/m]':>15}  {'k_modal [N/m]':>14}"
)
print("  " + "-" * 72)

for mi, pk_idx in enumerate(top6_peaks):
    # Use the DOF with the largest response at this mode for bandwidth
    dof_ref = int(np.argmax(np.abs(H2[:, freq_idx][:, pk_idx])))
    H_ref = np.abs(H2[dof_ref, freq_idx])

    f1, f2, fn_bw, xi = half_power_bw(F_p, H_ref, pk_idx)
    wn = 2.0 * np.pi * fn_bw
    Hpk = H_ref[pk_idx]

    # 1-DOF identification (eq. 5–8 from manual):
    #   |H(ωn)| = 1 / (2 ξ ωn² m)
    m_modal = 1.0 / (2.0 * xi * wn**2 * Hpk)
    k_modal = m_modal * wn**2
    d_modal = 2.0 * xi * m_modal * wn

    modal_results.append(
        dict(mode=mi + 1, fn=fn_bw, xi=xi, m=m_modal, d=d_modal, k=k_modal)
    )

    # Mode shape: Im(H1_j(fn)) ∝ −φ_j  → sign information preserved
    for j in range(N_ACC):
        mode_shapes[j, mi] = np.imag(H1[j, freq_idx][pk_idx])

    print(
        f"  {mi+1:>4}  {fn_bw:>8.4f}  {xi:>9.5f}  "
        f"{m_modal:>13.4f}  {d_modal:>15.5f}  {k_modal:>14.4f}"
    )

# Normalise mode shapes column-wise (reference: DOF with max |amplitude|)
for mi in range(N_ACC):
    col = mode_shapes[:, mi]
    i_ref = int(np.argmax(np.abs(col)))
    if abs(col[i_ref]) > 0:
        mode_shapes[:, mi] = col / col[i_ref]  # signed normalisation

print("\n" + "=" * 62)
print("  PHASE 2 – EXPERIMENTAL MODE SHAPES  (normalised)")
print("=" * 62)
dof_labels = [f"Mass {i+1}" for i in range(N_ACC)]
header = f"  {'DOF':<8}" + " ".join(f"{'Mode '+str(m+1):>10}" for m in range(N_ACC))
print(header)
print("  " + "-" * (len(header) - 2))
for dof in range(N_ACC):
    row = f"  {dof_labels[dof]:<8}" + " ".join(
        f"{mode_shapes[dof, m]:>+10.4f}" for m in range(N_ACC)
    )
    print(row)

# ═══════════════════════════════════════════════════════
# PHASE 2 – PLOT FRF WITH RESONANCE MARKERS
# ═══════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for i in range(N_ACC):
    axes2[0].semilogy(
        F_p, np.abs(H1[i, freq_idx]), color=colors[i], linewidth=1.0, label=acc_names[i]
    )
    axes2[1].semilogy(
        F_p, np.abs(H2[i, freq_idx]), color=colors[i], linewidth=1.0, label=acc_names[i]
    )
    axes2[2].plot(
        F_p, Coh[i, freq_idx], color=colors[i], linewidth=1.0, label=acc_names[i]
    )

for ax in (axes2[0], axes2[1]):
    for fn in fn_exp:
        ax.axvline(fn, color="gray", linestyle=":", linewidth=0.9)
    ax.grid(True, which="both")
    ax.set_xlim([0.5, Fs / 2])
    ax.legend(fontsize=9)

axes2[0].set_ylabel("|H1| [(m/s²)/N]", fontsize=12)
axes2[0].set_title("Phase 2 – FRF H1 with resonance markers", fontsize=12)
axes2[1].set_ylabel("|H2| [(m/s²)/N]", fontsize=12)
axes2[1].set_title("Phase 2 – FRF H2 with resonance markers", fontsize=12)

axes2[2].set_xlim([0.5, Fs / 2])
axes2[2].set_ylim([0, 1.1])
axes2[2].grid(True)
axes2[2].set_ylabel("Coherence", fontsize=12)
axes2[2].set_xlabel("Frequency [Hz]", fontsize=12)
axes2[2].legend(fontsize=9)

fig2.tight_layout()

# ═══════════════════════════════════════════════════════
# PHASE 2 – PLOT EXPERIMENTAL MODE SHAPES
# ═══════════════════════════════════════════════════════
height_positions = list(range(1, N_ACC + 1))  # DOF 1 = base, DOF 6 = tip
mass_labels = [
    "Platform 1",
    "Platform 2",
    "Platform 3",
    "Platform 4",
    "Blade 5",
    "Blade 6",
]

fig3, axes3 = plt.subplots(2, 3, figsize=(14, 9))
axes3 = axes3.flatten()

for mi in range(N_ACC):
    ax = axes3[mi]
    phi = mode_shapes[:, mi]

    ax.barh(height_positions, phi, color="steelblue", edgecolor="k", height=0.6)
    ax.axvline(0, color="k", linewidth=1.0)
    ax.set_yticks(height_positions)
    ax.set_yticklabels(mass_labels, fontsize=9)
    ax.set_title(f"Mode {mi+1}  –  fn = {fn_exp[mi]:.3f} Hz", fontsize=11)
    ax.set_xlabel("Relative amplitude (normalised)", fontsize=9)
    ax.grid(True, axis="x")

fig3.suptitle("Phase 2 – Experimental Mode Shapes (EMA)", fontsize=14)
fig3.tight_layout()

plt.show()
