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

ax_frf.set_xlim([0.5, 10])
ax_frf.grid(True, which="both")
ax_frf.set_ylabel("|H2| [(m/s²)/N]", fontsize=13)
ax_frf.set_title("Phase 1 – FRF Magnitude (H2) – All 6 Accelerometers", fontsize=13)
ax_frf.legend(fontsize=10)

ax_coh.set_xlim([0.5, 10])
ax_coh.set_ylim([0, 1.1])
ax_coh.grid(True)
ax_coh.set_ylabel("Coherence γ²(f)", fontsize=13)
ax_coh.set_xlabel("Frequency [Hz]", fontsize=13)
ax_coh.legend(fontsize=10)

fig1.tight_layout()

# ═══════════════════════════════════════════════════════
# PHASE 1 – IDENTIFY 6 NATURAL FREQUENCIES
# Guided search: for each expected frequency, run find_peaks
# inside a ±search_bw window of the composite FRF and keep
# the most prominent one. Falls back to argmax if no peak
# is found (e.g. very flat region due to low coherence).
# ═══════════════════════════════════════════════════════
H2_composite = np.sum(np.abs(H2[:, freq_idx]) ** 2, axis=0)
H2_composite /= H2_composite.max()

expected_fn = np.array([1.27, 1.77, 2.30, 4.40, 6.71, 8.85])
search_bw = 0.35  # search each mode inside ±0.35 Hz around its expected position
df = F_p[1] - F_p[0]  # frequency resolution [Hz]
min_dist = max(1, int(0.15 / df))  # local minimum spacing inside each search window

identified_peaks = []
for target_fn in expected_fn:
    in_window = np.where(
        (F_p >= target_fn - search_bw) & (F_p <= target_fn + search_bw)
    )[0]

    if len(in_window) == 0:
        identified_peaks.append(int(np.argmin(np.abs(F_p - target_fn))))
        continue

    window_response = H2_composite[in_window]
    local_peaks, _ = find_peaks(window_response, distance=min_dist)

    if len(local_peaks) > 0:
        local_proms, _, _ = peak_prominences(window_response, local_peaks)
        best_local_peak = local_peaks[int(np.argmax(local_proms))]
        identified_peaks.append(int(in_window[best_local_peak]))
    else:
        identified_peaks.append(int(in_window[np.argmax(window_response)]))

top6_peaks = np.array(identified_peaks, dtype=int)

fn_exp = F_p[top6_peaks]  # natural frequencies [Hz]
wn_exp = 2 * np.pi * fn_exp  # [rad/s]

print("\n" + "=" * 58)
print("  PHASE 1 – IDENTIFIED NATURAL FREQUENCIES")
print("=" * 58)
print(f"  {'Mode':>4}  {'fn [Hz]':>10}  {'wn [rad/s]':>12}")
print("  " + "-" * 30)
for k, (fn, wn) in enumerate(zip(fn_exp, wn_exp), 1):
    print(f"  {k:>4}  {fn:>10.4f}  {wn:>12.4f}")

# ═══════════════════════════════════════════════════════
# PHASE 2 – MODAL PARAMETERS  (Least squares fit of 1-DOF model)
# ═══════════════════════════════════════════════════════
fit_radius = 2

modal_mass = np.zeros((N_ACC, N_ACC))
modal_stiffness = np.zeros((N_ACC, N_ACC))
modal_damping = np.zeros((N_ACC, N_ACC))
modal_zeta = np.zeros((N_ACC, N_ACC))
mode_shapes = np.zeros((N_ACC, N_ACC))

for mode_idx, peak_idx in enumerate(top6_peaks):
    start_idx = max(0, peak_idx - fit_radius)
    stop_idx = min(len(F_p), peak_idx + fit_radius + 1)
    fit_idx = np.arange(start_idx, stop_idx)

    omega_fit = 2 * np.pi * F_p[fit_idx]
    peak_shape = H2[:, peak_idx].imag.copy()
    ref_sign = np.sign(peak_shape[np.argmax(np.abs(peak_shape))])
    if ref_sign == 0:
        ref_sign = 1.0
    mode_shapes[:, mode_idx] = peak_shape / np.max(np.abs(peak_shape)) * ref_sign

    for acc_idx in range(N_ACC):
        frf_fit = H2[acc_idx, fit_idx]
        valid = np.abs(frf_fit) > 1e-12

        if np.count_nonzero(valid) < 3:
            continue

        omega_valid = omega_fit[valid]
        frf_valid = frf_fit[valid]
        dynamic_term = -(omega_valid**2) / frf_valid

        real_system = np.column_stack((-(omega_valid**2), np.ones_like(omega_valid)))
        imag_system = omega_valid[:, None]

        m_fit, k_fit = np.linalg.lstsq(real_system, dynamic_term.real, rcond=None)[0]
        d_fit = np.linalg.lstsq(imag_system, dynamic_term.imag, rcond=None)[0][0]

        modal_mass[acc_idx, mode_idx] = m_fit
        modal_stiffness[acc_idx, mode_idx] = k_fit
        modal_damping[acc_idx, mode_idx] = d_fit

        if (m_fit > 0 and k_fit < 0) or (m_fit < 0 and k_fit > 0):
            modal_zeta[acc_idx, mode_idx] = np.nan
        else:
            modal_zeta[acc_idx, mode_idx] = d_fit / (2 * np.sqrt(m_fit * k_fit))

print("\n" + "=" * 86)
print(
    "  PHASE 2 (Least Square Method)  – MODAL PARAMETERS FROM H2 (PER ACCELEROMETER / MODE)"
)
print("=" * 86)

phase2_lines = []
phase2_lines.append("=" * 86)
phase2_lines.append(
    "  PHASE 2 (Least Square Method) - MODAL PARAMETERS FROM H2 (PER ACCELEROMETER / MODE)"
)
phase2_lines.append("=" * 86)

for mode_idx, fn in enumerate(fn_exp):
    print(f"\n  Mode {mode_idx + 1}  |  fn = {fn:0.4f} Hz")
    print(
        f"  {'Acc':>6}  {'m_modal':>12}  {'k_modal':>12}  {'d_modal':>12}  {'zeta':>10}"
    )
    print("  " + "-" * 62)

    phase2_lines.append("")
    phase2_lines.append(f"  Mode {mode_idx + 1}  |  fn = {fn:0.4f} Hz")
    phase2_lines.append(
        f"  {'Acc':>6}  {'m_modal':>12}  {'k_modal':>12}  {'d_modal':>12}  {'zeta':>10}"
    )
    phase2_lines.append("  " + "-" * 62)

    for acc_idx in range(N_ACC):
        row = (
            f"  {acc_names[acc_idx]:>6}  "
            f"{modal_mass[acc_idx, mode_idx]:>12.4e}  "
            f"{modal_stiffness[acc_idx, mode_idx]:>12.4e}  "
            f"{modal_damping[acc_idx, mode_idx]:>12.4e}  "
            f"{modal_zeta[acc_idx, mode_idx]:>10.4f}"
        )
        print(row)
        phase2_lines.append(row)

modal_table_txt_path = "EMA_modal_mass_stiffness_damping.txt"
with open(modal_table_txt_path, "w", encoding="utf-8") as txt_file:
    txt_file.write("\n".join(phase2_lines) + "\n")

print(f"\nSaved modal parameter table to: {modal_table_txt_path}")

# ═══════════════════════════════════════════════════════
# PHASE 2 – MODAL PARAMETERS  (Half power point method)
# ═══════════════════════════════════════════════════════

# Determine damping through the -3 dB (half-power) bandwidth around each peak.
modal_zeta_hpp = np.full((N_ACC, N_ACC), np.nan)

for mode_idx, peak_idx in enumerate(top6_peaks):
    omega_n = 2 * np.pi * F_p[peak_idx]

    for acc_idx in range(N_ACC):
        frf_mag = np.abs(H2[acc_idx, freq_idx])
        peak_amp = frf_mag[peak_idx]

        if peak_amp <= 0:
            continue

        target_amp = peak_amp / np.sqrt(2)
        idx_lower = peak_idx
        idx_upper = peak_idx

        while idx_lower > 0 and frf_mag[idx_lower] > target_amp:
            idx_lower -= 1

        while idx_upper < len(frf_mag) - 1 and frf_mag[idx_upper] > target_amp:
            idx_upper += 1

        if idx_lower == 0 or idx_upper == len(frf_mag) - 1:
            continue

        omega_lower = 2 * np.pi * F_p[idx_lower]
        omega_upper = 2 * np.pi * F_p[idx_upper]
        modal_zeta_hpp[acc_idx, mode_idx] = (omega_upper - omega_lower) / (2 * omega_n)

print("\n" + "=" * 86)
print(
    "  PHASE 2 (Half Power Point)  – MODAL PARAMETERS FROM H2 (PER ACCELEROMETER / MODE)"
)
print("=" * 86)

phase2_lines = []
phase2_lines.append("=" * 86)
phase2_lines.append(
    "  PHASE 2 (Half Power Point) - MODAL PARAMETERS FROM H2 (PER ACCELEROMETER / MODE)"
)
phase2_lines.append("=" * 86)

for mode_idx, fn in enumerate(fn_exp):
    print(f"\n  Mode {mode_idx + 1}  |  fn = {fn:0.4f} Hz")
    print(f"  {'Acc':>6} {'zeta':>10}")
    print("  " + "-" * 62)

    phase2_lines.append("")
    phase2_lines.append(f"  Mode {mode_idx + 1}  |  fn = {fn:0.4f} Hz")
    phase2_lines.append(f"  {'Acc':>6}  {'zeta':>10}")
    phase2_lines.append("  " + "-" * 62)

    for acc_idx in range(N_ACC):
        row = (
            f"  {acc_names[acc_idx]:>6}  " f"{modal_zeta_hpp[acc_idx, mode_idx]:>10.4f}"
        )
        print(row)
        phase2_lines.append(row)

modal_table_txt_path = "EMA_modal_damping_hpp.txt"
with open(modal_table_txt_path, "w", encoding="utf-8") as txt_file:
    txt_file.write("\n".join(phase2_lines) + "\n")

print(f"\nSaved modal parameter table to: {modal_table_txt_path}")

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
axes2[0].set_title("FRF H1", fontsize=12)
axes2[1].set_ylabel("|H2| [(m/s²)/N]", fontsize=12)
axes2[1].set_title("FRF H2", fontsize=12)

axes2[2].set_xlim([0.5, 10])
axes2[2].set_ylim([0, 1.1])
axes2[2].grid(True)
axes2[2].set_ylabel("Coherence", fontsize=12)
axes2[2].set_xlabel("Frequency [Hz]", fontsize=12)
axes2[2].legend(fontsize=9)


fig2.tight_layout()

plt.savefig("Phase2_FRF_Analysis.pdf", bbox_inches='tight')



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
