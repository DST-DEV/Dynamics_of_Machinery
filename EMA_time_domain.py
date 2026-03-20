#########################################################
# 41514 DYNAMICS OF MACHINERY                           #
# MEK - DEPARTMENT OF MECHANICAL ENGINEERING            #
# DTU - TECHNICAL UNIVERSITY OF DENMARK                 #
#                                                       #
#  EMA – PHASE 3 & 4: TIME DOMAIN + MODEL VALIDATION   #
#                                                       #
#  Phase 3: Free-decay analysis (all 6 modes)           #
#           Logarithmic decrement → ξ_i (eq. 9)        #
#           FFT verification of single-frequency decay  #
#  Phase 4: Digital-twin comparison table               #
#           Theoretical ωn from mass-stiffness model    #
#           % discrepancy: frequency and damping        #
#########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import eigh

# ─────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────
Fs = 500  # sampling frequency [Hz]
N_MODES = 6

decay_files = [f"data/Decay_Mode{i}_acc4.txt" for i in range(1, N_MODES + 1)]

# Mode-specific settings for logarithmic decrement.
# start_peak_number is 1-based after positive-peak filtering.
# n_decrement_points is the number of peaks used, including A0.
MODE_LOGDEC_CONFIG = [
    {
        "min_dist": 4,
        "prominence": None,
        "start_peak_number": None,
    },
    {
        "min_dist": 1,
        "prominence": 0.3,
        "start_peak_number": None,
    },
    {
        "min_dist": 4,
        "prominence": None,
        "start_peak_number": None,
    },
    {
        "min_dist": 4,
        "prominence": 0.6,
        "start_peak_number": 7,
    },
    {
        "min_dist": 1,
        "prominence": None,
        "start_peak_number": 13,
    },
    {
        "min_dist": 1,
        "prominence": None,
        "start_peak_number": 16,
    },
]

# ═══════════════════════════════════════════════════════
# PHASE 4 SETUP – THEORETICAL MODEL (digital twin)
# Parameters from model_dummy.py
# Topology: 6-DOF chain
#   ground ─k1─ m1 ─k2─ m2 ─k3─ m3 ─k4─ m4 ─┬─k5─ m5 (blade 1)
#                                               └─k6─ m6 (blade 2)
# ═══════════════════════════════════════════════════════

# Lengths [m]
l1, l2, l3, l4 = 0.193, 0.292, 0.220, 0.271
l5, l6 = 0.430, 0.430

# Platform masses [kg]
m1, m2, m3, m4 = 2.294, 1.941, 1.943, 2.732
m5, m6 = 0.774, 0.774
macc = 0.073  # accelerometer mass [kg] added to each DOF

# Cross-section properties
b1, h1 = 0.0291, 0.0011  # tower beam
b2, h2 = 0.0350, 0.0015  # blade beam
E = 2e11  # Young's modulus [Pa]

I1 = (b1 * h1**3) / 12
I2 = (b2 * h2**3) / 12

# Stiffness of structural elements [N/m]
# 4 clamped-clamped columns:  k = 4 × (12EI/L³)
# 2 clamped-clamped columns:  k = 2 × (12EI/L³)
# 1 cantilever blade:         k = 3EI/L³
k1 = 4 * (12 * E * I1) / l1**3
k2 = 4 * (12 * E * I1) / l2**3
k3 = 2 * (12 * E * I1) / l3**3
k4 = 2 * (12 * E * I1) / l4**3
k5 = (3 * E * I2) / l5**3
k6 = (3 * E * I2) / l6**3

# Mass matrix – diagonal (include accelerometer mass at each DOF)
masses = np.array([m1, m2, m3, m4, m5, m6]) + macc
M_th = np.diag(masses)

# Stiffness matrix (6 × 6)
K_th = np.zeros((6, 6))
K_th[0, 0] = k1 + k2
K_th[0, 1] = -k2
K_th[1, 0] = -k2
K_th[1, 1] = k2 + k3
K_th[1, 2] = -k3
K_th[2, 1] = -k3
K_th[2, 2] = k3 + k4
K_th[2, 3] = -k4
K_th[3, 2] = -k4
K_th[3, 3] = k4 + k5 + k6
K_th[3, 4] = -k5
K_th[4, 3] = -k5
K_th[4, 4] = k5
K_th[3, 5] = -k6
K_th[5, 3] = -k6
K_th[5, 5] = k6

# Solve undamped generalised eigenvalue problem: K φ = ω² M φ
# eigh returns eigenvalues in ascending order (guaranteed real & positive)
eigenvalues, eigenvectors = eigh(K_th, M_th)
fn_th = np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)  # theoretical fn [Hz]

print("\n" + "=" * 48)
print("  THEORETICAL NATURAL FREQUENCIES (digital twin)")
print("=" * 48)
print(f"  {'Mode':>4}  {'fn_th [Hz]':>12}  {'ωn_th [rad/s]':>15}")
print("  " + "-" * 34)
for i, fn in enumerate(fn_th, 1):
    print(f"  {i:>4}  {fn:>12.4f}  {2*np.pi*fn:>15.4f}")

# ═══════════════════════════════════════════════════════
# PHASE 3 – LOGARITHMIC DECREMENT ANALYSIS
# ═══════════════════════════════════════════════════════


def log_decrement(t, y, Fs, config):
    """
    Estimate damping ratio ξ and undamped natural frequency fn [Hz]
    from a free-decay signal using the logarithmic decrement method.

    Equation (9):  δ = (1/n) × ln(A₀ / Aₙ)
                   ξ = δ / √(4π² + δ²)

    Strategy:
    - Detect positive peaks using mode-specific peak detection parameters.
    - Start from a user-defined peak index (or global maximum if not set).
    - Use a user-defined number of decrement peaks for each mode.
    - Average δ over A₀/Aₙ pairs in the selected peak segment.
    - fd estimated from mean inter-peak spacing; fn = fd / √(1−ξ²).
    """
    min_dist = int(config.get("min_dist", 4))
    prominence = config.get("prominence", None)

    find_kwargs = {"distance": min_dist}
    if prominence is not None:
        find_kwargs["prominence"] = prominence

    peaks, _ = find_peaks(y, **find_kwargs)

    # Restrict to positive peaks (genuine oscillation half-cycles)
    pos_mask = y[peaks] > 0
    peaks = peaks[pos_mask]

    if len(peaks) < 4:
        return np.nan, np.nan, np.nan, np.array([], dtype=int)

    start_peak_number = config.get("start_peak_number", None)
    if start_peak_number is None:
        # Default behavior: global maximum as A0.
        i0 = int(np.argmax(y[peaks]))
    else:
        # 1-based user setting mapped to valid peak index range.
        i0 = int(np.clip(int(start_peak_number) - 1, 0, len(peaks) - 1))

    # Use detected peaks from the selected start peak onward.
    # Do not force monotonic decay, since real measurements can have local
    # amplitude rises that should still be included in the user-selected set.
    decay_peaks = peaks[i0:]
    decay_amps = y[decay_peaks]

    # 1. Define the 20% threshold
    threshold = 0.2 * decay_amps[0]

    # 2. Find peaks that are still above this threshold
    valid_mask = decay_amps >= threshold
    decay_peaks = decay_peaks[valid_mask]
    decay_amps = decay_amps[valid_mask]

    # 3. Proceed with log-dec calculation using the adapted N
    n_total = len(decay_amps)
    if n_total < 2:
        return np.nan, np.nan, np.nan, np.array([], dtype=int)

    # Log-decrement: average (1/n)·ln(A₀/Aₙ) over all n from 1 to end
    n_total = len(decay_amps)
    deltas = [
        (1.0 / n) * np.log(decay_amps[0] / decay_amps[n])
        for n in range(1, n_total)
        if decay_amps[n] > 0 and decay_amps[0] / decay_amps[n] > 0
    ]
    if not deltas:
        return np.nan, np.nan, np.nan, np.array([], dtype=int)

    delta = np.mean(deltas)
    if not np.isfinite(delta) or delta <= 0:
        return np.nan, np.nan, np.nan, np.array([], dtype=int)

    # Damping ratio
    xi = delta / np.sqrt(4.0 * np.pi**2 + delta**2)

    # Damped natural frequency from mean inter-peak spacing
    Td = np.mean(np.diff(t[decay_peaks]))  # damped period [s]
    fd = 1.0 / Td  # damped fn [Hz]
    fn = fd / np.sqrt(max(1e-12, 1.0 - xi**2))

    return fn, xi, delta, decay_peaks


# ─────────────────────────────────────────────────────
# LOOP OVER ALL 6 DECAY FILES
# ─────────────────────────────────────────────────────
exp_fn = []
exp_xi = []
exp_del = []

fig_td, axes_td = plt.subplots(N_MODES, 2, figsize=(14, 4 * N_MODES))

for i, fpath in enumerate(decay_files):
    data_i = pd.read_csv(fpath, delimiter=";")
    y = data_i.iloc[:, 1].values  # acceleration (acc4 channel)
    N = len(y)
    t = np.arange(N) / Fs

    cfg_i = (
        MODE_LOGDEC_CONFIG[i] if i < len(MODE_LOGDEC_CONFIG) else MODE_LOGDEC_CONFIG[-1]
    )
    fn_i, xi_i, delta_i, decay_peaks_i = log_decrement(t, y, Fs, cfg_i)
    exp_fn.append(fn_i)
    exp_xi.append(xi_i)
    exp_del.append(delta_i)

    # FFT – check for single dominant frequency
    y_fft = np.abs(np.fft.rfft(y)) / N
    f_fft = np.fft.rfftfreq(N, d=1.0 / Fs)
    f_max = min(3.0 * fn_i if not np.isnan(fn_i) else Fs / 2, Fs / 2)

    # ── Time-domain subplot ──────────────────────────
    ax_t = axes_td[i, 0]
    ax_t.plot(t, y, "k-", linewidth=0.6)
    ax_t.grid(True)
    if not np.isnan(fn_i):
        title = (
            f"Mode {i+1} – Free decay  |  "
            f"fn = {fn_i:.3f} Hz  |  ξ = {xi_i:.4f}  |  δ = {delta_i:.4f}"
        )
    else:
        title = f"Mode {i+1} – Free decay  |  peak detection failed"
    ax_t.set_title(title, fontsize=9)
    ax_t.set_ylabel("acc [m/s²]", fontsize=9)
    ax_t.set_xlabel("time [s]", fontsize=9)

    # Overlay the decay peaks actually used for log decrement
    if len(decay_peaks_i) > 0:
        ax_t.plot(
            t[decay_peaks_i],
            y[decay_peaks_i],
            "ro",
            markersize=3,
            label=f"peaks used ({len(decay_peaks_i)})",
        )
    ax_t.legend(fontsize=8)

    # ── FFT subplot ──────────────────────────────────
    ax_f = axes_td[i, 1]
    mask = f_fft <= f_max
    ax_f.plot(f_fft[mask], y_fft[mask], "r-", linewidth=0.8)
    if not np.isnan(fn_i):
        ax_f.axvline(
            fn_i, color="b", linestyle="--", linewidth=1.2, label=f"fn = {fn_i:.3f} Hz"
        )
        ax_f.legend(fontsize=9)
    ax_f.grid(True)
    ax_f.set_title(f"Mode {i+1} – FFT (single-frequency verification)", fontsize=9)
    ax_f.set_ylabel("|FFT| [m/s²]", fontsize=9)
    ax_f.set_xlabel("Freq [Hz]", fontsize=9)

    # Print per-mode summary
    if not np.isnan(fn_i):
        print(
            f"\n  Mode {i+1}: fn = {fn_i:.4f} Hz | ξ = {xi_i:.5f} | δ = {delta_i:.5f}"
            f" | start_peak={cfg_i.get('start_peak_number')} | N_peaks={cfg_i.get('n_decrement_points')}"
        )
    else:
        print(f"\n  Mode {i+1}: peak detection failed – check signal quality")

fig_td.suptitle("Phase 3 – Free-Decay Analysis: Logarithmic Decrement", fontsize=13)
fig_td.tight_layout()

# ═══════════════════════════════════════════════════════
# PHASE 4 – MODEL VALIDATION TABLE
# ═══════════════════════════════════════════════════════
# Assumed structural damping for steel (~0.5 %) – update if
# your Digital-Twin model provides mode-specific ξ values.
xi_th_assumed = 0.005

print("\n" + "═" * 86)
print("  PHASE 4 – MODEL VALIDATION: DIGITAL TWIN vs EXPERIMENT")
print("═" * 86)
print(
    f"  {'Mode':>4}  {'fn_th [Hz]':>11}  {'fn_exp [Hz]':>12}  "
    f"{'Δfn [%]':>9}  {'ξ_th [-]':>10}  {'ξ_exp [-]':>10}  {'Δξ [%]':>9}"
)
print("  " + "─" * 80)

for i in range(N_MODES):
    fn_t = fn_th[i] if i < len(fn_th) else np.nan
    fn_e = exp_fn[i] if i < len(exp_fn) and exp_fn[i] is not None else np.nan
    xi_e = exp_xi[i] if i < len(exp_xi) and exp_xi[i] is not None else np.nan

    err_fn = (
        abs(fn_e - fn_t) / fn_t * 100
        if not (np.isnan(fn_e) or np.isnan(fn_t) or fn_t == 0)
        else np.nan
    )
    err_xi = (
        abs(xi_e - xi_th_assumed) / xi_th_assumed * 100
        if not np.isnan(xi_e)
        else np.nan
    )

    fn_t_s = f"{fn_t:>11.4f}" if not np.isnan(fn_t) else f"{'N/A':>11}"
    fn_e_s = f"{fn_e:>12.4f}" if not np.isnan(fn_e) else f"{'N/A':>12}"
    err_fn_s = f"{err_fn:>9.2f}" if not np.isnan(err_fn) else f"{'N/A':>9}"
    xi_e_s = f"{xi_e:>10.5f}" if not np.isnan(xi_e) else f"{'N/A':>10}"
    err_xi_s = f"{err_xi:>9.2f}" if not np.isnan(err_xi) else f"{'N/A':>9}"

    print(
        f"  {i+1:>4}  {fn_t_s}  {fn_e_s}  "
        f"{err_fn_s}  {xi_th_assumed:>10.5f}  {xi_e_s}  {err_xi_s}"
    )

print("  " + "─" * 80)
print(f"  ξ_th = assumed structural damping ratio ({xi_th_assumed*100:.1f} %).")
print(f"  Update K_th / M_th matrices if model discrepancy exceeds ±10 %.")
print("═" * 86)

plt.show()
