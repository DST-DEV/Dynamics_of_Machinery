#########################################################
# 41514 DYNAMICS OF MACHINERY                           #
# MEK - DEPARTMENT OF MECHANICAL ENGINEERING            #
# DTU - TECHNICAL UNIVERSITY OF DENMARK                 #
#                                                       #
#  EMA – GLOBAL COMPARISON (FREQ + TIME + DIGITAL TWIN)#
#                                                       #
#  Compares, per mode:                                  #
#  - Frequency-domain natural frequencies               #
#  - Time-domain free-decay frequencies                 #
#  - Theoretical (digital twin) frequencies             #
#  - Damping ratios (LS, HPP, log decrement)            #
#  - Modal parameters from LS text output (m, k, d)     #
#########################################################

import re

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, lfilter
from scipy.signal import csd as _csd
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import welch as _welch


# -------------------------------
# Shared settings
# -------------------------------
N_MODES = 6
N_ACC = 6
REF_ACC = 4  # 1-based accelerometer index used for cross-method comparison


def identify_frequency_domain_modes(
    data_file="data/Downsweep50Hz.txt",
    bl=25,
    fs=50,
    ovlp=1000,
    expected_fn=None,
    search_bw=0.35,
):
    """Replicate the guided peak-picking method used in EMA_frequency_domain.py."""
    if expected_fn is None:
        expected_fn = np.array([1.27, 1.77, 2.30, 4.40, 6.71, 8.85])

    data = pd.read_csv(data_file, delimiter=";")
    x_raw = data["Force"].values

    b_f, a_f = butter(5, 0.99)
    x_filt = lfilter(b_f, a_f, x_raw)

    n_block = round(len(x_filt) / bl)
    win = n_block
    noverlap = min(ovlp, win - 1)

    h2 = np.zeros((N_ACC, win // 2 + 1), dtype=complex)

    for i in range(N_ACC):
        y_raw = data.iloc[:, i + 1].values
        y_filt = lfilter(b_f, a_f, y_raw)

        _, pxy = _csd(
            x_filt,
            y_filt,
            fs=fs,
            window="hann",
            nperseg=win,
            noverlap=noverlap,
            nfft=win,
        )
        _, pyx = _csd(
            y_filt,
            x_filt,
            fs=fs,
            window="hann",
            nperseg=win,
            noverlap=noverlap,
            nfft=win,
        )
        _, pyy = _welch(
            y_filt,
            fs=fs,
            window="hann",
            nperseg=win,
            noverlap=noverlap,
            nfft=win,
        )

        # Keep the same H2 definition used in the frequency-domain script.
        h2[i] = pyy / pyx

    f = np.fft.rfftfreq(win, d=1.0 / fs)
    freq_idx = slice(0, win // 2)
    f_p = f[freq_idx]

    h2_composite = np.sum(np.abs(h2[:, freq_idx]) ** 2, axis=0)
    h2_composite /= np.max(h2_composite)

    df = f_p[1] - f_p[0]
    min_dist = max(1, int(0.15 / df))

    identified_peaks = []
    for target_fn in expected_fn:
        in_window = np.where(
            (f_p >= target_fn - search_bw) & (f_p <= target_fn + search_bw)
        )[0]

        if len(in_window) == 0:
            identified_peaks.append(int(np.argmin(np.abs(f_p - target_fn))))
            continue

        window_response = h2_composite[in_window]
        local_peaks, _ = find_peaks(window_response, distance=min_dist)

        if len(local_peaks) > 0:
            local_proms, _, _ = peak_prominences(window_response, local_peaks)
            best_local_peak = local_peaks[int(np.argmax(local_proms))]
            identified_peaks.append(int(in_window[best_local_peak]))
        else:
            identified_peaks.append(int(in_window[np.argmax(window_response)]))

    top_peaks = np.array(identified_peaks, dtype=int)
    return f_p[top_peaks]


def theoretical_frequencies():
    """Compute digital-twin natural frequencies (same model as EMA_time_domain.py)."""
    l1, l2, l3, l4 = 0.193, 0.292, 0.220, 0.271
    l5, l6 = 0.430, 0.430

    m1, m2, m3, m4 = 2.294, 1.941, 1.943, 2.732
    m5, m6 = 0.774, 0.774
    macc = 0.073

    b1, h1 = 0.0291, 0.0011
    b2, h2 = 0.0350, 0.0015
    e = 2e11

    i1 = (b1 * h1**3) / 12
    i2 = (b2 * h2**3) / 12

    k1 = 4 * (12 * e * i1) / l1**3
    k2 = 4 * (12 * e * i1) / l2**3
    k3 = 2 * (12 * e * i1) / l3**3
    k4 = 2 * (12 * e * i1) / l4**3
    k5 = (3 * e * i2) / l5**3
    k6 = (3 * e * i2) / l6**3

    masses = np.array([m1, m2, m3, m4, m5, m6]) + macc
    m_th = np.diag(masses)

    k_th = np.zeros((6, 6))
    k_th[0, 0] = k1 + k2
    k_th[0, 1] = -k2
    k_th[1, 0] = -k2
    k_th[1, 1] = k2 + k3
    k_th[1, 2] = -k3
    k_th[2, 1] = -k3
    k_th[2, 2] = k3 + k4
    k_th[2, 3] = -k4
    k_th[3, 2] = -k4
    k_th[3, 3] = k4 + k5 + k6
    k_th[3, 4] = -k5
    k_th[4, 3] = -k5
    k_th[4, 4] = k5
    k_th[3, 5] = -k6
    k_th[5, 3] = -k6
    k_th[5, 5] = k6

    eigenvalues, _ = eigh(k_th, m_th)
    return np.sqrt(np.abs(eigenvalues)) / (2.0 * np.pi)


def log_decrement(t, y, config):
    """Estimate fn and xi from free decay using logarithmic decrement."""
    min_dist = int(config.get("min_dist", 4))
    prominence = config.get("prominence", None)

    find_kwargs = {"distance": min_dist}
    if prominence is not None:
        find_kwargs["prominence"] = prominence

    peaks, _ = find_peaks(y, **find_kwargs)
    pos_mask = y[peaks] > 0
    peaks = peaks[pos_mask]

    if len(peaks) < 4:
        return np.nan, np.nan

    start_peak_number = config.get("start_peak_number", None)
    if start_peak_number is None:
        i0 = int(np.argmax(y[peaks]))
    else:
        i0 = int(np.clip(int(start_peak_number) - 1, 0, len(peaks) - 1))

    decay_peaks = peaks[i0:]
    decay_amps = y[decay_peaks]

    threshold = 0.2 * decay_amps[0]

    # 2. Find peaks that are still above this threshold
    valid_mask = decay_amps >= threshold
    decay_peaks = decay_peaks[valid_mask]
    decay_amps = decay_amps[valid_mask]

    # 3. Proceed with log-dec calculation using the adapted N
    n_total = len(decay_amps)
    if n_total < 2:
        return np.nan, np.nan, np.nan, np.array([], dtype=int)

    deltas = [
        (1.0 / n) * np.log(decay_amps[0] / decay_amps[n])
        for n in range(1, len(decay_amps))
        if decay_amps[n] > 0 and decay_amps[0] / decay_amps[n] > 0
    ]

    if not deltas:
        return np.nan, np.nan

    delta = np.mean(deltas)
    if not np.isfinite(delta) or delta <= 0:
        return np.nan, np.nan

    xi = delta / np.sqrt(4.0 * np.pi**2 + delta**2)
    td = np.mean(np.diff(t[decay_peaks]))
    fd = 1.0 / td
    fn = fd / np.sqrt(max(1e-12, 1.0 - xi**2))
    return fn, xi


def time_domain_modes(fs=500):
    """Replicate phase-3 frequency and damping extraction from free-decay files."""
    decay_files = [f"data/Decay_Mode{i}_acc4.txt" for i in range(1, N_MODES + 1)]

    mode_logdec_config = [
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

    fn_list = []
    xi_list = []

    for i, fpath in enumerate(decay_files):
        data_i = pd.read_csv(fpath, delimiter=";")
        y = data_i.iloc[:, 1].values
        t = np.arange(len(y)) / fs
        cfg = mode_logdec_config[i]
        fn_i, xi_i = log_decrement(t, y, cfg)
        fn_list.append(fn_i)
        xi_list.append(xi_i)

    return np.array(fn_list), np.array(xi_list)


def parse_modal_mass_stiffness_damping(
    file_path="EMA_modal_mass_stiffness_damping.txt",
):
    """Parse LS modal table into per-mode arrays for m, k, d and zeta."""
    mode_data = {
        mode: {"m": [], "k": [], "d": [], "zeta": []} for mode in range(1, N_MODES + 1)
    }
    current_mode = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            mode_match = re.search(r"Mode\s+(\d+)\s+\|", line)
            if mode_match:
                current_mode = int(mode_match.group(1))
                continue

            row_match = re.search(
                r"Acc\s+\d+\s+([+-]?\d+\.\d+e[+-]\d+)\s+([+-]?\d+\.\d+e[+-]\d+)\s+([+-]?\d+\.\d+e[+-]\d+)\s+([+-]?\d+\.\d+)",
                line,
            )
            if row_match and current_mode is not None:
                m_val, k_val, d_val, zeta_val = map(float, row_match.groups())
                mode_data[current_mode]["m"].append(m_val)
                mode_data[current_mode]["k"].append(k_val)
                mode_data[current_mode]["d"].append(d_val)
                mode_data[current_mode]["zeta"].append(zeta_val)

    return mode_data


def parse_modal_damping_hpp(file_path="EMA_modal_damping_hpp.txt"):
    """Parse HPP damping table into per-mode arrays."""
    mode_data = {mode: [] for mode in range(1, N_MODES + 1)}
    current_mode = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            mode_match = re.search(r"Mode\s+(\d+)\s+\|", line)
            if mode_match:
                current_mode = int(mode_match.group(1))
                continue

            row_match = re.search(r"Acc\s+\d+\s+([+-]?\d+\.\d+)", line)
            if row_match and current_mode is not None:
                mode_data[current_mode].append(float(row_match.group(1)))

    return mode_data


def safe_pct_err(a, b):
    """Percent error |a-b|/|b|*100, NaN-safe."""
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return abs(a - b) / abs(b) * 100.0


def main():
    print("\n" + "=" * 90)
    print("  EMA GLOBAL COMPARISON: FREQUENCY DOMAIN, TIME DOMAIN, AND DIGITAL TWIN")
    print("=" * 90)

    fn_freq = identify_frequency_domain_modes()
    fn_time, xi_time = time_domain_modes()
    fn_theory = theoretical_frequencies()

    ls_data = parse_modal_mass_stiffness_damping()
    hpp_data = parse_modal_damping_hpp()

    rows = []
    ref_acc_idx = int(np.clip(REF_ACC - 1, 0, N_ACC - 1))

    for mode in range(1, N_MODES + 1):
        idx = mode - 1

        m_vals = np.array(ls_data[mode]["m"], dtype=float)
        k_vals = np.array(ls_data[mode]["k"], dtype=float)
        d_vals = np.array(ls_data[mode]["d"], dtype=float)
        zeta_ls_vals = np.array(ls_data[mode]["zeta"], dtype=float)
        zeta_hpp_vals = np.array(hpp_data[mode], dtype=float)

        # Use one reference accelerometer instead of averaging across all channels.
        m_ref = m_vals[ref_acc_idx] if ref_acc_idx < m_vals.size else np.nan
        k_ref = k_vals[ref_acc_idx] if ref_acc_idx < k_vals.size else np.nan
        d_ref = d_vals[ref_acc_idx] if ref_acc_idx < d_vals.size else np.nan
        zeta_ls_ref = (
            zeta_ls_vals[ref_acc_idx] if ref_acc_idx < zeta_ls_vals.size else np.nan
        )
        zeta_hpp_ref = (
            zeta_hpp_vals[ref_acc_idx] if ref_acc_idx < zeta_hpp_vals.size else np.nan
        )

        rows.append(
            {
                "Mode": mode,
                "fn_freq_Hz": fn_freq[idx] if idx < len(fn_freq) else np.nan,
                "fn_time_Hz": fn_time[idx] if idx < len(fn_time) else np.nan,
                "fn_theory_Hz": fn_theory[idx] if idx < len(fn_theory) else np.nan,
                "err_freq_vs_theory_pct": safe_pct_err(
                    fn_freq[idx] if idx < len(fn_freq) else np.nan,
                    fn_theory[idx] if idx < len(fn_theory) else np.nan,
                ),
                "err_time_vs_theory_pct": safe_pct_err(
                    fn_time[idx] if idx < len(fn_time) else np.nan,
                    fn_theory[idx] if idx < len(fn_theory) else np.nan,
                ),
                "err_time_vs_freq_pct": safe_pct_err(
                    fn_time[idx] if idx < len(fn_time) else np.nan,
                    fn_freq[idx] if idx < len(fn_freq) else np.nan,
                ),
                "zeta_time_logdec": xi_time[idx] if idx < len(xi_time) else np.nan,
                "zeta_ls_ref_acc": zeta_ls_ref,
                "zeta_hpp_ref_acc": zeta_hpp_ref,
                "err_zeta_time_vs_hpp_pct": safe_pct_err(
                    xi_time[idx] if idx < len(xi_time) else np.nan, zeta_hpp_ref
                ),
                "err_zeta_time_vs_ls_pct": safe_pct_err(
                    xi_time[idx] if idx < len(xi_time) else np.nan, zeta_ls_ref
                ),
                "m_modal_ref_acc": m_ref,
                "k_modal_ref_acc": k_ref,
                "d_modal_ref_acc": d_ref,
                "ref_accelerometer": ref_acc_idx + 1,
            }
        )

    df = pd.DataFrame(rows)

    print("\nMode-by-mode comparison (key values):")
    with pd.option_context("display.max_columns", None, "display.width", 170):
        print(df.round(6).to_string(index=False))

    # Additional detailed table: per-mode, per-accelerometer LS + HPP damping values.
    detail_rows = []
    for mode in range(1, N_MODES + 1):
        for acc in range(N_ACC):
            m_val = ls_data[mode]["m"][acc] if acc < len(ls_data[mode]["m"]) else np.nan
            k_val = ls_data[mode]["k"][acc] if acc < len(ls_data[mode]["k"]) else np.nan
            d_val = ls_data[mode]["d"][acc] if acc < len(ls_data[mode]["d"]) else np.nan
            z_ls = (
                ls_data[mode]["zeta"][acc]
                if acc < len(ls_data[mode]["zeta"])
                else np.nan
            )
            z_hpp = hpp_data[mode][acc] if acc < len(hpp_data[mode]) else np.nan

            detail_rows.append(
                {
                    "Mode": mode,
                    "Accelerometer": acc + 1,
                    "m_modal": m_val,
                    "k_modal": k_val,
                    "d_modal": d_val,
                    "zeta_ls": z_ls,
                    "zeta_hpp": z_hpp,
                }
            )

    df_detail = pd.DataFrame(detail_rows)

    out_main_csv = "EMA_global_comparison_summary.csv"
    out_detail_csv = "EMA_global_comparison_per_acc.csv"
    out_txt = "EMA_global_comparison_summary.txt"

    df.to_csv(out_main_csv, index=False)
    df_detail.to_csv(out_detail_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("EMA GLOBAL COMPARISON SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write(df.round(6).to_string(index=False))
        f.write("\n")

    print("\nSaved comparison files:")
    print(f"  - {out_main_csv}")
    print(f"  - {out_detail_csv}")
    print(f"  - {out_txt}")


if __name__ == "__main__":
    main()
