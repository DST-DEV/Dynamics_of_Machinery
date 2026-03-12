#########################################################
# 41514 DYNAMICS OF MACHINARY                           #
# MEK - DEPARTMENT OF MECHANICAL ENGINEERING            #
# DTU - TECHNICAL UNIVERSITY OF DENMARK                 #
#                                                       #
#              Copenhagen, March 3rd,    2022           #
#                                                       #
#                                                       #
# PROJECT 1 - ROUTINE TO ESTIMATE FRF AND COHERENCE     #
#             USING SIGNAL ANALYSIS TOOLS               #
#                                                       #
#                            Ilmar Ferreira Santos      #
#                                                       #
#########################################################

# OBSERVATION: The whole signal in time domain is divided
# into blocks of size "N" (points)
# Hanning window is used
# Averaging is also done, and it is depending on the number
# of points of signal in time domain, on the number of points
# in the blocks and finally on the overlap factor.
# For facilitate the understanding of the code, the nomenclature
# used is:
# x for the input signal  (normally force signal)
# y for the output signal (normally acceleration)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, csd, welch, coherence

# IMPORTANT PARAMETERS OBTAINED FROM THE ACQUISITION SYSTEM

BL = 25  # Number of blocks
Fs = 50  # sampling frequency used while acquiring the signals [Hz]
OVLP = 1000  # "overlap" factor among the blocks (0 until N)
acc = 5  # Select accelerometer ([1,2,3,4,5,6])
# IMPORTANT:
# With acc = 5 you choose the response of mass 5 (upper blade)
# With acc = 1 you choose the response of mass 1 (lowest platform mass)

# LOADING THE INPUT AND OUTPUT SIGNALS FROM FILES

file = "data/Downsweep50Hz.txt"
data = pd.read_csv(file, delimiter=";")

x = data["Force"].values
y = data.iloc[:, acc].values  # acc column index (0-based: acc=5 → column 5)

x_original = x.copy()
y_original = y.copy()

# FILTER DESIGN
# Butterworth 5th order lowpass digital filter
# cut-off frequency: 0.99 * (Fs/2) = 24.75 Hz
# In scipy, Wn=0.99 means 0.99 * Nyquist = 0.99 * (Fs/2)

b, a = butter(5, 0.99)  # 5th order Butterworth, Wn=0.99 (normalised)

# FILTERING OF INPUT AND OUTPUT SIGNALS
x = lfilter(b, a, x)  # input signal after filtering
y = lfilter(b, a, y)  # output signal after filtering

# POWER SPECTRAL DENSITY & CROSS SPECTRAL DENSITY

N = round(len(x) / BL)  # number of points in each block
WIN = N  # Hanning window length = block size

# Cross-spectral densities (Welch's averaged periodogram)
# scipy.signal.csd returns (frequencies, Pxy)
_, PXY = csd(x, y, fs=Fs, window="hann", nperseg=WIN, noverlap=OVLP, nfft=N)
_, PYX = csd(y, x, fs=Fs, window="hann", nperseg=WIN, noverlap=OVLP, nfft=N)

# Power spectral densities
_, PXX = welch(x, fs=Fs, window="hann", nperseg=WIN, noverlap=OVLP, nfft=N)
_, PYY = welch(y, fs=Fs, window="hann", nperseg=WIN, noverlap=OVLP, nfft=N)

# FRF ESTIMATION USING THE ESTIMATORS H1 AND H2
H1y = PXY / PXX
H2y = PYY / PYX

# CALCULATION OF THE COHERENCE FUNCTION
#   Cy = |Pxy|^2 / (Pxx * Pyy)
F, Cy = coherence(x, y, fs=Fs, window="hann", nperseg=WIN, noverlap=OVLP, nfft=N)

# GRAPHICS OF THE PROCESSED SIGNALS IN TIME AND FREQUENCY DOMAINS

idx = slice(0, N // 2)  # up to 25 Hz  (N/2 points, 0-based)
t = np.arange(1, len(x) + 1) / Fs  # time vector

# Phase calculated using arccos of the normalised real part
fase1y = np.arccos(np.real(H1y) / np.abs(H1y))
fase2y = np.arccos(np.real(H2y) / np.abs(H2y))

# -------------------------------------------------------
# Graphics 1 - Analysis in the time domain
fig1, axes1 = plt.subplots(2, 1, figsize=(10, 6))

axes1[0].plot(t, y, "k-")
axes1[0].grid(True)
axes1[0].set_title("(a) Acceleration in Time Domain")
axes1[0].set_ylabel("acc [m/s²]")
axes1[0].set_xlabel("time [s]")
axes1[0].set_xlim([0, t.max()])
axes1[0].set_ylim([-6, 6])

axes1[1].plot(t, x, "k-")
axes1[1].grid(True)
axes1[1].set_title("(b) Force Signal in Time Domain")
axes1[1].set_ylabel("force [N]")
axes1[1].set_xlabel("time [s]")
axes1[1].set_xlim([0, t.max()])
axes1[1].set_ylim([-12, 12])

fig1.tight_layout()

# -------------------------------------------------------
# Graphics 2 - Coherence, FRF magnitude and phase (H1 & H2)
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 9))

axes2[0].plot(F[idx], Cy[idx], "k-", linewidth=1.5)
axes2[0].set_xlim([0, F[idx].max()])
axes2[0].set_ylim([0, 1.1])
axes2[0].grid(True)
axes2[0].tick_params(labelsize=14)
axes2[0].set_ylabel("Coherence", fontstyle="italic", fontsize=14)

axes2[1].plot(F[idx], np.abs(H1y[idx]), "b-", linewidth=1.5, label="H1")
axes2[1].plot(F[idx], np.abs(H2y[idx]), "r-", linewidth=1.5, label="H2")
axes2[1].set_xlim([0, F[idx].max()])
axes2[1].set_ylim([0, 1.1 * np.abs(H2y[idx]).max()])
axes2[1].grid(True)
axes2[1].tick_params(labelsize=14)
axes2[1].set_ylabel("FRF [(m/s²)/N]", fontstyle="italic", fontsize=14)
axes2[1].legend()

axes2[2].plot(F[idx], np.degrees(fase1y[idx]), "b-", linewidth=1.5, label="H1")
axes2[2].plot(F[idx], np.degrees(fase2y[idx]), "r-", linewidth=1.5, label="H2")
axes2[2].set_xlim([0, F[idx].max()])
axes2[2].set_ylim([0, 200])
axes2[2].set_yticks(np.arange(-180, 181, 90))
axes2[2].grid(True)
axes2[2].tick_params(labelsize=14)
axes2[2].set_xlabel("Frequency [Hz]", fontstyle="italic", fontsize=14)
axes2[2].set_ylabel("Phase [°]", fontstyle="italic", fontsize=14)
axes2[2].legend()

fig2.tight_layout()

# -------------------------------------------------------
# Graphics 3 - Log-scale FRF magnitude and H2 phase
fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6))

axes3[0].semilogy(F[idx], np.abs(H1y[idx]), "b-", linewidth=1.5, label="H1")
axes3[0].semilogy(F[idx], np.abs(H2y[idx]), "r-", linewidth=1.5, label="H2")
axes3[0].set_xlim([0.7, 15])
axes3[0].set_ylim([None, 1.1 * np.abs(H2y[idx]).max()])
axes3[0].grid(True, which="both")
axes3[0].tick_params(labelsize=14)
axes3[0].set_ylabel("FRF [(m/s²)/N]", fontstyle="italic", fontsize=14)
axes3[0].legend()

axes3[1].plot(F[idx], np.degrees(fase2y[idx]), "r-", linewidth=1.5)
axes3[1].set_xlim([0.7, 15])
axes3[1].set_ylim([0, 200])
axes3[1].set_yticks(np.arange(-180, 181, 90))
axes3[1].grid(True)
axes3[1].tick_params(labelsize=14)
axes3[1].set_xlabel("Frequency [Hz]", fontstyle="italic", fontsize=14)
axes3[1].set_ylabel("Phase [°]", fontstyle="italic", fontsize=14)

fig3.tight_layout()
plt.show()

# -------------------------------------------------------
# Final results table: FREQ [Hz] | H1 | H2 | COHERENCE
RESULT = np.column_stack([F[idx], H1y[idx], H2y[idx], Cy[idx]])
