#########################################################
# 41514 DYNAMICS OF MACHINARY                           #
# MEK - DEPARTMENT OF MECHANICAL ENGINEERING            #
# DTU - TECHNICAL UNIVERSITY OF DENMARK                 #
#                                                       #
#              Copenhagen, March 3rd,   2022            #
#                                                       #
#                                                       #
# PROJECT 1 - ROUTINE TO ESTIMATE FRF AND COHERENCE     #
#             USING SIGNAL ANALYSIS TOOLS               #
#                                                       #
#                            Ilmar Ferreira Santos      #
#                                                       #
#########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For facilitate the understanding of the code, the nomenclature
# used is:
# x for the input signal  (normally force signal)
# y for the output signal (normally acceleration)


# IMPORTANT PARAMETERS OBTAINED FROM THE ACQUISITION SYSTEM
Fs = 500  # sampling frequency used while acquiring the signals [Hz]

# LOADING THE INPUT AND OUTPUT SIGNALS FROM FILES
# IMPORTANT:
# Choosing 'Decay_Mode1_acc4.txt' you see the transient decay of mode 1
# Choosing 'Decay_Mode2_acc4.txt' you see the transient decay of mode 2
# Choosing 'Decay_Mode5_acc4.txt' you see the transient decay of mode 5

file = "data/Decay_Mode1_acc4.txt"
data = pd.read_csv(file, delimiter=";")

x = data["Force"].values
y = data.iloc[:, 1].values  # do not change it!

# GRAPHICS OF THE PROCESSED SIGNALS IN TIME DOMAINS
N = len(x)
t = np.arange(1, N + 1) / Fs  # time vector
Freq = np.arange(0, Fs / 2, Fs / N)  # frequency vector

y_fft = np.abs(np.fft.fft(y[:N])) / N  # fft of the acceleration signal
f_fft = np.abs(np.fft.fft(x[:N])) / N  # fft of the force signal

n_freq = N // 20  # plot up to Fs/20 Hz range

# -------------------------------------------------------
# Graphics 1 - Analysis in the time domain
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

axes1[0].plot(t, y, "k-")
axes1[0].grid(True)
axes1[0].set_title("(a) Acceleration in Time Domain")
axes1[0].set_ylabel("acc [m/s²]")
axes1[0].set_xlabel("time [s]")
axes1[0].set_xlim([0, t.max()])
axes1[0].set_ylim([-35, 35])

axes1[1].plot(t, x, "k-")
axes1[1].grid(True)
axes1[1].set_title("(b) Force Signal in Time Domain")
axes1[1].set_ylabel("force [N]")
axes1[1].set_xlabel("time [s]")
axes1[1].set_xlim([0, t.max()])
axes1[1].set_ylim([-2, 2])

fig1.tight_layout()

# -------------------------------------------------------
# Graphics 2 - Analysis in the time and frequency domains
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

axes2[0, 0].plot(t, x, "r-")
axes2[0, 0].grid(True)
axes2[0, 0].set_title("(a) Force in Time Domain")
axes2[0, 0].set_ylabel("force [N]")
axes2[0, 0].set_xlabel("time [s]")

axes2[1, 0].plot(Freq[:n_freq], f_fft[:n_freq], "r-")
axes2[1, 0].grid(True)
axes2[1, 0].set_title("(b) Force in Frequency Domain")
axes2[1, 0].set_ylabel("FFT(force) [N]")
axes2[1, 0].set_xlabel("Freq [Hz]")

axes2[0, 1].plot(t, y, "r-")
axes2[0, 1].grid(True)
axes2[0, 1].set_title("(c) Acceleration in Time Domain")
axes2[0, 1].set_ylabel("acc [m/s²]")
axes2[0, 1].set_xlabel("time [s]")

axes2[1, 1].plot(Freq[:n_freq], y_fft[:n_freq], "r-")
axes2[1, 1].grid(True)
axes2[1, 1].set_title("(d) Acceleration in Frequency Domain")
axes2[1, 1].set_ylabel("FFT(acc) [m/s²]")
axes2[1, 1].set_xlabel("Freq [Hz]")

fig2.tight_layout()
plt.show()
