from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scivis

alpha = 0.0441
beta = 0.000115
colors = ["#0D0887", "#CB4679", "#FFB300"]
exp_fld = Path(__file__).parent / "plots"

omega_Hz = np.linspace(0, 10, 300)
omega_Hz[0]=1e-2
omega_rads = omega_Hz * 2 * np.pi

xi_alpha = alpha / (2*omega_rads)
xi_beta =  beta * omega_rads/ 2
xi = xi_alpha + xi_beta
rcparams = scivis.rcparams._prepare_rcparams(profile="partsize", scale=.65)
with mpl.rc_context(rcparams):
    fig, ax = scivis.subplots(figsize=(16, 8))
    ax.plot(omega_Hz, xi_alpha, ls="-", lw=2, c=colors[0],
            label=r"$\xi=\alpha/(2\omega)$")
    ax.plot(omega_Hz, xi_beta, ls="-", lw=2, c=colors[1],
            label=r"$\xi=\beta\cdot\omega/2$")
    ax.plot(omega_Hz, xi, ls="-", lw=2, c=colors[2],
            label=r"$\xi=\alpha/(2\omega)+\beta\cdot\omega/2$")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$\omega$", size=35)
    ax.set_ylabel(r"$\xi$", size=35)
    ax.set_ylim([0, 0.006])
    ax.legend(loc="center", bbox_to_anchor=(0.5, .9))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


    exp_fld.mkdir(exist_ok=True)
    fig.savefig(exp_fld / "proportional_damping_sketch.pdf")
