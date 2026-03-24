from pathlib import Path
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.optimize import least_squares

try: import scivis
except: pass
try: import numpyTolatex
except: pass

# %% User settings
show_plots = True
save_fig = True

# Plot settings
exp_fld = Path(__file__).parent / "plots" / "validation"
colors = ["#0D0887", "#CB4679", "#FFB300"]
profile = "partsize"
plot_scale = .62

# %% Model Parameters
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

# %% Calculate modal parameters of the mathematical model
def system_matrices(m, k):
    m1, m2, m3, m4, m5, m6 = m
    k1, k2, k3, k4, k5, k6 = k

    M = np.array([[m1, 0, 0, 0, 0, 0],
                   [m2, m2, 0, 0, 0, 0],
                   [m3, m3, m3, 0, 0, 0],
                   [m4, m4, m4, m4, 0, 0],
                   [m5, m5, m5, m5, m5, 0],
                   [m6, m6, m6, m6, 0, m6]])

    K = -np.array([[-k1, k2, 0, 0, 0, 0],
                   [0, -k2, k3, 0, 0, 0],
                   [0, 0, -k3, k4, 0, 0],
                   [0, 0, 0, -k4, k5, k6],
                   [0, 0, 0, 0, -k5, 0],
                   [0, 0, 0, 0, 0, -k6]])

    return M, K

def modal_analysis(M, K):
    n_dof = M.shape[0]

    zero_mat = np.zeros((n_dof, n_dof))
    A= np.block([[M, zero_mat],
                 [zero_mat, M]])
    B= np.block([[zero_mat, K],
                 [-M, zero_mat]])

    eigval, eigvec= eig(-B, A)
    eigfreqs= np.abs(eigval) / (2 * np.pi)  # Eigenfrequencies in [Hz]

    # Sort by eigenfrequency
    idx_sorted = np.argsort(eigfreqs)
    eigfreqs= eigfreqs[idx_sorted]
    eigval= eigval[idx_sorted]
    eigvec= eigvec[:, idx_sorted]

    # Normalize eigenvectors
    idx = N_DOF + np.argmax(np.abs(eigvec[n_dof:]), axis=0)
    scale = eigvec[idx, np.arange(eigvec.shape[1])]
    eigvec_norm = np.round(eigvec/ scale, 4)

    # Extract modal matrices shape matrix
    modeshapes= eigvec_norm[n_dof:, ::2].real

    return eigfreqs[::2], modeshapes

def fit_damping(xi, eifreqs):
    b = np.asarray(xi).reshape((-1, 1))

    A_alpha = 1/(2*eifreqs * (2 * np.pi)).reshape((-1, 1))
    A_beta = (eifreqs * (2 * np.pi) / 2).reshape((-1, 1))
    A_rayleigh = np.hstack([A_alpha, A_beta])

    sol_damping, *_ = np.linalg.lstsq(A_rayleigh, b)
    alpha, beta = sol_damping.flatten()

    return alpha, beta

m = [m1, m2, m3, m4, m5, m6]
k = [k1, k2, k3, k4, k5, k6]

M, K = system_matrices(m=m, k=k)
eigfreqs, modeshapes = modal_analysis(M, K)

alpha, beta = fit_damping(xi=[0.003]*N_DOF, eifreqs=eigfreqs)
D = alpha*M + beta*K
xi = alpha / (2*eigfreqs * 2*np.pi) + beta * (eigfreqs * 2*np.pi) / 2

# %% Compare modal parameters to experimental values

res_exp = pd.read_csv(Path(__file__).parent / "EMA_global_comparison_summary.csv")
eigfreqs_exp = res_exp["fn_freq_Hz"].values
xi_exp_steady = res_exp["zeta_ls_ref_acc"].values
xi_exp_transient = res_exp["zeta_time_logdec"].values

# %% Adjust model parameters with least squares fit to the eigenfrequencies
def state_func(x):
    m1, m2, m3, m4, m5, m6, k1, k2, k3, k4, k5, k6 = x

    M, K = system_matrices(m=[m1, m2, m3, m4, m5, m6],
                           k=[k1, k2, k3, k4, k5, k6])

    eigfreqs, modeshapes = modal_analysis(M, K)

    domega = eigfreqs_exp - eigfreqs

    return domega

lbounds = np.array([vi*0.85 for vi in [m1, m2, m3, m4, m5, m6,
                                       k1, k2, k3, k4, k5, k6]])
ubounds = np.array([vi*1.15 for vi in [m1, m2, m3, m4, m5, m6,
                                       k1, k2, k3, k4, k5, k6]])

res = least_squares(fun=state_func,
                    x0=[m1, m2, m3, m4, m5, m6, k1, k2, k3, k4, k5, k6],
                    bounds=(lbounds, ubounds))

m_adjusted = res.x[:6]
k_adjusted = res.x[6:]

M_adjusted, K_adjusted = system_matrices(m=m_adjusted, k=k_adjusted)
eigfreqs_adjusted, modeshapes_adjusted = modal_analysis(M_adjusted, K_adjusted)

domega_adjusted = eigfreqs_adjusted - eigfreqs_exp
domega_adjusted_rel = domega_adjusted / eigfreqs_exp


# Prepare tables for report
if "numpyTolatex" in sys.modules:

    eigfreq_table = np.vstack([eigfreqs_exp,
                               eigfreqs,
                               eigfreqs_adjusted,
                               domega_adjusted_rel*100]).T
    eigfreq_table_latex = numpyTolatex.np2latex(
        eigfreq_table,
        row_labels=[fr"$\omega_{i+1}$" for i in range(N_DOF)],
        body_only=True)

    print("Eigenfrequency comparison:")
    print(eigfreq_table_latex)
    print()

    param_table = np.vstack([m, m_adjusted, m_adjusted - m,
                             k, k_adjusted, k_adjusted - k]).T
    param_table_latex = numpyTolatex.np2latex(
        param_table,
        row_labels=[fr"${i+1}$" for i in range(N_DOF)],
        body_only=True)

    print("Parameter comparison:")
    print(param_table_latex)
    print()


# %% Calculate damping parameters for proportional damping
xi_fit = xi_exp_transient  # Damping ratios to use for fitting damping parameters
idx_fit = [0, 1, 2]  # Indices of damping ratios to consider for fitting the damping parameters

# Fit alpha to new data, keep beta from previous fitting
b = np.asarray(xi_fit[idx_fit]).reshape((-1, 1))
A_alpha = 1/(2*eigfreqs_adjusted[idx_fit] * (2 * np.pi)).reshape((-1, 1))
alpha_adjusted, *_ = np.linalg.lstsq(A_alpha, b)

alpha_adjusted = alpha_adjusted.item()
beta_adjusted = beta

xi_adjusted = alpha_adjusted / (2*eigfreqs_adjusted * 2*np.pi) \
    + beta_adjusted * (eigfreqs_adjusted * 2*np.pi) / 2

D_adjusted = alpha_adjusted*M_adjusted + beta_adjusted*K_adjusted

# Prepare tables for report
if "numpyTolatex" in sys.modules:

    xi_table = np.vstack([xi_exp_steady, xi_exp_transient, xi, xi_adjusted,
                          (xi_adjusted-xi_fit)/xi_fit*100]).T
    xi_table_latex = numpyTolatex.np2latex(
        xi_table,
        row_labels=[fr"${i+1}$" for i in range(N_DOF)],
        float_format=["%.5f"]*4 + ["%.2f"],
        body_only=True)

    print("Damping ratio comparison:")
    print(xi_table_latex)
    print()

if show_plots and "scivis" in sys.modules:
    eigfreqs_plot = np.linspace(0, np.ceil(np.max(eigfreqs)/10)*10, 300)
    eigfreqs_plot[0] = 1e-2
    omega_plot = eigfreqs_plot * 2*np.pi

    xi = alpha / (2*omega_plot) + beta * omega_plot / 2
    xi_adjusted =  alpha_adjusted / (2*omega_plot) \
        + beta_adjusted * omega_plot / 2

    # For comparison: fit alpha & beta to all experimental damping ratios
    alpha_adjusted_all, beta_adjusted_all = \
        fit_damping(xi=xi_fit, eifreqs=eigfreqs_adjusted)
    xi_adjusted_all =  alpha_adjusted_all / (2*omega_plot) \
        + beta_adjusted_all * omega_plot / 2

    rcparams = scivis.rcparams._prepare_rcparams(profile=profile,
                                                 scale=plot_scale)
    rcparams["legend.fontsize"] = rcparams["legend.fontsize"]*1.05

    with mpl.rc_context(rcparams):
        fig, ax = scivis.subplots(figsize=(20, 8))

        ylims_upper = np.max((np.ceil(np.max(xi_fit[idx_fit])/.001)*.001,
                              .02))
        fig, ax, _ = scivis.plot_line(ax=ax,
                                      x=eigfreqs_plot,
                                      y=np.stack([xi, xi_adjusted,
                                                  xi_adjusted_all], axis=0),
                                      plt_labels=[r"$\xi_\text{unadjusted}$",
                                                  r"$\xi_\text{adjusted}$",
                                                  r"$\xi_\text{adjusted, all}$"],
                                      ax_lims=[None, [0, ylims_upper]],
                                      ax_labels=[r"\omega", r"\xi"],
                                      ax_units=[r"\text{Hz}", ""],
                                      linestyles = ["-"]*2 + ["--"],
                                      linewidths=1.5,
                                      colors=colors[:2] + [colors[1]],
                                      show_legend=False,
                                      override_axes_settings=True,
                                      profile=profile, scale=plot_scale*1.08)
        ax.scatter(eigfreqs_exp, xi_fit,
                   marker="x", s=150, zorder=4,
                   label=r"$\xi_\text{experimental}$")

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1),
                  fontsize=rcparams["legend.fontsize"])

        for i, f in enumerate(eigfreqs_exp):
            vline, text, _ = scivis.axvline(
                ax=ax, x=f, text=r"$\omega_" + f"{i+1}" + r"$", c="0.5",
                rel_pos_x="center", rel_pos_y="top outside", margin_alpha=1)
            text.set_size(rcparams["legend.fontsize"]*1.05)  # Increase font size of annotation
            vline.set_zorder(1)  # Put lines in the background
        ax.axhline(0.003, ls="-.", c=".2", lw=2, zorder=1)

        mpl.pyplot.show()

        if save_fig:
            exp_fld.mkdir(exist_ok=True)
            fig.savefig(exp_fld / "proportional_damping_adjusted.svg")
