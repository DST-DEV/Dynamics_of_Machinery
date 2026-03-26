from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eig
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import scivis
try: import numpyTolatex
except: pass

# %% User settings
show_plots = True
save_fig = True

# Plot settings
exp_fld = Path(__file__).parent / "plots" / "modal_analysis"
colors = ["#0D0887", "#CB4679", "#FFB300"]
profile = "partsize"
plot_scale = .62

# %% Parameters
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

xi = .003

N_DOF = 6

# %% Linearized system matrices
# %%% Linearized at 90 deg
M_lin90 = np.array([[m1, 0, 0, 0, 0, 0],
                    [m2, m2, 0, 0, 0, 0],
                    [m3, m3, m3, 0, 0, 0],
                    [m4 + m5 - m6, m4 + m5 - m6, m4 + m5 - m6, m4 + m5 - m6, 0, 0],
                    [0, 0, 0, 0, m5, 0],
                    [0, 0, 0, 0, 0, m6]])
K_lin90 = -np.array([[-k1, k2, 0, 0, 0, 0],
                    [0, -k2, k3, 0, 0, 0],
                    [0, 0, -k3, k4, 0, 0],
                    [0, 0, 0, -k4, 0, 0],
                    [0, 0, 0, 0, -k5, 0],
                    [0, 0, 0, 0, 0, -k6]])

# %%% Linearized at 0 deg
M_lin0 = np.array([[m1, 0, 0, 0, 0, 0],
                   [m2, m2, 0, 0, 0, 0],
                   [m3, m3, m3, 0, 0, 0],
                   [m4, m4, m4, m4, 0, 0],
                   [m5, m5, m5, m5, m5, 0],
                   [m6, m6, m6, m6, 0, m6]])
K_lin0 = -np.array([[-k1, k2, 0, 0, 0, 0],
                   [0, -k2, k3, 0, 0, 0],
                   [0, 0, -k3, k4, 0, 0],
                   [0, 0, 0, -k4, k5, k6],
                   [0, 0, 0, 0, -k5, 0],
                   [0, 0, 0, 0, 0, -k6]])

# %% Solve eigenvalue problem
zero_mat = np.zeros_like(M_lin90)

# %%% Linearized at 90 deg
A_lin90 = np.block([[M_lin90, zero_mat],
                    [zero_mat, M_lin90]])
B_lin90 = np.block([[zero_mat, K_lin90],
                    [-M_lin90, zero_mat]])

eigval_lin90, eigvec_lin90 = eig(-B_lin90, A_lin90)
eigfreqs_lin90 = np.abs(eigval_lin90) / (2 * np.pi)  # Eigenfrequencies in [Hz]

# Sort by eigenfrequency
idx_sorted = np.argsort(eigfreqs_lin90)
eigfreqs_lin90 = eigfreqs_lin90[idx_sorted]
eigval_lin90 = eigval_lin90[idx_sorted]
eigvec_lin90 = eigvec_lin90[:, idx_sorted]

# Normalize eigenvectors
idx = 6 + np.argmax(np.abs(eigvec_lin90[6:]), axis=0)
scale = eigvec_lin90[idx, np.arange(eigvec_lin90.shape[1])]
eigvec_lin90_norm = np.round(eigvec_lin90 / scale, 4)

if "numpyTolatex" in sys.modules:
    tab_lin90 = numpyTolatex.np2latex(
        np.hstack([eigfreqs_lin90[::2].reshape((-1, 1)),
                   eigvec_lin90_norm[6:, ::2].real.T]),
        body_only=True)

    print("Eigenfrequency and eigenvalue table for 90 deg lineraization:")
    print(tab_lin90)
    print()

# %%% Linearized at 0 deg
A_lin0 = np.block([[M_lin0, zero_mat],
                    [zero_mat, M_lin0]])
B_lin0 = np.block([[zero_mat, K_lin0],
                    [-M_lin0, zero_mat]])

eigval_lin0, eigvec_lin0 = eig(-B_lin0, A_lin0)
eigfreqs_lin0 = np.abs(eigval_lin0) / (2 * np.pi)  # Eigenfrequencies in [Hz]

# Sort by eigenfrequency
idx_sorted = np.argsort(eigfreqs_lin0)
eigfreqs_lin0 = eigfreqs_lin0[idx_sorted]
eigval_lin0 = eigval_lin0[idx_sorted]
eigvec_lin0 = eigvec_lin0[:, idx_sorted]

# Normalize eigenvectors
idx = N_DOF + np.argmax(np.abs(eigvec_lin0[N_DOF:]), axis=0)
scale = eigvec_lin0[idx, np.arange(eigvec_lin0.shape[1])]
eigvec_lin0_norm = np.round(eigvec_lin0 / scale, 4)

# Extract modal matrices shape matrix
modeshapes_lin0 = eigvec_lin0_norm[N_DOF:, ::2].real

M_modal_lin0 = modeshapes_lin0.T @ M_lin0 @ modeshapes_lin0
K_modal_lin0 = modeshapes_lin0.T @ K_lin0 @ modeshapes_lin0

# Create latex table for eigenfrequencies and eigenvectors
if "numpyTolatex" in sys.modules:
    tab_lin0 = numpyTolatex.np2latex(
        np.hstack([eigfreqs_lin0[::2].reshape((-1, 1)),
                   eigvec_lin0_norm[N_DOF:, ::2].real.T]),
        body_only=True)

    print("Eigenfrequency and eigenvalue table for 0 deg lineraization:")
    print(tab_lin0)
    print()

# %% Calculate damping parameters for proportional damping
b = np.full((N_DOF,1), xi)

# Mass-proportional damping
A_D1 = 1/(2*np.abs(eigval_lin0)[::2]).reshape((-1, 1))

alpha_D1, *_ = np.linalg.lstsq(A_D1, b)
alpha_D1 = alpha_D1.item()

# Stiffness-proportional damping
A_D2 = (np.abs(eigval_lin0)[::2]/2).reshape((-1, 1))

beta_D2, *_ = np.linalg.lstsq(A_D2, b)
beta_D2 = beta_D2.item()

# Rayleigh damping
A_D3 = np.hstack([A_D1, A_D2])

sol_D3, *_ = np.linalg.lstsq(A_D3, b)
alpha_D3, beta_D3 = sol_D3.flatten()

if show_plots:
    eigfreqs_plot = np.linspace(0, np.ceil(np.max(eigfreqs_lin0)/10)*10, 300)
    eigfreqs_plot[0] = 1e-2
    omega_plot = eigfreqs_plot * 2*np.pi

    rcparams = scivis.rcparams._prepare_rcparams(profile=profile,
                                                 scale=plot_scale)
    rcparams["legend.fontsize"] = rcparams["legend.fontsize"]*1.05

    # Compare results for least squares fit
    xi_D1 = alpha_D1 / (2*omega_plot)
    xi_D2 = beta_D2 * omega_plot/ 2
    xi_D3_alpha = alpha_D3 / (2*omega_plot)
    xi_D3_beta =  beta_D3 * omega_plot/ 2
    xi_D3 = xi_D3_alpha + xi_D3_beta
    with mpl.rc_context(rcparams):
        fig, ax = scivis.subplots(figsize=(20,8))

        fig, ax, _ = scivis.plot_line(
            ax=ax,
            x=eigfreqs_plot,
            y=np.stack([xi_D1, xi_D2, xi_D3, xi_D3_alpha, xi_D3_beta], axis=0),
            plt_labels=[r"$\xi_a$", r"$\xi_b$", r"$\xi_c$",
                        r"$\xi_{c,\:\alpha}$", r"$\xi_{c,\:\beta}$"],
            ax_lims=[None, [0, .006]], ax_labels=[r"\omega", r"\xi"],
            ax_units=[r"\text{Hz}", ""],
            linestyles = ["-"]*3 + ["--", "-."], linewidths=1.5,
            colors=colors + [colors[-1]]*2,
            show_legend=False,
            override_axes_settings=True, profile=profile, scale=plot_scale*1.08
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1),
                  fontsize=rcparams["legend.fontsize"])

        for i, f in enumerate(eigfreqs_lin0[::2]):
            vline, text, _ = scivis.axvline(
                ax=ax, x=f, text=r"$\omega_" + f"{i+1}" + r"$", c="0.5",
                rel_pos_x="center", rel_pos_y="top outside")
            text.set_size(rcparams["legend.fontsize"]*1.05)  # Increase font size of annotation
            vline.set_zorder(1)  # Put lines in the background
        ax.axhline(xi, ls="-.", c=".2", lw=2, zorder=1)

        plt.show()

        if save_fig:
            exp_fld.mkdir(exist_ok=True)
            fig.savefig(exp_fld / "proportional_damping_lstsq.svg")

    # Compare results for fit with only first eigenmode
    alpha_D1_first = xi*2*np.abs(eigval_lin0)[0]
    beta_D2_first = xi*2/np.abs(eigval_lin0)[0]

    xi_D1_first = alpha_D1_first / (2*omega_plot)
    xi_D2_first = beta_D2_first * omega_plot/ 2
    with mpl.rc_context(rcparams):
        fig, ax = scivis.subplots(figsize=(20, 8))

        fig, ax, _ = scivis.plot_line(
            ax=ax,
            x=eigfreqs_plot,
            y=np.stack([xi_D1_first, xi_D1, xi_D2_first, xi_D2], axis=0),
            plt_labels=[r"$\xi_{a,1}$", r"$\xi_{a,lstsq}$",
                        r"$\xi_{b,1}$", r"$\xi_{b,lstsq}$"],
            ax_lims=[None, [0, .006]], ax_labels=[r"\omega", r"\xi"],
            ax_units=[r"\text{Hz}", ""],
            linestyles = ["-", "--"]*2, linewidths=1.5,
            colors=[colors[0]]*2 + [colors[1]]*2,
            show_legend=False,
            override_axes_settings=True, profile=profile, scale=plot_scale*1.08
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1),
                  fontsize=rcparams["legend.fontsize"])

        for i, f in enumerate(eigfreqs_lin0[::2]):
            vline, text, _ = scivis.axvline(
                ax=ax, x=f, text=r"$\omega_" + f"{i+1}" + r"$", c="0.5",
                rel_pos_x="center", rel_pos_y="top outside")
            text.set_size(rcparams["legend.fontsize"]*1.05)  # Increase font size of annotation
            vline.set_zorder(1)  # Put lines in the background
        ax.axhline(xi, ls="-.", c=".2", lw=2, zorder=1)

        plt.show()

        if save_fig:
            exp_fld.mkdir(exist_ok=True)
            fig.savefig(exp_fld / "proportional_damping_first.svg")

# %% Calculate damped eigenvalues and eigenvectors

alpha_final = xi*2*np.abs(eigval_lin0[0])
beta_final = 2*xi/np.abs(eigval_lin0[-1])
D_lin0 = alpha_final * M_lin0 + beta_final*K_lin0
B_lin0D = np.block([[D_lin0, K_lin0],
                    [-M_lin0, zero_mat]])

eigval_lin0D, eigvec_lin0D = eig(-B_lin0D, A_lin0)
eigfreqs_lin0D = np.abs(eigval_lin0D) / (2 * np.pi)  # Eigenfrequencies in [Hz]

# Sort by eigenfrequency
idx_sorted = np.argsort(eigfreqs_lin0D)
eigfreqs_lin0D = eigfreqs_lin0D[idx_sorted]
eigval_lin0D = eigval_lin0D[idx_sorted]
eigvec_lin0D = eigvec_lin0D[:, idx_sorted]


xi_D3 = alpha_D3 / (2*abs(eigval_lin0D)) + beta_D3 * abs(eigval_lin0D)/ 2
eigfreqs_lin0D2 = eigfreqs_lin0 * np.sqrt(1-xi_D3**2)

# Normalize eigenvectors
idx = N_DOF + np.argmax(np.abs(eigvec_lin0D[N_DOF:]), axis=0)
scale = eigvec_lin0D[idx, np.arange(eigvec_lin0D.shape[1])]
eigvec_lin0D_norm = np.round(eigvec_lin0D / scale, 4)

if "numpyTolatex" in sys.modules:
    tab_damp = numpyTolatex.np2latex(
        np.hstack([eigfreqs_lin0D[::2].reshape((-1, 1)),
                   eigvec_lin0D_norm[6:, ::2].real.T]),
        body_only=True)

    print("Damped Eigenfrequency and eigenvalue table for 0 deg lineraization:")
    print(tab_lin0)
    print()

if show_plots:
    rcparams = scivis.rcparams._prepare_rcparams(profile=profile,
                                                 scale=plot_scale)
    rcparams["legend.fontsize"] = rcparams["legend.fontsize"]*1.05

    # Plot damping ratio over frequency
    with mpl.rc_context(rcparams):
        xi_alpha_final = alpha_final / (2*omega_plot)
        xi_beta_final = beta_final * omega_plot/ 2
        xi_final = xi_alpha_final + xi_beta_final

        fig, ax = scivis.subplots(figsize=(20, 8))

        fig, ax, _ = scivis.plot_line(
            ax=ax,
            x=eigfreqs_plot,
            y=np.stack([xi_final, xi_alpha_final, xi_beta_final], axis=0),
            plt_labels=[r"$\xi_{final}$", r"$\xi_{final,\:\alpha}$",
                        r"$\xi_{final,\:\beta}$"],
            ax_lims=[None, [0, .006]], ax_labels=[r"\omega", r"\xi"],
            ax_units=[r"\text{Hz}", ""],
            linestyles = ["-"] + ["--"]*2, linewidths=1.5,
            colors=colors[0],
            show_legend=False,
            override_axes_settings=True, profile=profile, scale=plot_scale*1.08
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1),
                  fontsize=rcparams["legend.fontsize"])

        for i, f in enumerate(eigfreqs_lin0[::2]):
            vline, text, _ = scivis.axvline(
                ax=ax, x=f, text=r"$\omega_" + f"{i+1}" + r"$", c="0.5",
                rel_pos_x="center", rel_pos_y="top outside")
            text.set_size(rcparams["legend.fontsize"]*1.05)  # Increase font size of annotation
            vline.set_zorder(1)  # Put lines in the background
        ax.axhline(xi, ls="-.", c=".2", lw=2, zorder=1)

        plt.show()

        if save_fig:
            exp_fld.mkdir(exist_ok=True)
            fig.savefig(exp_fld / "proportional_damping_final.svg")

# %% Visualize the modeshapes
def visualize_mode(modeshape):
    # Illustrate mode shapes
    y_beams = np.array([[0, l1],
                        [l1, l1+l2],
                        [l1+l2, l1+l2+l3],
                        [l1+l2+l3, l1+l2+l3+l4],
                        [l1+l2+l3+l4, l1+l2+l3+l4+l5],
                        [l1+l2+l3+l4, l1+l2+l3+l4-l6]]).T

    x_beams = np.zeros((2, N_DOF))
    x_beams[0, 1:-1] = np.cumsum(modeshape[:-2])
    x_beams[0, -1] = x_beams[0, -2]
    x_beams[1, :-1] = np.cumsum(modeshape[:-1])
    x_beams[1, -1] = x_beams[0, -2] + modeshape[-1]

    x_beams /= np.max(np.abs(x_beams))  # Normalize

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_aspect(2)
    ax.set_xlim([-1.05, 1.05])
    ax.axis("off")

    for j in range(6):
        ax.plot(x_beams[:, j], y_beams[:, j],
                ls="-", c="b", marker="o", ms=15)
        ax.plot(-x_beams[:, j], y_beams[:, j],
                ls="-", c="b", marker="o", ms=15, alpha=.15)

        # Reference undeformed centerline
        ax.plot([0, 0], y_beams[:, j],
                ls="--", c="k", marker="o", ms=15)

    return fig, ax


# =============================================================================
# def visualize_mode_splines(modeshape):
#     y_nodes = np.array([0, l1, l1+l2, l1+l2+l3, l1+l2+l3+l4,
#                         l1+l2+l3+l4+l5, l1+l2+l3+l4-l6])
#
#     x_nodes = np.zeros(N_DOF+1)
#     x_nodes[1:-1] = np.cumsum(modeshape[:-1])
#     x_nodes[-1] = x_nodes[-3] + modeshape[-1]
#
#     x_nodes /= np.max(np.abs(x_nodes))
#
#     fig, ax = plt.subplots(figsize=(10,14))
#     ax.set_aspect(2)
#     ax.set_xlim([-1.05, 1.05])
#     ax.axis("off")
#
#     # Tower splines (1-4)
#     for tower_idx in range(4):
#         x_tower = x_nodes[[tower_idx, tower_idx + 1]]
#         y_tower = y_nodes[[tower_idx, tower_idx + 1]]
#
#         s = np.linspace(0, 1, len(x_tower))
#         s_dense = np.linspace(0, 1, 200)
#
#         sx = CubicSpline(s, x_tower, bc_type=((1, 0.0), (1, 0.0)))
#         sy = CubicSpline(s, y_tower)
#
#         ax.plot(sx(s_dense), sy(s_dense), c="b", lw=2)
#         ax.plot(-sx(s_dense), sy(s_dense), c="b", lw=2, alpha=.15)
#
#     # Blade spline (5-4-6)
#     beam_idx = [5, 4, 6]
#
#     x_beam = x_nodes[beam_idx]
#     y_beam = y_nodes[beam_idx]
#
#     s = np.linspace(0, 1, len(x_beam))
#     sx = CubicSpline(s, x_beam)
#     sy = CubicSpline(s, y_beam)
#
#     ax.plot(sx(s_dense), sy(s_dense), c="b", lw=2)
#     ax.plot(-sx(s_dense), sy(s_dense), c="b", lw=2, alpha=.15)
#
#     # Plot masses
#     ax.scatter(x_nodes, y_nodes, c="b", s=130, zorder=3)
#     ax.scatter(-x_nodes, y_nodes, c="b", s=130, alpha=.15)
#
#     # Reference undeformed centerline
#     ax.plot([0]*len(y_nodes), y_nodes, ls="--", c="k", marker="o", ms=13)
# =============================================================================

def visualize_mode_splines(modeshape, n_steps=2, cmap="coolwarm"):
    y_nodes = np.array([0, l1, l1+l2, l1+l2+l3, l1+l2+l3+l4,
                        l1+l2+l3+l4+l5, l1+l2+l3+l4-l6])

    x_nodes = np.zeros(N_DOF+1)
    x_nodes[1:-1] = np.cumsum(modeshape[:-1])
    x_nodes[-1] = x_nodes[-3] + modeshape[-1]

    x_nodes /= np.max(np.abs(x_nodes))

    if n_steps > 1:
        interp_points = np.tile(np.linspace(0, 1, n_steps), (N_DOF+1, 1)).T
        x_nodes_interp = (-x_nodes + 2*x_nodes*interp_points)
        colors = mpl.colormaps[cmap](np.linspace(0, 1, n_steps))
    elif n_steps == 1:
        x_nodes_interp = np.atleast_2d(x_nodes)
        colors = [mpl.colormaps[cmap](1.0)]  # Last color of colormap
    else:
        raise ValueError("n_steps must be a positive integer value.")

    fig, ax = plt.subplots(figsize=(10,14))
    ax.set_aspect(2)
    ax.set_xlim([-1.05, 1.05])
    ax.axis("off")

    for step in range(n_steps):
        x_nodes_i = x_nodes_interp[step]

        # Tower splines (1-4)
        for tower_idx in range(4):
            x_tower = x_nodes_i[[tower_idx, tower_idx + 1]]
            y_tower = y_nodes[[tower_idx, tower_idx + 1]]

            s = np.linspace(0, 1, len(x_tower))
            s_dense = np.linspace(0, 1, 200)

            sx = CubicSpline(s, x_tower, bc_type=((1, 0.0), (1, 0.0)))
            sy = CubicSpline(s, y_tower)

            ax.plot(sx(s_dense), sy(s_dense), c=colors[step], lw=2)
            ax.plot(-sx(s_dense), sy(s_dense), c=colors[step], lw=2, alpha=.15)

        # Blade spline (5-4-6)
        beam_idx = [5, 4, 6]

        x_beam = x_nodes_i[beam_idx]
        y_beam = y_nodes[beam_idx]

        s = np.linspace(0, 1, len(x_beam))
        sx = CubicSpline(s, x_beam)
        sy = CubicSpline(s, y_beam)

        ax.plot(sx(s_dense), sy(s_dense), c=colors[step], lw=2)
        ax.plot(-sx(s_dense), sy(s_dense), c=colors[step], lw=2, alpha=.15)

        # Plot masses
        ax.scatter(x_nodes_i, y_nodes, color=colors[step], s=130, zorder=3)
        ax.scatter(-x_nodes_i, y_nodes, color=colors[step], s=130, alpha=.15)

    # Reference undeformed centerline
    ax.plot([0]*len(y_nodes), y_nodes, ls="--", c="k", marker="o", ms=13)


for i in range(N_DOF):
    mode = eigvec_lin0D_norm[N_DOF:, i*2].real

    visualize_mode(mode)
    visualize_mode_splines(mode, n_steps=1)


# %% Save results
