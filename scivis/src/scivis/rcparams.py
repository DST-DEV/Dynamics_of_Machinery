"""
Module with custom rcParam settings for matplotlib.

This module provides custom rcParam settings for matplotlib as well as methods
to scale and assemble these settings.
"""

import matplotlib as mpl
import numpy as np

from scivis import utils

__all__ = ["rcparams_figure", "rcparams_axes", "rcparams_line",
           "rcparams_ticks",  "rcparams_legend",  "rcparams_text",
           "rcparams_export",  "grid_style",  "latex_text_profile",
           "default_text_profile",  "mss",  "scale_factor",
           "_prepare_rcparams", "_scale_dict_params"]

rcparams_figure = {
    'figure.figsize': (16, 8),
    'figure.dpi': 144.0,
    'figure.edgecolor': 'white',
    'figure.facecolor': 'white',
    'figure.subplot.top': .94,    # Distance between suptitle and subplots
    'figure.constrained_layout.use': True
    }

rcparams_axes = {
    'axes.labelsize': 28,
    'axes.titlesize': 33,
    'axes.labelpad': 15,
    'axes.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'None',
    'axes.titlepad': 6.0,
    'axes.titlecolor': 'auto',
    'axes.titlelocation': 'center',
    'axes.prop_cycle': mpl.cycler('color', ['k', 'k', 'k', 'k', 'k', 'k'])
                      + mpl.cycler('linestyle',
                                   ['-', '--', '-.', ':',
                                    (0, (4, 2, 1, 2, 1, 2)),
                                    (0, (5, 2, 5, 2, 1, 2))])
    }

rcparams_line = {
    'lines.linestyle': "-",
    'lines.linewidth': 1.2,
    'lines.marker': "None",
    'lines.markersize': 10,
    'scatter.marker': "+",
    'lines.color': "k",
    'lines.markeredgecolor': "auto",
    'lines.markeredgewidth': 1.0,
    'lines.markerfacecolor': "auto",
    }

rcparams_ticks = {
    # Direction and visibility
    'xtick.direction': 'in',       # {"in", "out", "inout"}
    'ytick.direction': 'in',
    "xtick.top": True,
    "ytick.right": True,

    # Tick length (points)
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,

    # Tick width (points)
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.75,
    "ytick.minor.width": 0.75,

    # Padding between ticks and labels (points)
    "xtick.major.pad": 10,
    "ytick.major.pad": 10,
    "xtick.minor.pad": 5,
    "ytick.minor.pad": 5,

    # Label font properties
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "xtick.labelcolor": "inherit",
    "ytick.labelcolor": "inherit",
    "xtick.alignment": "center",
    "ytick.alignment": "center",
}

rcparams_legend = {
    'legend.borderaxespad': 0.5,
    'legend.borderpad': 0.4,
    'legend.columnspacing': 2.0,
    'legend.edgecolor': '0.8',
    'legend.facecolor': (1, 1, 1, .2),
    'legend.fancybox': True,
    'legend.fontsize': 22,
    'legend.framealpha': 0.8,
    'legend.frameon': True,
    'legend.handleheight': 0.7,
    'legend.handlelength': 2.0,
    'legend.handletextpad': 0.8,
    'legend.labelcolor': 'None',
    'legend.labelspacing': 0.5,
    'legend.loc': 'best',
    'legend.markerscale': 1.0,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.shadow': False,
    'legend.title_fontsize': None
    }

rcparams_text = {
    'text.color': 'black',
    'font.size': 22
    }

rcparams_export = {
    'savefig.bbox': "tight",
    'savefig.dpi': 300,
    'savefig.edgecolor': 'auto',
    'savefig.facecolor': 'auto',
    'savefig.format': 'pdf',
    'savefig.orientation': 'portrait',
    'savefig.transparent': True,
    'savefig.pad_inches': .1
    }

grid_style = {
    "major": {
        "axis": "both",
        "lw": 1.5,
        "color": "0.5",
        "alpha": .6,
        "zorder": 1
        },
    "minor": {
        "axis": "both",
        "lw": 1,
        "color": "0.65",
        "alpha": .6,
        "zorder": 1
        },
    }

latex_text_profile = {
    'text.usetex': True,  # Activate latex rendering
    'font.family': 'serif',  # LaTeX default font family
    'pgf.texsystem': "pdflatex",  # Use pdflatex for generating PDFs
    'pgf.rcfonts': False,  # Ignore Matplotlib's default font settings
    }

# Note: Don't ask me why but matplotlib will ignore these settings both in
# rc_contexts and if they are set globally within the plotting function.
# Ideally, these rcParams should not be adjusted globally but it is the only
# way it got it to work reliably for the figure legend
mpl.rcParams['text.latex.preamble'] = \
    "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
               r'\usepackage{siunitx}',
               # r'\usepackage{times}',  # Times new roman font
               r"\usepackage[utf8]{inputenc}",
               r"\usepackage[T1]{fontenc}"])
mpl.rcParams.update({"pgf.preamble": "\n".join([
        r'\usepackage{amsmath}',  # Optional, for math symbols
        r"\usepackage[utf8]{inputenc}",
        # r'\usepackage{times}',  # Times new roman font
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])})

default_text_profile = {
    "font.family": "serif",
    "font.sans-serif": ["Times New Roman"],
    'mathtext.fontset': 'cm'
    }

mss = {".": {"marker":".", "ms":15, "mec":"k"},
       "+": {"marker":"+", "ms":15, "mec":"k"},
       "x": {"marker":"x", "ms":15, "mec":"k"},
       "d": {"marker":"d", "ms":8, "mec":"k"},
       "1": {"marker":"1", "ms":15, "mec":"k"},
       "v": {"marker":"v", "ms":8, "mec":"k"},
       "default": {"marker":"+", "ms":10, "mec":"k"}
       }

scale_factor = .65


def _prepare_rcparams(latex=False, profile="fullsize", scale=1):
    """Assembles the individual rcParam dicts to a single dictionary.

    Parameters
    ----------
    latex : bool, optional
        Selection whether to use latex math interpreting. \n
        The default is False.
    profile : String, optional
        Profile settings for the scaling of font sizes, padding, etc..\n
        - "fullsize": Optimized for using the figure in full size (i.e.
          width = text width) on A4 paper in portrait\n
        - "halfsize": Optimized for using the figure in half size (i.e.
          width = 0.5 * text width) on A4 paper in portrait\n
        - "partsize": Optimized for using the figure in partial size (i.e.
          width = factor * text width) on A4 paper in portrait.
          The parameter 'scale' signifies the scale of the figure on the page\n
        - "custom_scale": Custom scaling factor for the rcParams\n
        The default is "fullsize".
    scale : {int, float, np.number}, optional
        Scaling factor. The default is 1.

    Raises
    ------
    ValueError
        If 'scale' is not a scalar, positive & non-zero value.
        If latex is not a boolean value

    Returns
    -------
    rc_profile : dict
        Scaled and assembled rcParams dict.
    """
    if not isinstance(profile, str):
        raise TypeError("profile must be a string.")

    if not isinstance(latex, bool):
        raise TypeError("latex must be a boolean value.")

    rc_profile = rcparams_figure | rcparams_axes | rcparams_line \
        | rcparams_ticks | rcparams_legend | rcparams_text \
        | rcparams_export

    # Scale plot appearance proportional to chosen scale
    rc_profile = _scale_dict_params(rc_profile, profile=profile, scale=scale)

    if latex:
        # Scale font sizes again (since latex handles them a bit differently)
        latex_scale_factor = 1.2
        rc_profile = {key: (val * latex_scale_factor
                            if (key.endswith(("size"))
                                and isinstance(val, (int, float, np.number)))
                            else val)
                      for key, val in rc_profile.items()
                      }

        rc_profile.update(latex_text_profile)
    else:
        rc_profile.update(default_text_profile)

    return rc_profile


def _scale_dict_params(param_dict, profile="fullsize", scale=1):
    """Scales the size parameters from the param_dict for a chosen profile.

    Automatically selects the rcParams associated with font sizes & padding
    and scales them according to the inputs.

    Parameters
    ----------
    param_dict : dict
        Dictionary of matplotlib rcParams.
    profile : String, optional
        Profile settings for the scaling.\n
        - "fullsize": Optimized for using the figure in full size (i.e.
          width = text width) on A4 paper in portrait\n
        - "halfsize": Optimized for using the figure in half size (i.e.
          width = 0.5 * text width) on A4 paper in portrait\n
        - "partsize": Optimized for using the figure in partial size (i.e.
          width = factor * text width) on A4 paper in portrait.
          The parameter 'scale' signifies the scale of the figure on the page\n
        - "custom_scale": Custom scaling factor for the rcParams\n
        The default is "fullsize".
    scale : {int, float, np.number}, optional
        Scaling factor. The default is 1.

    Raises
    ------
    ValueError
        If 'scale' is not a scalar, positive & non-zero value.

    Returns
    -------
    param_dict_scaled : dict
        rcParams with scaled values.
    """
    param_dict_scaled = param_dict.copy()

    if not isinstance(profile, str):
        raise TypeError("profile must be a string.")

    if profile != "fullsize":
        if profile == "halfsize":
            scale = .5
        else:
            if not utils._validate_numeric(scale, allow_neg=False,
                                           allow_zero=False):
                raise ValueError("scale factor must be a scalar "
                                 "non-negative positive numeric value.")

        if profile in ("partsize", "halfsize"):
            scale = (1/scale-1)*scale_factor + 1  # Empirical scale

        param_dict_scaled = {key: (val * scale
                                   if (key.endswith(("size", "pad", "width",
                                                     "spacing", "length",
                                                     "height", "pad_inches"))
                                       and isinstance(val,
                                                      (int, float, np.number)))
                                   else val)
                             for key, val in param_dict_scaled.items()
                             }
    return param_dict_scaled
