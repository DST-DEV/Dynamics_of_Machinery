"""Matplotlib-based scientific visualisation package.

Scivis is a highly customizeable and at the same time low effort visualisation
package for scientific purposes. It offers default plot settings for common
plots in scientific reports as well as full customization if necessary. All of
this is combined into a few comprehensive API commands
"""

# Built-in packages
import pathlib
from pathlib import Path

# Third-party packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# User-defined packages
import scivis.formatting as scifrmt
import scivis.utils as sciutils
import scivis.latex_formatting as ltx
from scivis import rcparams

__all__ = ["subplots", "plot_line", "axhline", "axvline"]


def subplots(nrows=1, ncols=1,  profile="fullsize", scale=1, latex=False,
             **kwargs):
    """Plot one or multiple lines.

    Parameters
    ----------
    nrows : int, optional
        Number of rows of the subplot grid.\n
        The default is 1
    ncols : int, optional
        Number of columns of the subplot grid.\n
        The default is 1
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
    scale : int | float | np.number, optional
        Scaling factor of font sizes & padding. Only applied if profile '
        partsize' or 'custom_scale' is selected. \n
        The default is 1.
    latex : bool, optional
        Selection whether to format use latex text interpretation.\n
        The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes._axes.Axes | list
        List of matplotlib.axes._axes.Axes objects or a single axes object, if
        nrows=1 and ncols=1.
    """
    if not all(isinstance(n, (int, np.integer)) for n in (nrows, ncols)):
        raise TypeError("Number of rows and columns must be integers")

    if not isinstance(profile, str):
        raise TypeError("profile must be a string.")

    if not isinstance(latex, bool):
        raise TypeError("latex must be a boolean value.")

    if "figsize" not in kwargs:
        figsize = rcparams.rcparams_figure["figure.figsize"]
        kwargs["figsize"] = (figsize[0]*ncols, figsize[1]*nrows)

    # Determine scale factor for the rcParams (based on ncols & figure size)
    figscale = max(kwargs["figsize"][0] / ncols
                   / rcparams.rcparams_figure["figure.figsize"][0],
                   kwargs["figsize"][1] / nrows
                   / rcparams.rcparams_figure["figure.figsize"][1])
    scale_sub = 1/ncols * 1 / figscale

    if profile == "fullsize":
        profile_sub = "partsize"
    elif profile == "halfsize":
        profile_sub = "partsize"
        scale_sub *= .5
    else:
        if not sciutils._validate_numeric(scale, allow_neg=False,
                                          allow_zero=False):
            raise ValueError("scale factor must be a scalar "
                             "non-negative positive numeric value.")
        if profile == "partsize":
            profile_sub = "partsize"
        else:
            profile_sub = "custom_scale"

        scale_sub *= scale

    rc_profile = rcparams._prepare_rcparams(latex=latex, profile=profile_sub,
                                            scale=scale_sub)

    if latex:
        # Save current rcParams
        rcparams_or = {
            "text.usetex": mpl.rcParams["text.usetex"],
            "pgf.texsystem": mpl.rcParams["pgf.texsystem"],
            "pgf.rcfonts": mpl.rcParams["pgf.rcfonts"],
            "text.latex.preamble": mpl.rcParams["text.latex.preamble"],
            "pgf.preamble": mpl.rcParams["pgf.preamble"],
            }

        # Note: These parameters are also part of the rc_profile. However,
        # leegends in matplotlib apparently ignore the rc_context settings.
        # Therefore they are changed globally and restored after plotting
        plt.rcParams.update(rcparams.latex_text_profile)

    with mpl.rc_context(rc_profile):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)

    if latex:
        # Restore original rcParams
        plt.rcParams.update(rcparams_or)

    return fig, ax


def plot_line(x, y, ax=None,
              plt_labels=None, show_legend=True, ax_labels=None, ax_units=None,
              colors=None, cmap=None, alpha=None,
              linestyles=None, linewidths=None, markers=None,
              ax_lims=None, margins=True, autoscale_y=True, rescale=True,
              overflow=True,
              ax_ticks=None, ax_tick_lbls=None,
              ax_ticks_minor=None, ax_tick_lbls_minor=None,
              ax_show_minor_ticks=True, ax_show_grid=True,
              ax_show_grid_minor=False,
              profile="fullsize", scale=1, latex=False,
              override_axes_settings=False,
              exp_fld=None, fname=None, ftype=".svg", savefig=False):
    """Plot one or multiple lines.

    Parameters
    ----------
    x : ArrayLike
        x-axis data. Must either be a vector, a 2d array or a higher
        dimensional array in which all but 2 dimensions have length 1.\n
        In the case of a 2d array, each row is intepreted as a separate line.
        The length of the vector / the number of elements on the second axis
        must match the y-axis values.
    y : ArrayLike
        y-axis data. Must either be a vector, a 2d array or a higher
        dimensional array in which all but 2 dimensions have length 1.\n
        In the case of a 2d array, each row is intepreted as a separate line.
        The length of the vector / the number of elements on the second axis
        must match the x-axis values.
    ax : matplotlib.axes._axes.Axes, optional
        Axes to plot the data on. If None is given, a new figure is created.\n
        The default is None.
    plt_labels : None | (tuple, list, np.ndarray) of str, optional
        Labels for each of the lines. The default is None.
    show_legend : bool, optional
        Selection whether a legend should be displayed. If no labels are
        specified via plt_labels, default names "var_<i>" are assigned for each
        line.\n
        The default is True.
    ax_labels : None | (tuple, list, np.ndarray) of str, optional
        Axis labels. Must be either None or a list of two Nones / strings.\n
        The default is None.
    ax_units : None | (tuple, list, np.ndarray) of str, optional
        Axis units. Must be either None or a list of two Nones / strings.\n
    colors : None | str | (tuple, list, np.ndarray), optional
        Line colors. Can be specified either as a single color which is applied
        globally to all lines, or as a sequence with one color for each line.
        Accepts any valid matplotlib color format.
        The default is None.
    cmap : None | str, optional
        Colormap to apply to the lines. This overwrites the color parameter.\n
        The default is None.
    alpha : None | int | float | (tuple, list, np.ndarray) of {int, float, numpy.number}, optional
        Transparency values. Can be specified either as a scalar global value
        or individually as a sequence with one element for each line.\n
        The default is None.
    linestyles : None | str | (tuple, list, np.ndarray) of str, optional
        Linestyles. Can be specified either as a scalar global value
        or individually as a sequence with one element for each line.\n
        The default is None.
    linewidths : None | int | float | (tuple, list, np.ndarray) of {int, float, numpy.number}, optional
        Linewidths. Can be specified either as a scalar global value
        or individually as a sequence with one element for each line.\n
        The default is None.
    markers : None | (tuple, list, np.ndarray) of str or dict, optional
        Markers. Can be specified either as string specifying the marker type,
        a list of marker types for (one for each line) or a list of
        dictionaries containing the marker parameters for each line.\n
        The default is None.
    ax_lims : None | (tuple, list, np.ndarray), optional
        Axis limits for the x- and y-axis. Must be either None or a 2-element
        Sequence in which each element is a 2-element Sequence consisting of
        the lower & upper axis limit.\n
        The first element pertains to the x-axis, the second to the y-axis.\n
        If no axis limits should be enforced, None can be given. This also
        applies to the elements of the sequence if limits should only be
        specified for one oxis.\n
        The default is None.
    margins : bool | (tuple, list, np.ndarray), optional
        Selection whether margins around the data should be displayed. \n
        Can either be specified globally as a single boolean, or individually
        for the x- and y-axis by providing a sequence with two boolean
        values.\n
        Note that this also adjusts the axis limits if they were set too large
        for the actual data.\n
        The default is True.
    autoscale_y : bool, optional
        Selection whether the y-axis should be scaled to match the plotted data
        within the x-axis limits. Thus only relevant, if x-axis limits are
        specified and no y-axis limits are given (If y-axis limits are
        specified, they overwrite this parameters).\n
        The default is True.
    rescale : bool, optional
        Selection whether the axis limits should be rescaled to fit all the
        data if the data is plotted on an existing axis.\n
        The default is True.
    overflow : bool | (tuple, list, np.ndarray), optional
        Selection whether overflow of the plotted values into the margins are
        allowed. This applies in the case that axis limits are specified and
        margins is set to true for at least one axis.\n
        Can either be specified globally as a single boolean, or individually
        for the x- and y-axis by providing a sequence with two boolean
        values.\n
        The default is True.
    ax_ticks : None | (tuple, list, np.ndarray) {int, float, np.number}, optional
        Major tick mark positions. The default is None.
    ax_tick_lbls : None | (tuple, list, np.ndarray) {int, float, np.number, str}, optional
        Major tick labels. The default is None.
    ax_ticks_minor : None | (tuple, list, np.ndarray) {int, float, np.number}, optional
        Minor tick mark positions. The default is None.
    ax_tick_lbls_minor : None | (tuple, list, np.ndarray) {int, float, np.number, str}, optional
        Minor tick labels. The default is None.
    ax_show_minor_ticks : bool, optional
        Selection whether to show major ticks.\n
        The default is True.
    ax_show_grid : bool, optional
        Selection whether to show major gird lines.\n
        The default is True.
    ax_show_grid_minor : bool, optional
        Selection whether to show minor gird lines. If ax_show_grid is False,
        the minor grid lines are automatically deactivated as well.\n
        The default is False.
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
    scale : int | float | np.number, optional
        Scaling factor of font sizes & padding. Only applied if profile '
        partsize' or 'custom_scale' is selected. \n
        The default is 1.
    latex : bool, optional
        Selection whether to format use latex text interpretation.\n
        The default is False.
    override_axes_profile : bool, optional
        Whether the profile settings of the axis should be overwritten, when an
        axes object is passed via the ax parameter.\n
        Note that this does not affect the settings for the latex selection.\n
        The default is False.
    exp_fld : None | str | pathlib.Path, optional
        Export folder to save the figure into. If None is given, the current
        working directory is used.\n
        The default is None.
    fname : str, optional
        Filename for saving the figure. If None is given, the filename is
        generated from the axis labels.\n
        The default is None.
    ftype : str, optional
        Filetype to use for exporting the figure. Must be a valid image file
        type that is supported by matplotlibs figure.savefig.\n
        The default is ".svg".
    savefig : bool, optional
        Whether the figure should be exported as an image file. \n
        The default is False.

    Raises
    ------
    TypeError
        If ax is neither None nor a matplotlib.axes._axes.Axes object.\n
        If exp_fld is neither None, nor a string nor a pathlib.Path object.\n
        If fname is neither None nor a string.\n
        If ftype is not a string.
    ValueError
        If the fname is None or an empty string and no axis label was specified
        for either the x- or y-axis. These labels are needed for the filename
        generation.
    OSError
        If the folder specified in exp_fld does not exist.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object or None if return_obj is set to False.
    ax : matplotlib.axes._axes.Axes
        Axes object or None if return_obj is set to False.
    fpath : None | pathlib.Path
        Export file path. If savefig=False, then None is returned.
    """
    # Prepare plot data
    x, y = scifrmt._prepare_xy_line(x, y)

    x, y, ax_lims = scifrmt._adjust_value_range(x, y, ax=ax, ax_lims=ax_lims,
                                                margins=margins,
                                                autoscale_y=autoscale_y,
                                                rescale=rescale,
                                                overflow=overflow)

    # Prepare style settings
    n_lines = max(x.shape[0], y.shape[0])
    plt_labels, axis_labels, col, alpha, ls, lw, markers = \
        scifrmt._resolve_style_line(
            n_lines=n_lines, plt_labels=plt_labels, show_legend=show_legend,
            ax_labels=ax_labels, ax_units=ax_units,
            latex=latex, colors=colors, cmap=cmap, alpha=alpha,
            linestyles=linestyles, linewidths=linewidths, markers=markers)

    # Plot
    rc_profile = rcparams._prepare_rcparams(latex=latex, profile=profile,
                                            scale=scale)

    if latex:
        # Save current rcParams
        rcparams_or = {
            "text.usetex": mpl.rcParams["text.usetex"],
            "pgf.texsystem": mpl.rcParams["pgf.texsystem"],
            "pgf.rcfonts": mpl.rcParams["pgf.rcfonts"],
            "text.latex.preamble": mpl.rcParams["text.latex.preamble"],
            "pgf.preamble": mpl.rcParams["pgf.preamble"],
            }

        # Note: These parameters are also part of the rc_profile. However,
        # leegends in matplotlib apparently ignore the rc_context settings.
        # Therefore they are changed globally and restored after plotting
        plt.rcParams.update(rcparams.latex_text_profile)

    with mpl.rc_context(rc_profile):
        # Create figure
        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, mpl.axes._axes.Axes):
            fig = ax.figure
            if override_axes_settings:
                scifrmt._apply_rcparams_to_axes(ax, latex=latex,
                                                profile=profile, scale=scale)
        else:
            raise TypeError("Axis must be a matplotlib axes object or None.")

        # Plot lines
        for i in range(x.shape[0]):
            if ls is not None:
                ax.plot(x[i, :], y[i, :], label=plt_labels[i], **markers[i],
                        lw=lw[i], ls=ls[i], c=col[i], alpha=alpha[i], zorder=2)
            else:
                ax.plot(x[i, :], y[i, :], label=plt_labels[i], **markers[i],
                        lw=lw[i], c=col[i], alpha=alpha[i], zorder=2)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles)>1:
                ax.legend()

        scifrmt._format_axes_line(
            ax=ax, ax_labels=axis_labels, ax_lims=ax_lims,
            ax_ticks=ax_ticks, ax_tick_lbls=ax_tick_lbls,
            ax_ticks_minor=ax_ticks_minor,
            ax_tick_lbls_minor=ax_tick_lbls_minor,
            ax_show_minor_ticks=ax_show_minor_ticks,
            ax_show_grid=ax_show_grid, ax_show_grid_minor=ax_show_grid_minor)

    if latex:
        # Restore original rcParams
        plt.rcParams.update(rcparams_or)

    # Export figure to file
    if savefig is True:
        if fname is None or (isinstance(fname, str) and len(fname) == 0):
            if any(ax_lbl_i is None for ax_lbl_i in ax_labels):
                raise ValueError("Export filename could not be determined. "
                                 "Either explicit filename via parameter "
                                 "'fname' or axis labels required")
            lbls = [lbl.replace("$", "").replace("\\", "")
                    for lbl in ax_labels]
            fname = f"{lbls[1]}_vs_{lbls[0]}"
        elif not isinstance(fname, str):
            raise TypeError("fname must be either None or str")

        # Check filetype
        if isinstance(ftype, str):
            if not ftype.startswith("."):
                ftype = "." + ftype

            if ftype not in (".png", ".jpg", ".svg", ".pdf", ".pgf"):
                raise ValueError("Unsupported export file type. Supported "
                                 "types are png, jpg, svg, pdf and pgf")
        else:
            raise TypeError("ftype must be str")

        # Check export folder
        if exp_fld is None:
            exp_fld = Path.cwd()
        if isinstance(exp_fld, (str, pathlib.PurePath)):
            if not Path(exp_fld).is_dir():
                raise OSError("Export folder does not exist at "
                              + str(Path(exp_fld).resolve()))
        else:
            raise TypeError("Export folder must be a str or a pathlib.Path "
                            "object")

        # Export figure
        fpath = Path(exp_fld, fname + ftype)
        fig.savefig(fpath)
    else:
        fpath = None

    return fig, ax, fpath



def axvline(ax, x, text=None, var_name=None, var_unit=None, latex=False,
            profile="fullsize", scale=1,
            n_decimals=2, rel_pos_x="left", rel_pos_y="bottom",
            ls="-.", c="k",
            margin=None, margin_alpha=.2, c_margin="inherit"):
    """Insert a vertical line at the specified x-position.

    A text label can additionally specified via the text parameter or via
    the var_name and var_unit parameter.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes:
        Axes to plot the line onto
    x : float
        x-position of the vertical line
    text : None or str, optional
        Text label for the axes.\n
        Alternatively, the variable name can be specified via the
        var_name parameter.\n
        The default is False.
    var_name: None or str, optional
        Name of the variable on the x-axis. This name is used for the
        label of the axes.
        Alternatively, the label text can be specified explicitely
        via the text parameter. If the text parameter has a value,
        then it is preferred over the variable name.\n
        The default is False.
    var_unit : None or str, optional
        Unit of the variable on the x-axis. This unit is included in
        the label of the axes if it is specified via the var_name
        parameter.
    latex : bool, optional
        Selection whether to format use latex text interpretation.\n
        The default is False.
    profile : String, optional
        Profile settings for the scaling.\n
        - "fullsize": Optimized for using the figure in full size (i.e.
          width = text width) on A4 paper in portrait\n
        - "halfsize": Optimized for using the figure in half size (i.e.
          width = 0.5 * text width) on A4 paper in portrait\n
        - "partsize": Optimized for using the figure in partial size (i.e.
          width = factor * text width) on A4 paper in portrait.
          The parameter 'scale' signifies the scale of the figure on the
          page\n
        - "custom_scale": Custom scaling factor for the rcParams\n
        The default is "fullsize".
    scale : int | float | np.number, optional
        Scaling factor of font sizes & padding. Only applied if profile '
        partsize' or 'custom_scale' is selected. \n
        The default is 1.
    n_decimals : int, optional
        Number of decimal points to display in the text label.
        Only relevant if the text label is specified via the var_name
        parameter.
    rel_pos_x : str, optional
        x-position of the text label relative to the vertical line.\n
        - "left": Label is plotted on the left of the line\n
        - "right": label is plotted on the right of the line\n
        The default is "left".
    rel_pos_xy : str, optional
        y-position of the text label relative to the axes.\n
        - "bottom": Label is plotted on the lower end of the y-axis\n
        - "top": label is plotted on the upper end of the y-axis\n
        The default is "bottom".
    ls : str, optional
        Linestyle of the line.\n
        The default is "-.".
    c : str | (tuple, list, np.ndarray), optional
        Color. Accepts any valid matplotlib color format.\n
        The default is "k".
    margin: None, (tuple, list, np.ndarray) of (int, float, np.number), optional
        Margin widths around the vertical line in the form {left width,
        right width}. If None is given, no margins are displayed.\n
        The default is None.
    margin_alpha : (int, float, np.number), optional
        Transparency of the margin highlight
    c_margin : str | (tuple, list, np.ndarray), optional
        Color of the margin. Accepts any valid matplotlib color format.\n
        Alternatively, "inherit" can be used to use the color of the line.\n
        The default is "inherit".

    Returns
    -------
    vline : matplotlib.lines.Line2D
        2D line vertical line.
    note : None | matplotlib.text.Annotation
        Annotation of the line or None if no label was specified.
    margin_rect : None | matplotlib.patches.Rectangle
        Rectangle with margins around the x-value or None if no margins were
        specified.
    """
    if not isinstance(latex, bool):
        raise TypeError("latex must be boolean.")

    # Prepare text label
    if text is None or (isinstance(text, str) and not text):
        if isinstance(var_name, str) and var_name:
            label = f"{var_name} = {round(x,n_decimals)}"

            if var_unit is not None and not isinstance(var_unit, str):
                raise TypeError("var_unit must be None or a str.")

            if isinstance(var_unit, str) and var_unit:
                if latex:
                    label = ltx.latex_notation(label, var_unit, brackets=False)
                else:
                    label += " " + var_unit
        else:
            label = ""
    elif isinstance(text, str):
        label = text
    else:
        raise TypeError("text must be None or a str.")

    # Prepare x-position of text
    if label:
        if not isinstance(rel_pos_x, str):
            raise TypeError("rel_pos_x must be a str.")

        if rel_pos_x == "left":
            ha = "right"
            x_text = -5
        elif rel_pos_x == "right":
            ha = "left"
            x_text = 5
        elif rel_pos_x == "center":
            ha = "center"
            x_text = 0
        else:
            raise ValueError("Relative x-position must be 'left' or "
                             + "'right'")

        if not isinstance(rel_pos_y, str):
            raise TypeError("rel_pos_y must be a str.")
        if rel_pos_y == "bottom":
            va = "bottom"
            y = ax.get_ylim()[0]
            y_text = 10
        elif rel_pos_y == "top":
            va = "top"
            y = ax.get_ylim()[-1]
            y_text = -10
        elif rel_pos_y == "top outside":
            va = "bottom"
            y = max(ax.get_ylim())
            y_text = 5
        else:
            raise ValueError("Relative y-position must be 'bottom' or "
                             + "'top")

    # Prepare color
    c = mpl.colors.to_hex(c)

    # Check margin value
    margin = scifrmt._check_style_variable(margin, name="margin",
                                           req_type=(int, float, np.number),
                                           n_elem=2)
    margin = [margin_i if margin_i is not None else 0 for margin_i in margin]
    if any(margin_i < 0 for margin_i in margin):
        raise ValueError("Margin widths must be positive.")

    if margin[1] + margin[0] > 0:
        # Check margin alpha value
        if not sciutils._validate_numeric(margin_alpha, allow_neg=False) \
                or margin_alpha>1:
            raise ValueError("Margin alpha must be a numeric value within "
                             "[0, 1].")

        # Check margin color
        if isinstance(c_margin, str) and c_margin == "inherit":
            c_margin = c
        else:
            c_margin = mpl.colors.to_hex(c_margin)

    # Prepare rcParam settings
    rc_profile = rcparams._prepare_rcparams(latex=latex, profile=profile,
                                            scale=scale)

    with mpl.rc_context(rc_profile):
        vline = ax.axvline(x, ls=ls, c=c)
        if margin[1] + margin[0] > 0:
            margin_rect = mpl.patches.Rectangle(
                (x-margin[0], ax.get_ylim()[0]),
                margin[0]+margin[1], ax.get_ylim()[1] - ax.get_ylim()[0],
                ec='none', fc=c_margin)
            margin_rect.set_alpha(margin_alpha)
            ax.add_patch(margin_rect)
        else:
            margin_rect = None

        if label:
            bbox = None
            if margin_alpha<1:
                if not ((margin[0] > 0 and x_text < 0)
                        or (margin[1] > 0 and x_text > 0)):
                    bbox=dict(facecolor='w', alpha=0.4, ls="none")

            arrowstyle = dict(arrowstyle="-", alpha=0)
            note = ax.annotate(
                text=label, xy=(x, y),
                xytext=(x_text, y_text),  textcoords="offset points",
                rotation="vertical", ha=ha, va=va, arrowprops=arrowstyle,
                bbox=bbox, c=c)
        else:
            note = None

    return vline, note, margin_rect


def axhline(ax, y, text=None, var_name=None, var_unit=None, latex=False,
            profile="fullsize", scale=1,
            n_decimals=2, rel_pos_x="left", rel_pos_y="below",
            ls="-.", c="k",
            margin=None, margin_alpha=.2):
    """Insert a horizontal line at the specified x-position.

    A text label can additionally specified via the text parameter or via
    the var_name and var_unit parameter.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Axes to plot the line onto
    y : float
        y-position of the horizontal line
    text : None or str, optional
        Text label for the axes.\n
        Alternatively, the variable name can be specified via the
        var_name parameter.\n
        The default is False.
    var_name : None or str, optional
        Name of the variable on the y-axis. This name is used for the
        label of the axes.
        Alternatively, the label text can be specified explicitely
        via the text parameter. If the text parameter has a value,
        then it is preferred over the variable name.\n
        The default is False.
    var_unit: None or str, optional
        Unit of the variable on the y-axis. This unit is included in
        the label of the axes if it is specified via the var_name
        parameter.
    latex : bool, optional
        Selection whether to format use latex text interpretation.\n
        The default is False.
    profile : String, optional
        Profile settings for the scaling.\n
        - "fullsize": Optimized for using the figure in full size (i.e.
          width = text width) on A4 paper in portrait\n
        - "halfsize": Optimized for using the figure in half size (i.e.
          width = 0.5 * text width) on A4 paper in portrait\n
        - "partsize": Optimized for using the figure in partial size (i.e.
          width = factor * text width) on A4 paper in portrait.
          The parameter 'scale' signifies the scale of the figure on the
          page\n
        - "custom_scale": Custom scaling factor for the rcParams\n
        The default is "fullsize".
    scale : int | float | np.number, optional
        Scaling factor of font sizes & padding. Only applied if profile '
        partsize' or 'custom_scale' is selected. \n
        The default is 1.
    n_decimals : int, optional
        Number of decimal points to display in the text label.
        Only relevant if the text label is specified via the var_name
        parameter.
    rel_pos_x : str, optional
        x-position of the text label relative to the axes.\n
        - "left": Label is plotted on the left end of the x-axis\n
        - "right": label is plotted on the right end of the x-axis\n
        The default is "left".
    rel_pos_xy : str, optional
        y-position of the text label relative to the vertical line.\n
        - "below": Label is plotted below the line\n
        - "above": label is plotted above the line\n
        The default is "below".
    ls : str, optional
        Linestyle of the line.\n
        The default is "-.".
    c : str | (tuple, list, np.ndarray), optional
        Color. Accepts any valid matplotlib color format.\n
        The default is "k".
    margin: None, (tuple, list, np.ndarray) of (int, float, np.number), optional
        Margin limits around the horizontal line in the form {lower limit,
        upper limit}. If None is given, no margins are displayed.\n
        The default is None.
    margin_alpha : (int, float, np.number), optional
        Transparency of the margin highlight
    c_margin : str | (tuple, list, np.ndarray), optional
        Color of the margin. Accepts any valid matplotlib color format.\n
        Alternatively, "inherit" can be used to use the color of the line.\n
        The default is "inherit".

    Returns
    -------
    hline : matplotlib.lines.Line2D
        2D line vertical line.
    note : None | matplotlib.text.Annotation
        Annotation of the line or None if no label was specified.
    margin_rect : None | matplotlib.patches.Rectangle
        Rectangle with margins around the x-value or None if no margins were
        specified.
    """
    if not isinstance(latex, bool):
        raise TypeError("latex must be boolean.")

    # Prepare text label
    if text is None or (isinstance(text, str) and not text):
        if isinstance(var_name, str) and var_name:
            label = f"{var_name} = {round(y,n_decimals)}"

            if var_unit is not None and not isinstance(var_unit, str):
                raise TypeError("var_unit must be None or a str.")

            if isinstance(var_unit, str) and var_unit:
                if latex:
                    label = ltx.latex_notation(label, var_unit, brackets=False)
                else:
                    label += " " + var_unit
        else:
            label = ""
    elif isinstance(text, str):
        label = text
    else:
        raise TypeError("text must be None or a str.")

    # Prepare x-position of text
    if label:
        if rel_pos_x == "left":
            x = ax.get_xlim()[0]
            x_text = 10
        elif rel_pos_x == "right":
            x = ax.get_xlim()[-1]
            x_text = -10
        else:
            raise ValueError("Relative x-position must be 'left' or "
                             + "'right'")

        if rel_pos_y == "below":
            va = "top"
            y_text = -5
        elif rel_pos_y == "above":
            va = "bottom"
            y_text = 5
        else:
            raise ValueError("Relative y-position must be 'below' or "
                             + "'above'")

    # Check margin value
    margin = scifrmt._check_style_variable(margin, name="margin",
                                           req_type=(int, float, np.number),
                                           n_elem=2)
    margin = [margin_i if margin_i is not None else 0 for margin_i in margin]
    if any(margin_i < 0 for margin_i in margin):
        raise ValueError("Margin widths must be positive.")

    # Check margin alpha value
    if not sciutils._validate_numeric(margin_alpha, allow_neg=False) \
            or margin_alpha>1:
        raise ValueError("Margin alpha must be a numeric value within [0, 1].")

    # Prepare color
    c = mpl.colors.to_hex(c)

    # Prepare rcParam settings
    rc_profile = rcparams._prepare_rcparams(latex=latex, profile=profile,
                                            scale=scale)

    with mpl.rc_context(rc_profile):
        hline = ax.axhline(y, ls=ls, c=c)

        if margin[1] + margin[0] > 0:
            margin_rect = mpl.patches.Rectangle(
                (ax.get_xlim()[0], y-margin[0]),
                ax.get_xlim()[1] - ax.get_xlim()[0], margin[0]+margin[1],
                ec='none', fc=c)
            margin_rect.set_alpha(margin_alpha)
            ax.add_patch(margin_rect)
        else:
            margin_rect = None

        if label:
            bbox = None
            if margin_alpha<1:
                if not ((margin[0] > 0 and y_text < 0)
                        or (margin[1] > 0 and y_text > 0)):
                    bbox=dict(facecolor='w', alpha=0.4, ls="none")

            arrowstyle = dict(arrowstyle="-", alpha=0)
            note = ax.annotate(
                text=label, xy=(x, y),
                xytext=(x_text, y_text), textcoords="offset points",
                ha=rel_pos_x, va=va, arrowprops=arrowstyle,
                bbox=bbox, c=c)
        else:
            note = None

    return hline, note, margin_rect
