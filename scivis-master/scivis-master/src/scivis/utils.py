"""
Utility module for the scivis formatting functionalities

Utility functions to validate variable types and value
ranges, calculate axes parameters and round values.
"""

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["get_ax_size", "calc_text_pos"]


def get_ax_size(fig, ax):
    """Calculate the axes size in pixels.

    Parameters
    ----------
    fig : matplotlibe figure
        Figure for which to calculate the axes size.
    ax : matplotlibe axes
        Axes for which to calculate the axes size.

    Returns
    -------
    width : float
        Width of the figure in pixels.
    height : float
        height of the figure in pixels.

    """
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


def calc_text_pos(ax_lims, ax_size, base_pos, offset=-80):
    """Calculate the position of Text based on an offset in pixels.

    Parameters
    ----------
    ax_lims : ArrayLike
        Lower and upper limits of the axis.
    ax_size : ArrayLike
        Size of the axis in pixels.
    base_pos: float
        Base position of the text in the unit of the axis ticks.
    offset : float, optional
        Desired offset of the text from the base_pos in pixels.\n
        Defaults to -80.

    Returns
    -------
    pos : float
        Adjusted position of the text in the unit of the axis ticks.

    """
    val_len = ax_lims[1]-ax_lims[0]
    pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset
    if pixel_pos < 100:
        offset = -offset*4
        pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset
    elif ax_size-pixel_pos < 100:
        offset = -offset*4
        pixel_pos = (base_pos-ax_lims[0])/val_len*ax_size+offset

    return (pixel_pos)/ax_size*val_len + ax_lims[0]


def _round_sig_digits(val, sig_figs=2):
    """Round a value to a specified number of significant digits.

    Parameters
    ----------
    val : int or float
        Value to round.
    sig_figs : int, optional
        Number of significant digits to round to.\n
        Defaults to 2.

    Returns
    -------
        val : int or float
            Rounded value.

    """
    if val == 0:
        return 0
    return round(val, -int(np.floor(np.log10(abs(val)))) + (sig_figs - 1))


def _validate_arraylike_numeric(arr: ArrayLike, name: str = "",
                                ndim: int | None = None,
                                allow_neg: bool = True,
                                allow_zero: bool = True,
                                allow_non_finite: bool = True):
    """Validate if a variabe is array-like and contains only numeric values.

    Parameters
    ----------
    arr : ArrayLike
        The variable to check.
    name : str, optional
        Name of the variable (used for exception texts). If name='', the
        default name 'arr' is used.\n
        The default is "".
    ndim : int or None, optional
        Number of ndimension that arr is supposed to have.
        If None is specified, the ndimension is not checked.\n
        The default is None.
    allow_neg : bool, optional
        Selection whether negative values are allowed in the array.\n
        The default is True.
    allow_zero : bool, optional
        Selection whether zeros are allowed in the array.\n
        The default is True.
    allow_non_finite : bool, optional
        Selection whether non-finite values (Nan, infinite values) are allowed
        in the array.\n
        The default is True.


    Raises
    ------
    TypeError
        If name is not a string.\n
        If arr contains non-numeric values.
    ValueError
        If ndim is not a positive integer or None.\n
        If arr does not have the number of ndimensions specified in ndim.\n
        If arr is empty.\n
        If arr contains negative values (only if allow_neg=False).\n
        If arr contains zeros (only if allow_zero=False).

    Returns
    -------
    arr : np.ndarray
        The input array-like as a numpy array.

    """
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if not name:
        name = "Array"

    arr = np.asarray(arr)
    if ndim is not None:
        if not isinstance(ndim, int) \
                or not _validate_numeric(ndim,
                                         allow_neg=True, allow_zero=True):
            raise ValueError("ndim must be None or a positive integer.")

        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D.")

    if not np.issubdtype(arr.dtype, np.number) or not np.isrealobj(arr):
        raise TypeError(f"{name} must contain only numeric values.")

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not allow_neg and np.any(arr < 0):
        raise ValueError(f"{name} must contain only positive values.")

    if not allow_zero and np.any(arr == 0):
        raise ValueError(f"{name} must contain only non-zero values.")

    if not allow_non_finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")

    return arr


def _validate_numeric(val: int | float | np.number,
                      allow_neg: bool = True,
                      allow_zero: bool = True):
    """Validate if a variable is a scalar numeric value.

    Parameters
    ----------
    val : int | float | np.number
        Variable to check.
    allow_neg : bool, optional
        Selection whether the variable is allowed to be negative.\n
        The default is True.
    allow_zero : bool, optional
        Selection whether the variable is allowed to be zero.\n
        The default is True.

    Returns
    -------
    bool
        True if the variable is a scalar numeric value (and non-zero if
        allow_zero=False and positive if allow_neg=False), else False.

    """
    if not isinstance(val, (int, float, np.number)) or np.isnan(val) \
            or np.isinf(val):
        return False

    if not allow_neg and val < 0:
        return False

    if not allow_zero and val == 0:
        return False

    return True


def _replace_outside_nan(arr, min_val, max_val):
    """Replace values outside of range with Nan.

    Replace all values in a numpy array that lie outside the range
    [min_val, max_val] with NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input array (1D or 2D).
    min_val : float or int
        Minimum allowed value (inclusive).
    max_val : float or int
        Maximum allowed value (inclusive).

    Raises
    ------
    TypeError
        If arr is not a numpy ndarray or min/max are not numeric.
    ValueError
        If arr has more than 2 dimensions or if min_val > max_val.

    Returns
    -------
    np.ndarray
        A new numpy array with out-of-range values replaced by np.nan.
    """
    # --- Input Validation ---
    if not isinstance(arr, np.ndarray):
        raise TypeError("Expected 'arr' to be a numpy.ndarray, got "
                        f"{type(arr)} instead.")

    if arr.ndim not in (1, 2):
        raise ValueError(f"Expected a 1D or 2D array, got {arr.ndim}D "
                         "array instead.")

    if not all(isinstance(val, (int, float, np.number))
               for val in (min_val, max_val)):
        raise TypeError("min_val and max_val must be numeric (int or float).")

    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) cannot be greater than max_val"
                         f" ({max_val}).")

    result = arr.astype(float, copy=True)  # make sure we can insert NaN safely
    mask = (result < min_val) | (result > max_val)
    result[mask] = np.nan

    return result
