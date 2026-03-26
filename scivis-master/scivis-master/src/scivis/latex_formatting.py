"""Module for converting unformatted equations into latex math notation.

Offers capabilities to translage typical units into their respective command
from the siunitx latex package (incl. support of unit prefixes). Also
offers conversion of label text into latex math notation.
"""

import re

__all__ = ["convert_to_siunitx", "ensure_math", "latex_notation"]

# %% SIunitx mappings

siunitx_units_mapping = {
    "A": r"\ampere",
    "cd": r"\candela",
    "K": r"\kelvin",
    "Bq": r"\becquerel",
    "°C": r"\degreeCelsius",
    "C": r"\coulomb",
    "F": r"\farad",
    "Gy": r"\gray",
    "Hz": r"\hertz",
    "H": r"\henry",
    "J": r"\joule",
    "lm": r"\lumen",
    "kat": r"\katal",
    "lx": r"\lux",
    "N": r"\newton",
    "Ω": r"\ohm",
    "Ohm": r"\ohm",
    "Pa": r"\pascal",
    "rad": r"\radian",
    "S": r"\siemens",
    "Sv": r"\sievert",
    "sr": r"\steradian",
    "T": r"\tesla",
    "V": r"\volt",
    "W": r"\watt",
    "Wb": r"\weber",
    "au": r"\astronomicalunit",
    "B": r"\bel",
    "Da": r"\dalton",
    "d": r"\day",
    "dB": r"\decibel",
    "deg": r"\degree",
    "°": r"\degree",
    "%": r"\percent",
    "eV": r"\v",
    "ha": r"\hectare",
    "h": r"\hour",
    "L": r"\litre",
    "'": r"\arcminute",
    "min": r"\minute",
    "\"": r"\arcsecond",
    "Np": r"\neper",
    "t": r"\tonne",
    "fg": r"\fg",
    "pg": r"\pg",
    "ng": r"\ng",
    "ug": r"\ug",
    "mg": r"\mg",
    "g": r"\g",
    "kg": r"\kg",
    "pm": r"\pm",
    "nm": r"\nm",
    "μm": r"\um",
    "mm": r"\mm",
    "cm": r"\cm",
    "dm": r"\dm",
    "m": r"\m",
    "km": r"\km",
    "as": r"\as",
    "fs": r"\fs",
    "ps": r"\ps",
    "ns": r"\ns",
    "μs": r"\us",
    "ms": r"\ms",
    "s": r"\s",
    "fmol": r"\fmol",
    "pmol": r"\pmol",
    "nmol": r"\nmol",
    "μmol": r"\umol",
    "mmol": r"\mmol",
    "mol": r"\mol",
    "kmol": r"\kmol",
}

siunitx_prefixes_mapping = {
    "q": r"\quecto",
    "r": r"\ronto",
    "y": r"\yocto",
    "z": r"\zepto",
    "a": r"\atto",
    "f": r"\femto",
    "p": r"\pico",
    "n": r"\nano",
    "μ": r"\micro",
    r"\mu": r"\micro",
    "m": r"\milli",
    "c": r"\centi",
    "d": r"\deci",
    "da": r"\deca",
    "h": r"\hecto",
    "k": r"\kilo",
    "M": r"\mega",
    "G": r"\giga",
    "T": r"\tera",
    "P": r"\peta",
    "E": r"\exa",
    "Z": r"\zetta",
    "Y": r"\yotta",
    "R": r"\ronna",
    "Q": r"\quetta",
    }


def convert_to_siunitx(unit: str, brackets=True) -> str:
    """
    Convert a string containing a unit into a siunitx unit command.

    Parameters
    ----------
    unit : str
        Input string containing the unit.
    brackets : bool, optional
        Whether brackets should be added.\n
        The default is True.

    Returns
    -------
    str
        Unit converted to a siunitx command.

    """
    # Check if input is valid
    if not isinstance(unit, str) or not unit:
        return ""

    # Replace round brackets for exponents by curly brackets
    unit = re.sub(r"(\^\(([^)]*)\))",
                  lambda match: r"^{" + match.group(2) + "}",
                  unit)

    # Replace round brackets for subscript by curly brackets
    unit = re.sub(r"(_\(([^)]*)\))",
                  lambda match: r"_{" + match.group(2) + "}",
                  unit)

    # Find all units in the string
    units = set(re.findall(r"[a-zA-Z°\"']+", unit))
    if not units:
        return ""

    # Replace units by the corresponding siunitx command
    for u in units:
        if u in siunitx_units_mapping:
            unit = unit.replace(u, siunitx_units_mapping[u])
        elif len(u) >= 2 and u[0] in siunitx_prefixes_mapping \
                and u[1:] in siunitx_units_mapping:
            unit.replace(u,
                         siunitx_prefixes_mapping[u[0]]
                         + siunitx_units_mapping[u[1:]])

    if brackets:
        return r"$\:\left[\unit{" + unit + r"}\right]$"
    else:
        return r"$\:\unit{" + unit + r"}$"


def ensure_math(text):
    """Convert a mathematical formula into latex math notation.

    Parameters
    ----------
    text : str
        Text which represents a mathematical formula.

    Returns
    -------
    str
        Formula converted to latex math notation.

    """
    # Check if input is valid
    if not isinstance(text, str) or not text:
        return ""

    # Place curly brackets around subscripts
    text = re.sub(r"(_(\S*)\s)",
                  lambda match: r"_{" + match.group(2) + "}",
                  text)
    # Place curly brackets around subscripts
    text = re.sub(r"(_(\S*\Z))",
                  lambda match: r"_{" + match.group(2) + "}",
                  text)

    # Replace round brackets for subscript by curly brackets
    text = re.sub(r"(_\(([^)]*)\))",
                  lambda match: r"_{" + match.group(2) + "}",
                  text)

    # Place curly brackets around exponents
    text = re.sub(r"(\^(\S*)\s)",
                  lambda match: r"^{" + match.group(2) + "}",
                  text)

    # Replace round brackets for subscript by curly brackets
    text = re.sub(r"(\^(\S*\Z))",
                  lambda match: r"^{" + match.group(2) + "}",
                  text)

    # Replace round brackets for subscript by curly brackets
    text = re.sub(r"(\^\(([^)]*)\))",
                  lambda match: r"^{" + match.group(2) + "}",
                  text)

    # Replace white spaces with respective latex math command
    text = text.replace(" ", r"\:")

    # Ensure inline math mode
    text = "$" + text.replace("$", "") + "$"

    return text


def latex_notation(lbl="", unit="", brackets=True):
    """Convert an axis label into latex math notation.

    Parameters
    ----------
    lbl : str, optional
        The axis label symbol. The default is "".
    unit : str, optional
        Unit of the label. The default is "".
    brackets : bool, optional
        Whether brackets around the unit should be added.\n
        The default is True.

    Returns
    -------
    str
        Axis label in latex math notation.

    """
    return ensure_math(lbl) + convert_to_siunitx(unit, brackets=brackets)
