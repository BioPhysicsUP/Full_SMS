"""Code based on H. Li and H, Yang, J. Phys. Chem. B, 123, 689-701 (2019).
https://github.com/PrincetonUniversity/GaussianEMBIC
Ported and adapted by Joshua L. Botha
"""

__docformat__ = 'NumPy'

import numpy as np


def find_cps(time_data: float, alpha: float) -> [int]:
    """Detects change points with an uncertainty of `alpha` % and returns the
    indexes of the change points, if any.

    Parameters
    ----------
    time_data: float
        Time trajectory in **ns**.
    alpha: float
        Error margin to be used.

    Returns
    -------
    cps: List of indexes of detected change points.
    """
