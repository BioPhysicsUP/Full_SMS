""" Module for fitting TCSPC data.

Based on MATLAB code by Jörg Enderlein
https://www.uni-goettingen.de/en/513325.html
"""

# from math import *
# from statistics import *
import numpy as np
from scipy.fftpack import fft, ifft


def convol(irf, x):
    """Performs a convolution of irf with x.

    Periodicity (period = len(x)) is assumed.
    """

    mm = np.mean(irf[-11:-1])
    try:
        p = np.size(x, 1)
        n = np.size(irf, 1)
    except IndexError:
        print('Input is not a row vector ([[a,b,..]]')
        raise

    if p > n:
        irf = np.append(irf, mm * np.ones(p - n))
    else:
        irf = irf[0:p]
    y = np.float_(ifft(np.outer(np.ones(np.size(x, 0)), fft(irf)) * fft(x)))

    # t needs to have the same length as irf (n) in each case:
    if n <= p:
        t = np.arange(0, n - 2)
    else:
        t = np.concatenate((np.arange(0, p - 2), np.arange(0, n - p - 1)))

    # Either remove from y or add from start to end so that y has same length as irf.
    y = y.take(t, 1)
    return y
