""" Module for fitting TCSPC data.

Based on MATLAB code by Jörg Enderlein
https://www.uni-goettingen.de/en/513325.html
"""

# from math import *
from statistics import *
import numpy as np
from scipy.fftpack import fft, ifft


def convol(irf, x):
    """Performs a convolution of irf with x.

    Periodicity (period = len(x)) is assumed.
    """

    mm = mean(irf[-11:-1])
    if x.ndim == 1:
        irf = irf.flatten()
        x = x.flatten()
        p = np.size(x,0)
    else:
        p = np.size(x, 1)
    print(p)
    n = np.size(irf, 0)
    if p > n:
        irf = np.append(irf, mm * np.ones(p - n))
    else:
        irf = irf[0:p]
    print(np.size(irf), np.size(x))
    print(np.ones(p), fft(irf))
    if x.ndim == 1:
        y = np.float_(ifft((fft(irf) * np.ones(p)) * fft(x)))
    else:
        print(irf)
        y = np.float_(ifft(np.outer(fft(irf), np.ones(np.size(x, 0))) * np.transpose(fft(x))))

    # t has the same length as irf:
    if n <= p:
        t = np.arange(0, n - 1)
    else:
        # print(np.arange(0, p-1), np.arange(0, n-p))
        print(n)
        t = np.concatenate((np.arange(0, p - 2), np.arange(0, n - p - 1)))
        print(t)

    y = y.take(t, 0)
    y = np.transpose(y)
    return y

