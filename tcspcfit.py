""" Module for fitting TCSPC data.

Based on MATLAB code by Jörg Enderlein
https://www.uni-goettingen.de/en/513325.html
"""

# from math import *
from statistics import *
import numpy as np
from scipy.fftpack import fft, ifft

bug = 'bug'

def convol(irf, x):
    """Performs a convolution of irf with x.

    Periodicity (period = len(x)) is assumed.
    """

    mm = mean(irf[-11:-1])
    if x.ndim == 1:
        # irf = irf.flatten()  # Needed in matlab, apparently not with numpy
        # x = x.flatten()
        p = np.size(x,0)
    else:
        p = np.size(x, 1)  # There should be a way to remove this if..else , maybe transpose the inputs
    n = np.size(irf, 0)
    if p > n:
        irf = np.append(irf, mm * np.ones(p - n))
    else:
        irf = irf[0:p]
    if x.ndim == 1:  # Again this if..else shouldn't be necessary
        y = np.float_(ifft((fft(irf) * np.ones(p)) * fft(x)))
    else:
        print(fft(irf))
        print(np.outer(fft(irf), np.ones(np.size(x, 0))))
        y = np.float_(ifft(np.outer(np.ones(np.size(x, 0)), fft(irf)) * fft(x)))

    # t needs to have the same length as irf (n) in each case:
    if n <= p:
        t = np.arange(0, n - 2)
    else:
        t = np.concatenate((np.arange(0, p - 2), np.arange(0, n - p - 1)))

    # Either remove from y or add from start to end so that y has same length as irf.
    if x.ndim == 1:  # This one should also go away.
        y = y.take(t, 0)
    else:
        y = y.take(t, 1)
    return y

