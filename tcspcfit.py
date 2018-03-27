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


def lsfit(param, irf, y, p):
    """Returns the Least-Squares deviation between the data y and the computed values.

    Assumes a function of the form:

    y =  yoffset + A(1)*convol(irf,exp(-t/tau(1)/(1-exp(-p/tau(1)))) + ...

    param(1) is the color shift value between irf and y.
    param(2) is the irf offset.
    param(3:...) are the decay times.
    irf is the measured Instrumental Response Function.
    y is the measured fluorescence decay curve.
    p is the time between to laser excitations (in number of TCSPC channels).
    """

    n = np.size(irf, 1)
    t = np.arange(1, n)
    tp = np.arange(1, p).transpose()
    c = param[1]
    tau = param[2:]
    tau = tau.flatten(1).transpose()
    x = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1-np.exp(-p / tau)))))
    irs = (1 - c + np.floor(c)) * irf[np.fmod(np.fmod(t - np.floor(c) - 1, n) + n, n) + 1]\
        + (c - np.floor(c)) * irf[np.fmod(np.fmod(t - np.ceil(c) - 1, n) + n, n) + 1]
    z = convol(irs, x)
    z = np.append(np.ones(np.size(z, 1), 1), z)
    A = np.linalg.solve(z, y)
    z = np.matmul(z, A)
    err = np.sum((z-y)**2/np.abs(z))/(n-np.size(tau, 1))
    return err, A, z