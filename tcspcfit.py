""" Module for fitting TCSPC data.

Based on MATLAB code by Jörg Enderlein
https://www.uni-goettingen.de/en/513325.html
"""

# from math import *
# from statistics import *
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
# from matplotlib import pyplot as plt

class InputError(Exception):
    """Custom exeption, raised when input is incompatible with a given function"""
    pass


def convol(irf, x):
    """Performs a convolution of irf with x.

    Periodicity (period = len(x)) is assumed.
    """

    mm = np.mean(irf[-11:-1])
    try:
        p = np.size(x, 1)
        n = np.size(irf, 1)
    except IndexError:
        raise InputError('Input is not a row vector ([[a,b,..]]')

    if p > n:
        irf = np.append(irf, mm * np.ones(p - n))
    else:
        irf = irf[0:p]
    y = np.float_(ifft(np.outer(np.ones(np.size(x, 0)), fft(irf)) * fft(x)))

    # t needs to have the same length as irf (n) in each case:
    if n <= p:
        t = np.arange(0, n)
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
    c = param[0, 1]
    tau = param[0, 2:]
    tau = tau.transpose()
    x = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1-np.exp(-p / tau)))))
    irs = [(1 - c + np.floor(c)) * irf[0, np.fmod(np.fmod(t - np.floor(c) - 1, n) + n, n).astype(int)]\
        + (c - np.floor(c)) * irf[0, np.fmod(np.fmod(t - np.ceil(c) - 1, n) + n, n).astype(int)]]

    z = convol(irs, x.transpose())
    z = np.concatenate((np.ones((np.size(z, 0), 1)), z), axis=1)
    A, residuals, rank, s = np.linalg.lstsq(z.transpose(), y.transpose())
    z = np.matmul(z.transpose(), A)
    y = y.transpose()
    err = np.sum((z-y)**2/np.abs(z))/(n-np.size(tau, 0))
    return err, A, z


def fluofit(irf, y, p, dt, tau, lim, init, ploton):
# The function FLUOFIT performs a fit of a multi-exponential decay curve.
# It is called by:
# [c, offset, A, tau, dc, doffset, dtau, irs, z, t, chi] = fluofit(irf, y, p, dt, tau, limits, init).
# The function arguments are:
# irf 	= 	Instrumental Response Function
# y 	= 	Fluorescence decay data
# p 	= 	Time between laser exciation pulses (in nanoseconds)
# dt 	= 	Time width of one TCSPC channel (in nanoseconds)
# tau 	= 	Initial guess times
# lim   = 	limits for the lifetimes guess times
# init	=	Whether to use a initial guess routine or not
#
# The return parameters are:
# c	=	Color Shift (time shift of the IRF with respect to the fluorescence curve)
# offset	=	Offset
# A	    =   Amplitudes of the different decay components
# tau	=	Decay times of the different decay components
# dc	=	Color shift error
# doffset	= 	Offset error
# dtau	=	Decay times error
# irs	=	IRF, shifted by the value of the colorshift
# zz	    Fitted fluorecence component curves
# t     =   time axis
# chi   =   chi2 value
#
# The program needs the following m-files: simplex.m, lsfit.m, mlfit.m, and convol.m.


    fitfun = 'lsfit'
    # fitfun = 'mlfit'

    irf = irf.flatten()
    offset = 0
    y = y.flatten()
    n = np.shape(irf)
    # if nargin>6:
    #     if isempty(init):
    #         init = 1
    # elif nargin>4:
    #     init = 0
    # else:
    #     init = 1
    init=0

    if init>0:
        # [cx, tau, ~, c] = DistFluofit(irf, y, p, dt, [-3 3], 0, 0)
        # cx = cx(:)'
        # tmp = cx>0
        # t = 1:length(tmp)
        # t1 = t(tmp(2:end)>tmp(1:end-1)) + 1
        # t2 = t(tmp(1:end-1)>tmp(2:end))
        # if length(t1)==length(t2)+1
        #     t1(end)=[]
        # end
        # if length(t2)==length(t1)+1
        #     t2(1)=[]
        # end
        # if t1(1)>t2(1)
        #     t1(end)=[]
        #     t2(1)=[]
        # end
        # tmp = []
        # for j=1:length(t1)
        #     tmp = [tmp cx(t1(j):t2(j))*tau(t1(j):t2(j))/sum(cx(t1(j):t2(j)))]
        # end
        # tau = tmp
        pass
    else:
        c = 0

    # if (nargin<6)||isempty(lim)
    lim = [np.zeros(1,np.shape(tau)), 100.*np.ones(1,np.shape(tau))]

    p = p/dt
    tp = np.arange(1, p).transpose()
    tau = tau.flatten().transpose()/dt
    lim_min = lim[np.arange(tau)]/dt
    lim_max = lim[np.shape(tau)+1:]/dt
    t = np.arange(y)
    m = np.shape(tau)
    x = np.exp(-(tp-1)*(1./tau))*np.diag(1./(1-np.exp(-p/tau)))
    irs = (1-c+np.floor(c))*irf(np.fmod(np.fmod(t-np.floor(c)-1, n)+n,n)+1) + (c-np.floor(c))*irf(np.floor(np.floor(t-np.ceil(c)-1, n)+n,n)+1)
    z = convol(irs, x)
    z = [np.ones(np.shape(z,1),1), z]
    #A = z\y
    A = np.linalg.lstsq(z,y)
    z = np.matmul(z, A)
    if init<2:
    #     disp('Fit =                Parameters =')
        param = [c, tau.transpose()]
        # Decay times and Offset are assumed to be positive.
        paramin = [-1/dt, lim_min]
        paramax = [ 1/dt, lim_max]
        [param, dparam] = minimize(fitfun, param, paramin, paramax, [], [], irf.transpose(), y.transpose(), p)
        c = param(1)
        dc = dparam(1)
        tau = param[2:np.shape(param)].transpose()
        dtau = dparam[2:np.shape(param)]
        x = np.exp(-(tp-1)*(1./tau))*np.diag(1./(1-np.exp(-p/tau)))
        irs = (1-c+np.floor(c))*irf(np.fmod(np.fmod(t-np.floor(c)-1, n)+n,n)+1) + (c-np.floor(c))*irf(np.fmod(np.fmod(t-np.ceil(c)-1, n)+n,n)+1)
        z = convol(irs, x)
        z = [np.ones(np.shape(z,1),1), z]
        z = z/(np.ones(n,1)*sum(z))
        #A = z\y
        A = np.linalg.lstsq(z,y)
        zz = z*(np.ones(np.shape(z,1),1)*A.transpose())
        z = z*A
    #     dtau = dtau
        dc = dt*dc
    else:
        dtau = 0
        dc = 0
    chi = sum((y-z)**2./abs(z))/(n-m)
    t = dt*t
    tau = dt*tau.transpose()
    c = dt*c
    offset = zz(1,1)
    A[1] = []