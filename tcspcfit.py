""" Module for fitting TCSPC data.

Based on MATLAB code by Jörg Enderlein
https://www.uni-goettingen.de/en/513325.html
"""

# from math import *
# from statistics import *
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize
from matplotlib import pyplot as plt


class InputError(Exception):
    """Raised when input is incompatible with a given function"""
    pass


def convol(irf, x):
    """Performs a convolution of irf with x.

    Periodicity (period = size(x)) is assumed.
    irf should be a row vector.
    Output is a convolution of irf with each row of x.
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
        irf = irf[:, :p]
    y = np.float_(ifft(np.outer(np.ones(np.size(x, 0)), fft(irf)) * fft(x)))

    # t needs to have the same length as irf (n) in each case:
    if n <= p:
        t = np.arange(0, n)
    else:
        t = np.concatenate((np.arange(0, p), np.arange(0, n - p)))
        # t = np.concatenate((np.arange(0, p - 2), np.arange(0, n - p - 1)))

    # Either remove from y or add from start to end so that y has same length as irf.
    y = y.take(t, 1)
    return y


def lsfit(param, irf, y, period):
    """Returns the Least-Squares deviation between y and computed values.

    Assumes a function of the form:

    y =  yoffset + A1*convol(irf,exp(-t/tau1/(1-exp(-p/tau1))) + ...

    The only use case of lsfit is in the call to
    scipy.optimize.minimize(). minimize() minimizes the output of lsfit by
    varying the param vector.

    Arguments:
    param is 1D vector as required by minimize()
        param[0] -- color shift value between irf and y.
        param[1] -- irf offset.
        param[2:] -- decay times.

    irf -- measured Instrumental Response Function.
    y -- measured fluorescence decay curve.
    period -- time between two laser excitations (in number of TCSPC channels).

    Output:
    err -- least-squares error.
    """
    param = np.reshape(param, (1, -1))

    irflength = np.size(irf, 1)
    t = np.arange(1, irflength)
    tp = np.arange(1, period)
    colorshift = param[0, 0]
    tau = param[0, 2:].transpose()
    x = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1 - np.exp(-period / tau)))))
    irs = (1 - colorshift + np.floor(colorshift)) * irf[0, np.fmod(np.fmod(t - np.floor(colorshift) - 1, irflength) + irflength, irflength).astype(int)]\
        + (colorshift - np.floor(colorshift)) * irf[0, np.fmod(np.fmod(t - np.ceil(colorshift) - 1, irflength) + irflength, irflength).astype(int)]
    irs = np.reshape(irs, (1, -1))
    z = convol(irs, x.transpose())
    z = np.concatenate((np.ones((np.size(z, 0), 1)), z), axis=1)
    A, residuals, rank, s = np.linalg.lstsq(z.transpose(), y.transpose())
    z = np.matmul(z.transpose(), A)
    y = y.transpose()
    print(irflength - np.size(tau, 0))
    err = np.sum(((z-y)**2)/np.abs(z))/(irflength-np.size(tau, 0))
    return err


def fluofit(irf, y, p, dt, tau, lim=0, init=0, ploton=False):
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
    n = np.size(irf)
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
    lim = np.concatenate((np.zeros((1, np.size(tau, 1))), 100*np.ones((1, np.size(tau, 1)))))

    p = p/dt
    tp = np.arange(1, p).transpose()
    tau = tau/dt
    lim_min = lim[0:np.size(tau)]/dt
    lim_max = lim[np.size(tau)+1:]/dt
    t = np.arange(np.size(y))
    m = np.size(tau)
    tau = tau.flatten()
    print('tp', np.shape(tp), 'tau:', np.shape(tau))
    # x = np.matmul(np.exp(np.matmul(-(tp-1), (1/tau))), np.diagflat(1/(1-np.exp(-p/tau))))
    x = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1-np.exp(-p / tau)))))

    irs = (1-c+np.floor(c))*irf[(np.fmod(np.fmod(t-np.floor(c)-1, n)+n, n)).astype(int)] \
        + (c-np.floor(c))*irf[(np.fmod(np.fmod(t-np.ceil(c)-1, n)+n, n)).astype(int)]
    irs = np.reshape(irs, (1, -1))
    # z = convol(np.reshape(irs, (1, -1)), x.transpose())
    # z = np.reshape([np.ones([np.size(z,1), 1]), z], (-1, 1))
    # x = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1-np.exp(-p / tau)))))
    # irs = [(1 - c + np.floor(c)) * irf[0, np.fmod(np.fmod(t - np.floor(c) - 1, n) + n, n).astype(int)]\
    #     + (c - np.floor(c)) * irf[0, np.fmod(np.fmod(t - np.ceil(c) - 1, n) + n, n).astype(int)]]

    z = convol(irs, x.transpose())
    # print(np.shape(z), np.shape(y))
    # z = np.concatenate((np.ones((np.size(z, 0), 1)), z), axis=1)
    # print(np.shape(irf), np.shape(y))
    A, residuals, rank, s = np.linalg.lstsq(z.transpose(), y.transpose())
    z = np.matmul(z.transpose(), A)

    if init<2:
    #     disp('Fit =                Parameters =')
    #     param = np.reshape(np.insert(tau, 0, 0), (1, -1))
        param = np.insert(tau, 0, 999)
        print(param)
        # Decay times and Offset are assumed to be positive.
        # paramin = [-1/dt, lim_min]
        # paramax = [ 1/dt, lim_max]
        # print(paramin, paramax)
        # TODO: paramin and paramax should be incorporated into the minimisation below
        print(param)
        result = minimize(lsfit, param, args=(np.reshape(irf, (1, -1)), np.reshape(y, (1, -1)), p), method='Nelder-Mead')
        param = result.x
        print(result)
        c = param[0]
        # dc = dparam(1)  # TODO: Get errors out of minimisation
        tau = param[1:np.size(param)].flatten()
        # dtau = dparam[2:np.shape(param)]

        # x = np.matmul(np.exp(np.matmul(-(tp-1), (1./tau))), np.diag(1./(1-np.exp(-p/tau))))
        x = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1-np.exp(-p / tau)))))
        # irs = (1-c+np.floor(c))*irf(np.fmod(np.fmod(t-np.floor(c)-1, n)+n,n)+1) + (c-np.floor(c))*irf(np.fmod(np.fmod(t-np.ceil(c)-1, n)+n,n)+1)
        irs = (1-c+np.floor(c))*irf[(np.fmod(np.fmod(t-np.floor(c)-1, n)+n, n)).astype(int)] \
            + (c-np.floor(c))*irf[(np.fmod(np.fmod(t-np.ceil(c)-1, n)+n, n)).astype(int)]
        irs = np.reshape(irs, (1, -1))
        print(np.shape(x))
        z = convol(irs, x.transpose())
        # z = np.concatenate((np.ones((np.size(z, 0), 1)), z), axis=1)
        z = z.transpose()/np.ones((n, 1))*np.sum(z)
        # A = z\y
        A, residuals, rank, s = np.linalg.lstsq(z,y)
        bla = z*(np.ones((np.shape(z)[0], 1)))
        # A = np.reshape(A, (1, -1)).transpose()
        print('A:', np.shape(A), 'bla:', np.shape(bla))
        zz = np.matmul(bla, A)
        z = z*A
        z = z.transpose()
    #     dtau = dtau
    #     dc = dt*dc
    else:
        dtau = 0
        dc = 0
    chi = sum((y-z)**2./abs(z))/(n-m)
    t = dt*t
    tau = dt*tau.transpose()
    c = dt*c
    print(np.shape(zz))
    offset = zz[0]
    # A[0] = np.array([])
    return c, offset, A, tau, 0, 0, irs, zz, t, chi
