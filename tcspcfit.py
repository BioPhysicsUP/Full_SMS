""" Module for fitting TCSPC data.

Based on MATLAB code by Jörg Enderlein
https://www.uni-goettingen.de/en/513325.html
"""

# from math import *
# from statistics import *
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize, curve_fit, nnls
from scipy.signal import convolve
from matplotlib import pyplot as plt


def makerow(vector):
    """Reshape 1D vector into a 2D row"""
    return np.reshape(vector, (1, -1))


def create_exp(tp, tau, period, irf, irflength, t):
    """Create a convolved exponential decay function"""
    decay = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1 - np.exp(-period / tau)))))
    # irs = colorshift(irf, cshift, irflength, t)
    calculated = convol(irf, decay.T)
    return calculated, irf


def convol(irf, decay):
    """Performs a convolution of irf with decay.

    Periodicity (period = size(decay)) is assumed.

    Arguments:
    irf -- row vector or 1D
    decay -- 1D or array with data in rows

    Output
    y -- convolution of irf with each row of decay.
    """

    # Make sure irf (and decay if 1D) are row vectors:
    irf = makerow(irf)
    if decay.ndim == 1:
        irf = makerow(decay)

    # Make irf the same length as decay:
    mm = np.mean(irf[-11:])
    decaylength = np.size(decay, 1)
    irflength = np.size(irf, 1)
    if decaylength > irflength:
        irf = np.append(irf, mm * np.ones(decaylength - irflength))
    else:
        irf = irf[:, :decaylength]

    # Duplicate rows of irf so dimensions are the same as decay and convolve:
    y = np.float_(np.real(ifft(np.outer(np.ones(np.size(decay, 0)), fft(irf)) * fft(decay))))

    # t needs to have the same length as irf (irflength) in each case:
    if irflength <= decaylength:
        t = np.arange(0, irflength)
    else:
        t = np.concatenate((np.arange(0, decaylength), np.arange(0, irflength - decaylength)))

    # Either remove from y or add from start to end so that y has same length as irf:
    y = y.take(t, 1)
    return y


def colorshift(irf, shift, irflength, t):
    """Shift irf left or right 'periodically'.

    A shift past the start or end results in those values of irf
    'wrapping around'.

    Arguments:
    irf -- row vector or 1D
    shift -- float
    irflength -- float
    t -- 1D vector

    Output:
    irs -- shifted irf, as row vector
    """
    irf = irf.flatten()
    irflength = np.size(irf)
    t = np.arange(irflength)
    new_index_left = np.fmod(np.fmod(t - np.floor(shift) - 1, irflength) + irflength, irflength).astype(int)
    new_index_right = np.fmod(np.fmod(t - np.ceil(shift) - 1, irflength) + irflength, irflength).astype(int)
    integer_left_shift = irf[new_index_left]
    integer_right_shift = irf[new_index_right]
    irs = (1 - shift + np.floor(shift)) * integer_left_shift + (shift - np.floor(shift)) * integer_right_shift
    return makerow(irs)


def fitfunc(model_in, tau1, tau2, scale, a1, a2):
    irflength = int(model_in[-1])
    startpoint = int(model_in[-2:-1])
    t = model_in[:irflength]
    irf = model_in[irflength:-2]
    # irs = colorshift(irf.flatten(), shift, np.size(irf), t)
    # irs = irs.flatten()

    model = a1 * np.exp(-t/tau1) + a2 * np.exp(-t/tau2)

    convd = convolve(irf, model)
    convd = convd * scale/np.max(convd)
    convd = convd[:irflength-startpoint]
    return convd


def lsfit(param, irf, measured, period):
    """Returns the Least-Squares deviation between measured and computed values.

    Assumes a function of the form:

    measured =  yoffset + A1*convol(irf,exp(-t/tau1/(1-exp(-p/tau1))) + ...

    The only use case of lsfit is in the call to
    scipy.optimize.minimize(), which minimizes the output of lsfit by
    varying the param vector.

    Arguments:
    param is 1D vector as required by minimize()
        param[0] -- color shift value between irf and measured.
        param[1] -- irf offset.
        param[2:] -- decay times.

    irf -- measured Instrumental Response Function.
    measured -- measured fluorescence decay curve.
    period -- time between two laser excitations (in number of TCSPC channels).

    Output:
    err -- least-squares error.
    """

    measured = measured.flatten()
    irf = irf.flatten()

    irflength = np.size(irf)
    t = np.arange(irflength)
    tp = np.arange(1, period)
    # cshift = param[0]
    # offset = param[1]
    # tau = param[2:]
    tau = param

    # Create data from parameters
    calculated, irs = create_exp(tp, tau, period, irf, irflength, t)

    # Calculate least-squares error between calculated and measured data
    amplitudes, residuals, rank, s = np.linalg.lstsq(calculated.T, measured, rcond=None)
    calculated = np.matmul(calculated.T, amplitudes)
    # err = np.sum(((calculated - measured) ** 2) / np.abs(calculated)) / (irflength - np.size(tau, 0))
    ind = ((measured > 0) & (calculated > 0)) | ((measured < 0) & (calculated < 0))
    # raise TypeError
    err = sum(measured[ind] * np.log(measured[ind] / calculated[ind]) - measured[ind] + calculated[ind]) / (irflength - np.size(tau))
    # print(err)
    return err


def distfluofit(irf, measured, period, channelwidth, cshift_bounds=[-3, 3], choose=False, ntau=100):
    """Quickly fit a multiexponential decay to use as 'initial guess'

        The function aims to identify the number of lifetimes in the
        measured data, as well as the values of the lifetimes. The result
        can be used as an initial guess for fluofit.

    Arguments:
    irf -- Instrumental Response Function measured -- Fluorescence decay data
    period -- Time between laser exciation pulses (in nanoseconds)
    channelwidth -- Time width of one TCSPC channel (in nanoseconds)
    tau -- Initial guess times
    taubounds -- limits for the lifetimes guess times - defaults to 0<tau<100
                 format: [[tau1_min, tau1_max], [tau2_min, tau2_max], ...]
    init -- Whether to use a initial guess routine or not

    Output:
    peak_tau	-- Decay times of the different decay components
    TODO: Add all other outputs
    c -- Color Shift (time shift of the IRF w.r.t. the fluorescence curve)
    offset -- Offset
    amplitudes -- Amplitudes of the different decay components
    dc -- Color shift error
    doffset -- Offset error
    dtau -- Decay times error
    irs -- IRF, shifted by the value of the colorshift
    separated_decays -- Fitted fluorecence component curves
    t -- time axis
    chisquared -- chi squared value

    """
    irf = irf.flatten()
    irflength = np.size(irf)
    tp = channelwidth*np.arange(1, period/channelwidth)  # Time index for whole window
    t = np.arange(irflength)  # Time index for IRF
    nrange = np.arange(ntau)
    # Distribution of inverse decay times:
    tau = (1/channelwidth) / np.exp(nrange / ntau * np.log(period / channelwidth))
    decays = convol(irf, np.exp(np.outer(tau, -tp)))
    decays = decays / np.sum(decays)
    amplitudes, residuals = nnls(decays.T, measured.flatten())
    tau = 1/tau

    peak_amplitudes = amplitudes > 0.1 * np.max(amplitudes)  # Pick out peaks
    peak_amplitudes = peak_amplitudes.flatten()
    t = np.arange(1, np.size(peak_amplitudes))

    # t1 are the 'start points' and t2 are the 'end points' of the peaks
    t1 = t[peak_amplitudes[1:] > peak_amplitudes[:-1]] + 1
    t2 = t[peak_amplitudes[:-1] > peak_amplitudes[1:]]
    # Make sure there isn't a peak at the edge:
    if t1[0] > t2[0]:
        t2 = np.delete(t2, 0)
    if t1[-1] > t2[-1]:
        t1 = np.delete(t1, -1)
    if np.size(t1) == np.size(t2) + 1:
        t2 = np.delete(t2, 0)
    if np.size(t2) == np.size(t1) + 1:
        t1 = np.delete(t1, -1)

    peak_tau = np.array([])
    #  Calculate weighted average tau for each peak:
    for j in range(np.size(t1)):
        peak_tau = np.append(peak_tau, np.dot(amplitudes[t1[j]-1:t2[j]], tau[t1[j]-1:t2[j]]) / np.sum(amplitudes[t1[j]-1:t2[j]]))

    return peak_tau


def fluofit(irf, measured, t, window, channelwidth, tau=None, taubounds=None, startpoint=0, init=0, ploton=False, method='Nelder-Mead'):
    """Fit of a multi-exponential decay curve.

    Arguments:
    irf -- Instrumental Response Function measured -- Fluorescence decay data
    window -- Time between laser exciation pulses (in nanoseconds)
    channelwidth -- Time width of one TCSPC channel (in nanoseconds)
    tau -- Initial guess times
    taubounds -- limits for the lifetimes guess times - defaults to 0<tau<100
                 format: [[tau1_min, tau1_max], [tau2_min, tau2_max], ...]
    init -- Whether to use a initial guess routine or not

    Output:
    c -- Color Shift (time shift of the IRF w.r.data_times. the fluorescence curve)
    offset -- Offset
    amplitudes -- Amplitudes of the different decay components
    tau	-- Decay times of the different decay components
    dc -- Color shift error
    doffset -- Offset error
    dtau -- Decay times error
    irs -- IRF, shifted by the value of the colorshift
    separated_decays -- Fitted fluorecence component curves
    data_times -- time axis
    chisquared -- chi squared value
    """

    irf = irf.flatten()
    measured = measured.flatten()
    irflength = np.size(irf)
    offset = 0
    cshift = 0

    if tau is None:
        # tau = distfluofit(irf, measured, window, channelwidth)
        # print('Initial guess:', tau)
        tau = [10, 25]

    if taubounds is None:
        taubounds = np.concatenate((0.001 * np.ones((np.size(tau), 1)), 30 * np.ones((np.size(tau), 1))), axis=1)
    taubounds = taubounds / channelwidth
    taubounds = tuple(map(tuple, taubounds))  # convert to tuple as required by minimize()

    window = window / channelwidth
    # tau = tau / channelwidth
    data_times = np.arange(np.size(measured))
    window_times = np.arange(1, window)
    taulength = np.size(tau)

    # param = np.array([cshift, offset])
    # param = np.append(param, tau)
    # param = tau

    # Decay times and offset are assumed to be positive.
    # offs_lower = np.array([-10])
    # offs_upper = np.array([10])
    # cshift_lower = np.array([-1])
    # cshift_upper = np.array([1])

    # bounds = (((-1/channelwidth, 1/channelwidth), (0, None)) + taubounds)
    # print(bounds)
    # result = minimize(lsfit, param, args=(irf, measured, window), method=method)

    scale = np.max(measured)
    measured = measured[startpoint:]
    irf = irf[startpoint:]
    # irflength = np.size(irf)
    model_in = np.append(t, irf)
    model_in = np.append(model_in, startpoint)
    model_in = np.append(model_in, irflength)
    popt, pcov = curve_fit(fitfunc, model_in, measured, bounds=([1, 1, 0, 0, 0], [100, 100, scale+1000, 1, 1]),
                           p0=[tau[0], tau[1], scale, 0.7, 0.3])
    print(popt)
    param = popt
    tau = param[:2]
    scale = param[2:3]
    amplitudes = param[4:]

    dtau = None
    irs = None
    separated_decays = None

    convd = fitfunc(model_in, popt[0], popt[1], popt[2], popt[3], popt[4])
    residuals = convd - measured
    chisquared = sum((convd[measured>0] - measured[measured>0]) ** 2 / np.abs(measured[measured>0]), 0.001) /np.size(measured[measured>0])

    if ploton:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_yscale('log')
        ax1.set_ylim([1, 50000])
        ax1.plot(measured)
        ax1.plot(convd)
        ax1.plot(irf)
        ax1.text(1500, 20000, 'Tau = %5.3f,     %5.3f' %(popt[0], popt[1]))
        ax1.text(1500, 8000, 'Amp = %5.3f,     %5.3f' %(popt[3], popt[4]))

        ax2.plot(residuals, '.')
        ax2.text(2500, 200, r'$\chi ^2 = $ %4.3f' %chisquared)
        plt.show()

    # tau = param
    # paramvariance = np.diag(result.hess_inv.matmat(np.identity(4)))
    # paramvariance = np.diag(result.hess_inv)
    # print(paramvariance)

    # cshift = param[0]
    # # dc = dparam(1)
    # tau = param
    # # dtau = dparam[2:np.shape(param)]

    # Calculate values from parameters
    # calculated, irs = create_exp(window_times, tau, window, irf, irflength, data_times)
    #
    # calculated = calculated.T/np.ones((irflength, 1))*np.sum(calculated)  # Normalize
    # amplitudes, residuals, rank, s = np.linalg.lstsq(calculated, measured, rcond=None)
    #
    # # Put individual decay curves into rows
    # separated_decays = calculated * np.matmul(np.ones(np.size(calculated, 1)), amplitudes.T)
    # total_decay = np.matmul(calculated, amplitudes).T
    #
    # #     dc = channelwidth*dc
    # chisquared = sum((measured - total_decay) ** 2. / abs(total_decay)) / (irflength - taulength)
    # dtau = 0  # channelwidth * np.sqrt(chisquared * paramvariance)
    # data_times = channelwidth * data_times
    # tau = channelwidth * tau.T
    # print(tau)
    # cshift = channelwidth * cshift
    # offset = separated_decays[0]
    return cshift, offset, amplitudes, tau, 0, dtau, irs, separated_decays, data_times, chisquared
