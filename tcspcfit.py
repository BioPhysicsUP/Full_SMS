""" Module for fitting TCSPC irf_data.

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


def convol(irf, decay):
    """Performs a convolution of irf with decay.

    Periodicity (period = size(decay)) is assumed.

    Arguments:
    irf -- row vector or 1D
    decay -- 1D or array with irf_data in rows

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


def colorshift(irf, shift, irflength=None, t=None):
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
    return irs


def distfluofit(irf, measured, period, channelwidth, cshift_bounds=[-3, 3], choose=False, ntau=100):
    """Quickly fit a multiexponential decay to use as 'initial guess'

        The function aims to identify the number of lifetimes in the
        measured irf_data, as well as the values of the lifetimes. The result
        can be used as an initial guess for fluofit.

    Arguments:
    irf -- Instrumental Response Function measured -- Fluorescence decay irf_data
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
    tp = channelwidth*np.arange(1, period/channelwidth)  # Time index for whole root
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


def fluofit(irf, measured, t, window, channelwidth, tau=None, taubounds=None, startpoint=0, endpoint=9000, init=0, ploton=False, method='Nelder-Mead'):
    """Fit of a multi-exponential decay curve.

    Arguments:
    irf -- Instrumental Response Function measured -- Fluorescence decay irf_data
    root -- Time between laser exciation pulses (in nanoseconds)
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
        # tau = distfluofit(irf, measured, root, channelwidth)
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
    # result = minimize(lsfit, param, args=(irf, measured, root), method=method)

    scale = np.max(measured)
    measured = measured[startpoint:endpoint]
    # irf = irf[startpoint:endpoint]
    # t = t[startpoint:]
    irflength = np.size(irf)
    model_in = np.append(t, irf)
    model_in = np.append(model_in, np.sum(measured))
    model_in = np.append(model_in, endpoint)
    model_in = np.append(model_in, startpoint)
    model_in = np.append(model_in, irflength)
    if np.size(tau) == 1:
        param, pcov = curve_fit(one_exp, model_in, measured, bounds=([1, 0, 0, -100], [100, scale + 1000, 1, 1000]),
                               p0=[tau[0], scale, 1, 0.1])
        tau = param[0]
        print("Tau:", tau)
        dtau = np.sqrt([pcov[0, 0]])
        print("dTau:", dtau)
        scale = param[1]
        print("Scale:", scale)
        amplitudes = param[2]
        shift = param[3]
        print('shift:', shift)

        irs = None
        separated_decays = None

        convd = one_exp(model_in, param[0], param[1], param[2], param[3])
        residuals = convd - measured
        convpos = convd[measured>0]
        measpos = measured[measured>0]
        chisquared = np.sum((convpos - measpos ** 2 / np.abs(measpos))) / np.size(measpos)

        if ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            ax1.set_ylim([1, 100])
            ax1.plot(measured.flatten())
            ax1.plot(convd.flatten())
            ax1.plot(irf[startpoint:endpoint])
            ax1.text(1500, 20000, 'Tau = %5.3f' % param[0])
            ax1.text(1500, 8000, 'Tau err = %5.3f' % dtau[0])
            ax1.text(1500, 3000, 'Amp = %5.3f' % param[2])

            ax2.plot(residuals, '.')
            ax2.text(2500, 200, r'$\chi ^2 = $ %4.3f' % chisquared)
            plt.show()

    elif np.size(tau) == 2:
        popt, pcov = curve_fit(two_exp, model_in, measured, bounds=([0.1, 0.1, 0, 0, -70, 0, 0], [10, 10, 100, 100, 14, 1e-30, 1e-30]),
                               p0=[tau[0], tau[1], 0.001, 0.001, -15, 0, 0])
        param = popt
        # print(param)
        # print(pcov)
        tau = param[:2]
        print(tau)
        dtau = np.sqrt([pcov[0, 0], pcov[1, 1]])
        print(dtau)
        # scale = param[2:3]
        amplitudes = param[2:4]
        sumamp = np.sum(amplitudes)
        print('ampl: ', amplitudes[0], amplitudes[1])
        print('rel ampl: ', amplitudes[0]/sumamp, amplitudes[1]/sumamp)
        shift = param[4]
        print('shift:', shift*channelwidth)
        bg = param[5]
        print('bg: ', bg*11928)
        irfbg = param[6]
        print('irf bg: ', irfbg*4047)

        irs = None
        separated_decays = None

        convd = two_exp(model_in, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
        residuals = convd - measured
        print(np.size(measured[measured>0]))
        print(np.size(measured))
        chisquared = np.sum((convd[measured>0] - measured[measured>0]) ** 2 / np.abs(measured[measured>0])) /np.size(measured[measured>0])

        if ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=300)
            ax1.set_yscale('log')
            ax1.set_ylim([1, 10000])
            ax1.plot(measured.flatten())
            ax1.plot(convd.flatten())
            ax1.plot(irf[startpoint:endpoint])
            ax1.text(500, 500, 'Tau = %5.3f,     %5.3f' %(popt[0], popt[1]))
            ax1.text(500, 300, 'Tau err = %5.3f,     %5.3f' %(dtau[0], dtau[1]))
            ax1.text(500, 150, 'Amp = %5.3f,     %5.3f' %(popt[3], popt[4]))

            ax2.plot(residuals, '.')
            ax2.text(500, 40, r'$\chi ^2 = $ %4.3f' %chisquared)
            print('chi = ', chisquared)
            plt.show()

    elif np.size(tau) == 3:
        popt, pcov = curve_fit(three_exp, model_in, measured, bounds=([0.01, 0.01, 0.01, 0, 0, 0, 0, 0], [100, 100, 100, scale + 1000, 1, 1, 1, 1]),
                               p0=[tau[0], tau[1], tau[2], scale, 0.333, 0.333, 0.334, 0.1])
        param = popt
        # print(param)
        # print(pcov)
        tau = param[:3]
        dtau = np.sqrt([pcov[0, 0], pcov[1, 1], pcov[2, 2]])
        print(dtau)
        scale = param[3]
        amplitudes = param[4:7]
        shift = param[7]
        print('shift:', shift)

        irs = None
        separated_decays = None

        convd = three_exp(model_in, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
        residuals = convd - measured
        chisquared = sum((convd[measured>0] - measured[measured>0]) ** 2 / np.abs(measured[measured>0]), 0.001) /np.size(measured[measured>0])

        if ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            ax1.set_ylim([1, 50000])
            ax1.plot(measured.flatten())
            ax1.plot(convd.flatten())
            ax1.plot(irf[startpoint:endpoint])
            ax1.text(1500, 20000, 'Tau = %5.3f,     %5.3f,     %5.3f' % (popt[0], popt[1], popt[2]))
            ax1.text(1500, 8000, 'Tau err = %5.3f,     %5.3f,     %5.3f' % (dtau[0], dtau[1], dtau[2]))
            ax1.text(1500, 3000, 'Amp = %5.3f,     %5.3f,     %5.3f' % (popt[4], popt[5], popt[6]))

            ax2.plot(residuals, '.')
            ax2.text(2500, 200, r'$\chi ^2 = $ %4.3f' %chisquared)
            print('chi = ', chisquared)
            plt.show()

    elif np.size(tau) == 4:
        print(np.size(measured))
        popt, pcov = curve_fit(four_exp, model_in, measured, bounds=([0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0], [100, 100, 100, 100, scale + 1000, 1, 1, 1, 1, 1]),
                               p0=[tau[0], tau[1], tau[2], tau[3], scale, 0.25, 0.25, 0.25, 0.25, 0.1])
        param = popt
        # print(param)
        # print(pcov)
        tau = param[:4]
        dtau = np.sqrt([pcov[0, 0], pcov[1, 1], pcov[2, 2], pcov[3, 3]])
        print(dtau)
        scale = param[4]
        amplitudes = param[5:9]
        shift = param[9]
        print('shift:', shift)

        irs = None
        separated_decays = None

        convd = four_exp(model_in, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
        residuals = convd - measured
        chisquared = sum((convd[measured>0] - measured[measured>0]) ** 2 / np.abs(measured[measured>0]), 0.001) /np.size(measured[measured>0])

        if ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            ax1.set_ylim([1, 50000])
            ax1.plot(measured.flatten())
            ax1.plot(convd.flatten())
            ax1.plot(irf[startpoint:endpoint])
            ax1.text(1500, 20000, 'Tau = %5.3f,     %5.3f,     %5.3f,     %5.3f' %(popt[0], popt[1], popt[2], popt[3]))
            ax1.text(1500, 8000, 'Tau err = %5.3f,     %5.3f,     %5.3f,     %5.3f' %(dtau[0], dtau[1], dtau[2], dtau[3]))
            ax1.text(1500, 3000, 'Amp = %5.3f,     %5.3f,     %5.3f,     %5.3f' %(popt[5], popt[6], popt[7], popt[8]))

            ax2.plot(residuals, '.')
            ax2.text(2500, 200, r'$\chi ^2 = $ %4.3f' %chisquared)
            plt.show()

    elif np.size(tau) == 5:
        popt, pcov = curve_fit(five_exp, model_in, measured, bounds=([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [100, 100, 100, 100, 100, scale + 1000, 1, 1, 1, 1, 1, 1]),
                               p0=[tau[0], tau[1], tau[2], tau[3], tau[4], scale, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
        param = popt
        # print(param)
        # print(pcov)
        tau = param[:2]
        dtau = np.sqrt([pcov[0, 0], pcov[1, 1]])
        print(dtau)
        scale = param[2:3]
        amplitudes = param[4:6]
        shift = param[5]
        print('shift:', shift)

        irs = None
        separated_decays = None

        convd = two_exp(model_in, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
        residuals = convd - measured
        chisquared = sum((convd[measured>0] - measured[measured>0]) ** 2 / np.abs(measured[measured>0]), 0.001) /np.size(measured[measured>0])

        if ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            ax1.set_ylim([1, 50000])
            ax1.plot(measured.flatten())
            ax1.plot(convd.flatten())
            ax1.plot(irf)
            ax1.text(1500, 20000, 'Tau = %5.3f,     %5.3f' %(popt[0], popt[1]))
            ax1.text(1500, 8000, 'Tau err = %5.3f,     %5.3f' %(dtau[0], dtau[1]))
            ax1.text(1500, 3000, 'Amp = %5.3f,     %5.3f' %(popt[3], popt[4]))

            ax2.plot(residuals, '.')
            ax2.text(2500, 200, r'$\chi ^2 = $ %4.3f' %chisquared)
            plt.show()

    return cshift, offset, amplitudes, tau, 0, dtau, irs, separated_decays, data_times, chisquared


class FluoFit:
    """Base class for fit of a multi-exponential decay curve.

    Arguments:
    irf -- Instrumental Response Function measured -- Fluorescence decay irf_data
    measured -- The measured decay irf_data
    channelwidth -- Time width of one TCSPC channel (in nanoseconds)
    tau -- Initial guess times
    taubounds -- Limits for the lifetimes guess times - defaults to 0<tau<100
                 format: [[tau1_min, tau1_max], [tau2_min, tau2_max], ...]
    startpoint -- Start of fitting range
    endpoint -- end of fitting range
    init -- Whether to use a initial guess routine or not
    ploton -- Whether to automatically plot the irf_data

    This class is only used for subclassing.
    """

    def __init__(self, irf, measured, t, channelwidth, tau=None, taubounds=None, startpoint=0, endpoint=9000,
                 init=0, ploton=False):

        self.channelwidth = channelwidth
        self.tau = tau
        self.taubounds = taubounds
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.init = init
        self.ploton = ploton
        self.t = t

        # Estimate background for IRF using average value up to start of the rise
        maxind = np.argmax(irf)
        for i in range(maxind):
            reverse = maxind - i
            if irf[reverse] == np.int(np.mean(irf[:20])):
                bglim = reverse
                break

        self.irfbg = np.mean(irf[:bglim])
        self.irf = irf - self.irfbg
        # self.irf = irf

        # Estimate background for decay in the same way
        maxind = np.argmax(measured)
        for i in range(maxind):
            reverse = maxind - i
            if irf[reverse] == np.int(np.mean(measured[:20])):
                bglim = reverse
                break

        self.bg = np.mean(measured[:bglim])
        measured = measured - self.bg

        self.measured = measured[startpoint:endpoint]

    def results(self, tau, dtau, shift, amp=1):

        print("Tau:", tau)
        print("dTau:", dtau)
        print('shift:', shift)

        # print(self.measured)

        residuals = self.convd - self.measured
        convpos = self.convd[self.measured > 0]
        measpos = self.measured[self.measured > 0]
        # print(measpos, convpos)
        chisquared = np.sum((convpos - measpos) ** 2 / measpos) / (np.size(measpos) - 5 - 1)
        print(chisquared)

        if self.ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            ax1.set_ylim([1, np.max(self.measured)])
            ax1.plot(self.measured.flatten())
            ax1.plot(self.convd.flatten())
            # ax1.plot(self.irf[self.startpoint:self.endpoint])
            ax1.text(1500, 30, 'Tau = %s' % tau)
            ax1.text(1500, 20, 'Tau err = %s' % dtau)
            ax1.text(1500, 10, 'Amp = %s' % amp)

            ax2.plot(residuals, '.')
            ax2.text(2000, 10, r'$\chi ^2 = $ %s' % chisquared)
            plt.show()

    def makeconvd(self, shift, model):

        irf = self.irf
        irf = irf * (np.sum(self.measured) / np.sum(irf))
        irf = colorshift(irf, shift, np.size(irf), self.t)
        convd = convolve(irf, model)
        convd = convd / np.sum(convd) * np.sum(self.measured)
        convd = convd[self.startpoint:self.endpoint]

        return convd


class OneExp(FluoFit):

    def __init__(self, irf, measured, t, channelwidth, tau=None, taubounds=None, startpoint=0, endpoint=9000,
                 init=0, ploton=False):

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, taubounds, startpoint, endpoint, init, ploton)

        param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=([0.01, -100], [100, 1000]),
                                p0=[tau[0], 0.1])

        tau = param[0]
        shift = param[1]
        dtau = pcov[0, 0]

        self.convd = self.fitfunc(self.t, tau, shift)
        self.results(tau, dtau, shift)

    def fitfunc(self, t, tau1, shift):

        model = np.exp(-t/tau1)
        return self.makeconvd(shift, model)


class TwoExp(FluoFit):

    def __init__(self, irf, measured, t, channelwidth, tau=None, taubounds=None, startpoint=0, endpoint=9000,
                 init=0, ploton=False):

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, taubounds, startpoint, endpoint, init, ploton)

        param, pcov = curve_fit(self.fitfunc, self.t, self.measured,
                                bounds=([0.01, 0.01, 0, 0, -60], [10, 10, 100, 100, 100]),
                                p0=[tau[0], tau[1], 50, 10, 0.1], verbose=2, ftol=5e-16, xtol=2e-16)

        tau = param[0:2]
        amp = param[2:4]
        print('Amp:', amp)
        shift = param[4]
        dtau = np.diag(pcov[0:2])

        # print(self.t, tau[0], tau[1], amp[0], amp[1], shift)

        self.convd = self.fitfunc(self.t, tau[0], tau[1], amp[0], amp[1], shift)
        self.results(tau, dtau, shift, amp)

    def fitfunc(self, t, tau1, tau2, a1, a2, shift):

        model = a1 * np.exp(-t / tau1) + (100 - a1) * np.exp(-t / tau2)
        return self.makeconvd(shift, model)


class ThreeExp(FluoFit):

    def __init__(self, irf, measured, t, channelwidth, tau=None, taubounds=None, startpoint=0, endpoint=9000,
                 init=0, ploton=False):

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, taubounds, startpoint, endpoint, init, ploton)

        param, pcov = curve_fit(self.fitfunc, self.t, self.measured,
                                bounds=([0.01, 0.01, 0.01, 0, 0, 0, -60], [10, 10, 10, 100, 100, 100, 100]),
                                p0=[tau[0], tau[1], tau[2], 50, 40, 50, 0.1], verbose=2, max_nfev=20000)#, ftol=5e-16, xtol=2e-16)

        tau = param[0:3]
        amp = param[3:6]
        print('Amp:', amp)
        shift = param[6]
        dtau = np.diag(pcov[0:3])

        self.convd = self.fitfunc(self.t, tau[0], tau[1], tau[2], amp[0], amp[1], amp[2], shift)
        self.results(tau, dtau, shift, amp)

    def fitfunc(self, t, tau1, tau2, tau3, a1, a2, a3, shift):

        model = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + a3 * np.exp(-t / tau3)
        return self.makeconvd(shift, model)

























