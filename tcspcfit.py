""" Module for fitting TCSPC irf_data.

The function distfluofit is based on MATLAB code by Jörg Enderlein:
https://www.uni-goettingen.de/en/513325.html

Bertus van Heerden
University of Pretoria
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


# TODO: simplify this as it is only used by distfluofit()
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
    if irflength is not None:
        raise Warning("Don't input irflength and t")
    irf = irf.flatten()
    irflength = np.size(irf)
    t = np.arange(irflength)
    new_index_left = np.fmod(np.fmod(t - np.floor(shift) - 1, irflength) + irflength, irflength).astype(int)
    new_index_right = np.fmod(np.fmod(t - np.ceil(shift) - 1, irflength) + irflength, irflength).astype(int)
    integer_left_shift = irf[new_index_left]
    integer_right_shift = irf[new_index_right]
    irs = (1 - shift + np.floor(shift)) * integer_left_shift + (shift - np.floor(shift)) * integer_right_shift
    return irs


# TODO: Add colourshift estimation to this function (important for actually getting correct lifetimes)
def distfluofit(irf, measured, period, channelwidth, cshift_bounds=[-3, 3], choose=False, ntau=50):
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
    tp = channelwidth*np.arange(1, period/channelwidth)  # Time index for whole window
    t = np.arange(irflength)  # Time index for IRF
    nrange = np.arange(ntau)
    # Distribution of inverse decay times:
    tau = (1/channelwidth) / np.exp(nrange / ntau * np.log(period / channelwidth))
    decays = convol(irf, np.exp(np.outer(tau, -tp)))
    decays = decays / np.sum(decays)
    amplitudes, residuals = nnls(decays.T, measured.flatten())
    tau = 1/tau
    print(period)
    plt.plot(amplitudes)
    plt.show()
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


class FluoFit:
    """Base class for fit of a multi-exponential decay curve.

    This class is only used for subclassing.

    Parameters
    ----------
    irf : ndarray
        Instrumental Response Function
    measured : ndarray
        The measured decay data
    channelwidth : float
        Time width of one TCSPC channel (in nanoseconds)
    tau : array_like, optional
        Initial guess times (in ns). This is either in the format
        [tau1, tau2, ...] or [[tau1, min1, max1, fix1], [tau2, ...], ...].
        When the "fix" value is False, the min and max values are ignored.
    amp : array_like, optional
        Initial guess amplitude. Format [amp1, amp2, ...] or [[amp1, fix1],
        [amp2, fix2], ...]
    shift : array_like, optional
        Initial guess IRF shift. Either a float, or [shift, min, max, fix].
    startpoint : int
        Start of fitting range - will default to channel of decay max or
        IRF max, whichever is least.
    endpoint : int
        End of fitting range - will default to the channel of the
        fluorescence decay that is either around 10 times higher than the
        background level or equivalent to 0.1% of the counts in the peak
        channel, whichever is greater.
    init : bool
        Whether to use an initial guess routine or not.
    ploton : bool
        Whether to automatically plot the irf_data

    """

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, startpoint=None, endpoint=None,
                 ploton=False):

        self.channelwidth = channelwidth
        # self.init = init
        self.ploton = ploton
        self.t = t

        self.tau = []
        self.taumin = []
        self.taumax = []
        try:
            for tauval in tau:
                try:
                    self.tau.append(tauval[0])
                    if tauval[-1]:
                        self.taumin.append(tauval[0] - 0.0001)
                        self.taumax.append(tauval[0] + 0.0001)
                    else:
                        self.taumin.append(tauval[1])
                        self.taumax.append(tauval[2])
                except TypeError:  # If tauval is not a list
                    self.tau.append(tauval)
                    self.taumin.append(0.01)
                    self.taumax.append(100)
        except TypeError:  # If tau is not a list
            self.tau = np.array([tau])
            self.taumin = np.array([0.01])
            self.taumax = np.array([100])

        if amp is not None:
            try:
                amplen = len(amp)
            except TypeError:
                amplen = 1
            if amplen == len(self.tau):
                self.amp = amp[:-1]
                if amp[-1]:
                    self.ampmin = []
                    self.ampmax = []
                    for amp in self.amp:
                        self.ampmin.append(amp - 0.0001)
                        self.ampmax.append(amp + 0.0001)
                else:
                    self.ampmin = []
                    self.ampmax = []
                    for tau in self.tau[:-1]:
                        self.ampmin.append(0)
                        self.ampmax.append(100)
            else:
                self.amp = []
                self.ampmin = []
                self.ampmax = []
                for tau in self.tau[:-1]:
                    self.amp.append(50)
                    self.ampmin.append(0)
                    self.ampmax.append(100)
        else:
            self.amp = []
            self.ampmin = []
            self.ampmax = []
            for tau in self.tau[:-1]:
                self.amp.append(50)
                self.ampmin.append(0)
                self.ampmax.append(100)

        if shift is None:
            shift = 0
        try:
            self.shift = shift[0]
            if shift[-1]:
                self.shiftmin = shift[0] - 0.0001
                self.shiftmax = shift[0] + 0.0001
            else:
                self.shiftmin = shift[1]
                self.shiftmax = shift[2]
        except TypeError:  # If shiftval is not a list
            self.shift = shift
            self.shiftmin = -100
            self.shiftmax = 300

        # Estimate background for IRF using average value up to start of the rise
        maxind = np.argmax(irf)
        for i in range(maxind):
            reverse = maxind - i
            if np.int(irf[reverse]) == np.int(np.mean(irf[:20])):
                bglim = reverse
                break

        self.irfbg = np.mean(irf[:bglim])
        self.irf = irf - self.irfbg
        self.irf = irf

        # Estimate background for decay in the same way
        maxind = np.argmax(measured)
        for i in range(maxind):
            reverse = maxind - i
            if irf[reverse] == np.int(np.mean(measured[:20])):
                bglim = reverse
                break

        self.bg = np.mean(measured[:bglim])
        # measured = measured - self.bg

        if startpoint is None:
            self.startpoint = np.argmax(self.irf)
        else:
            self.startpoint = startpoint

        if endpoint is None:
            great_than_bg, = np.where(measured > 10 * self.bg)
            hundredth_of_peak, = np.where(measured > 0.01 * measured.max())
            max1 = great_than_bg.max()
            max2 = hundredth_of_peak.max()
            self.endpoint = max(max1, max2)
        else:
            self.endpoint = endpoint

        self.measured = measured[self.startpoint:self.endpoint]

    def results(self, tau, dtau, shift, amp=1):

        # print("Tau:", tau)
        self.tau = tau
        self.dtau = dtau
        self.amp = amp
        self.shift = shift*self.channelwidth
        # print("dTau:", dtau)
        # print('shift:', shift)

        # print(self.measured)

        residuals = self.convd - self.measured
        residuals = residuals / np.sqrt(np.abs(self.measured))
        # pos_ind = self.measured > 0
        # measpos = self.measured[pos_ind]
        # residuals = residuals[measpos]
        # print(measpos, convpos)
        residualsnotinf = residuals != np.inf
        residuals = residuals[residualsnotinf]  # For some reason this is the only way i could find that works
        chisquared = np.sum((residuals ** 2 )) / (np.size(self.measured) - 4 - 1)
        self.chisq = chisquared
        self.t = self.t[self.startpoint:self.endpoint]
        self.residuals = residuals
        # print(chisquared)

        if self.ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            ax1.set_ylim([1, self.measured.max() * 2])
            ax1.plot(self.t, self.measured.flatten(), color='gray', linewidth=1)
            ax1.plot(self.t, self.convd.flatten(), color='C3', linewidth=1)
            # ax1.plot(self.irf[self.startpoint:self.endpoint])
            # ax1.plot(colorshift(self.irf, shift)[self.startpoint:self.endpoint])
            textx = (self.endpoint - self.startpoint) * self.channelwidth * 0.8
            texty = self.measured.max()
            try:
                ax1.text(textx, texty, 'Tau = ' + ' '.join('{:#.3g}'.format(F) for F in tau))
                ax1.text(textx, texty / 2, 'Amp = ' + ' '.join('{:#.3g}\% '.format(F) for F in amp))
            except TypeError:  # only one component
                ax1.text(textx, texty, 'Tau = {:#.3g}'.format(tau))
                # ax1.text(textx, texty / 2, 'Amp = {:#.3g}\% '.format(amp))
                # ax1.text(textx, texty / 2, 'Tau err = %s' % dtau)
            ax2.plot(self.t[residualsnotinf], residuals, '.', markersize=2)
            print(residuals.max())
            ax2.text(textx, residuals.max() / 1.1, r'$\chi ^2 = $ {:3.4f}'.format(chisquared))
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Weighted residual')
            ax1.set_ylabel('Number of photons in channel')
            plt.show()

    def makeconvd(self, shift, model):

        irf = self.irf
        # irf = irf * (np.sum(self.measured) / np.sum(irf))
        irf = colorshift(irf, shift)
        convd = convolve(irf, model)
        # convd = convd[:np.size(irf)]
        convd = convd[self.startpoint:self.endpoint]
        self.scalefactor = np.sum(self.measured) / np.sum(convd)
        convd = convd * self.scalefactor
        # convd = convd[self.startpoint:self.endpoint]

        return convd


class OneExp(FluoFit):
    """"Single exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, startpoint=None, endpoint=None,
                 ploton=False):

        if tau is None:
            tau = 5
        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, startpoint, endpoint, ploton)

        self.tau = self.tau[0]
        self.taumin = self.taumin[0]
        self.taumax = self.taumax[0]
        paramin = [self.taumin, self.shiftmin]
        paramax = [self.taumax, self.shiftmax]
        paraminit = [self.tau, self.shift]
        param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax),
                                p0=paraminit)

        tau = param[0]
        shift = param[1]
        dtau = np.sqrt(pcov[0, 0])

        self.convd = self.fitfunc(self.t, tau, shift)
        self.results(tau, dtau, shift)

    def fitfunc(self, t, tau1, shift):

        model = np.exp(-t/tau1)
        return self.makeconvd(shift, model)


class TwoExp(FluoFit):
    """"Double exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, startpoint=None, endpoint=None,
                 ploton=False):

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, startpoint, endpoint, ploton)

        paramin = self.taumin + self.ampmin + [self.shiftmin]
        paramax = self.taumax + self.ampmax + [self.shiftmax]
        paraminit = self.tau + self.amp + [self.shift]
        param, pcov = curve_fit(self.fitfunc, self.t, self.measured,# sigma=np.sqrt(np.abs(self.measured)),
                                bounds=(paramin, paramax), p0=paraminit, ftol=1e-16, gtol=1e-16, xtol=1e-16)

        tau = param[0:2]
        amp = np.append(param[2], 100-param[2])
        # print('Amp:', amp)
        shift = param[3]
        dtau = np.sqrt(np.diag(pcov[0:2]))

        # print(self.t, tau[0], tau[1], amp[0], amp[1], shift)

        self.convd = self.fitfunc(self.t, tau[0], tau[1], amp[0], shift)
        self.results(tau, dtau, shift, amp)

    def fitfunc(self, t, tau1, tau2, a1, shift):

        model = a1 * np.exp(-t / tau1) + (100 - a1) * np.exp(-t / tau2)
        return self.makeconvd(shift, model)


# TODO: make this class also use normalised amplitudes
class ThreeExp(FluoFit):
    """"Triple exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, startpoint=None, endpoint=None,
                 ploton=False):

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, startpoint, endpoint, ploton)

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

























