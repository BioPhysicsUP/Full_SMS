""" Module for fitting TCSPC data.

The function distfluofit is based on MATLAB code by Jörg Enderlein:
https://www.uni-goettingen.de/en/513325.html

Bertus van Heerden
University of Pretoria
"""

import numpy as np
import pyqtgraph as pg
import scipy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QLineEdit, QCheckBox, QDialog, QComboBox, QMessageBox
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.optimize import curve_fit, nnls
from scipy.signal import convolve
import file_manager as fm
from PyQt5 import uic

from my_logger import setup_logger

logger = setup_logger(__name__)

fitting_dialog_file = fm.path(name="fitting_dialog.ui", file_type=fm.Type.UI)
UI_Fitting_Dialog, _ = uic.loadUiType(fitting_dialog_file)


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


def colorshift(irf, shift, irflength=None):
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
        peak_tau = np.append(peak_tau, np.dot(amplitudes[t1[j]-1:t2[j]],
                                              tau[t1[j]-1:t2[j]]) / np.sum(amplitudes[t1[j]-1:t2[j]]))

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
    bg : float
        Background value for decay. Will be estimated if not given.
    irfbg : float
        Background value for IRF. Will be estimated if not given.
    startpoint : int
        Start of fitting range - will default to channel of decay max or
        IRF max, whichever is least.
    endpoint : int
        End of fitting range - will default to the channel of the
        fluorescence decay that is either around 10 times higher than the
        background level or equivalent to 0.1% of the counts in the peak
        channel, whichever is greater.
    init : bool  # TODO: add this functionality
        Whether to use an initial guess routine or not.
    ploton : bool
        Whether to automatically plot the irf_data

    """

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 startpoint=None, endpoint=None, ploton=False):

        self.channelwidth = channelwidth
        # self.init = init
        self.ploton = ploton
        self.t = t

        self.tau = []
        self.taumin = []
        self.taumax = []
        self.amp = []
        self.ampmin = []
        self.ampmax = []
        self.shift = None
        self.shiftmin = None
        self.shiftmax = None
        self.setup_params(amp, shift, tau)

        self.bg = None
        self.irfbg = None
        self.calculate_bg(bg, irf, irfbg, measured)

        self.irf = irf - self.irfbg
        # self.irf = self.irf / np.sum(self.irf)  # Normalize IRF
        self.irf = self.irf / self.irf.max()  # Normalize IRF

        measured = measured - self.bg

        self.startpoint = None
        self.endpoint = None
        self.calculate_boundaries(endpoint, measured, startpoint)

        self.meas_max = measured.max()
        measured = measured / self.meas_max  # Normalize measured
        # measured = measured / measured.max()  # Normalize measured
        self.measured = measured[self.startpoint:self.endpoint]
        self.dtau = None
        self.chisq = None
        self.residuals = None
        self.convd = None

    def calculate_boundaries(self, endpoint, measured, startpoint):
        """Set the start and endpoints

        Sets the values to the given ones or automatically find good ones.
        The start value is chosen as the maximum point of the IRF, and the end
        value is chosen as either the point where the decay is 1 % of
        maximum or 10 times the background value, whichever is highest.

        Parameters
        ----------
        endpoint : int
                End of fitting range
        measured : ndarray
            The measured decay data
        startpoint : int
                Start of fitting range

        """
        if startpoint is None:
            self.startpoint = np.argmax(measured)
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

    def calculate_bg(self, bg, irf, irfbg, measured):
        """Calculate decay and IRF background values

        If not given, the  background value is estimated from the first part
        of the curve, before the rise.

        Parameters:
        -----------

        bg : float
            Decay background value
        irf : ndarray
            Instrument response function
        irfbg : float
            IRF background value
        measured : ndarray
            Measured decay data
        """
        if irfbg is None:
            maxind = np.argmax(irf)
            for i in range(maxind):
                reverse = maxind - i
                if np.int(irf[reverse]) == np.int(np.mean(irf[:20])):
                    bglim = reverse
                    break

            self.irfbg = np.mean(irf[:bglim])
        else:
            self.irfbg = irfbg
        if bg is None:
            maxind = np.argmax(measured)
            for i in range(maxind):
                reverse = maxind - i
                if measured[reverse] == np.int(np.mean(measured[:20])):
                    bglim = reverse
                    break

            self.bg = np.mean(measured[:bglim])
        else:
            self.bg = bg

    def setup_params(self, amp, shift, tau):
        """Setup fitting parameters

        This method handles the input of initial parameters for fitting. The
        input system is flexible, allowing optional input of min and max
        values as well as choosing to fix a value.

        Parameters
        ----------

        amp : list or float
            Amplitude(s)
        shift : list or float
            IRF colour shift
        tau : list or float
            Lifetime(s)

        """
        try:
            for tauval in tau:
                try:
                    self.tau.append(tauval[0])
                    if tauval[-1]:  # If tau is fixed
                        self.taumin.append(tauval[0] - 0.0001)
                        self.taumax.append(tauval[0] + 0.0001)
                    else:
                        self.taumin.append(tauval[1])
                        self.taumax.append(tauval[2])
                except TypeError:  # If tauval is not a list - i.e. no bounds given
                    self.tau.append(tauval)
                    self.taumin.append(0.01)
                    self.taumax.append(100)
        except TypeError:  # If tau is not a list - i.e. only one lifetime
            self.tau = tau
            self.taumin = 0.01
            self.taumax = 100
        try:
            for ampval in amp:
                try:
                    self.amp.append(ampval[0])
                    if ampval[-1]:  # If amp is fixed
                        self.ampmin.append(ampval[0] - 0.0001)
                        self.ampmax.append(ampval[0] + 0.0001)
                    else:
                        self.ampmin.append(ampval[1])
                        self.ampmax.append(ampval[2])
                except TypeError:  # If ampval is not a list - i.e. no bounds given
                    self.amp.append(ampval)
                    self.ampmin.append(0)
                    self.ampmax.append(100)
        except TypeError:  # If amp is not a list - i.e. only one lifetime
            self.amp = amp
            self.ampmin = 0
            self.ampmax = 100
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
        except (TypeError, IndexError) as e:  # If shiftval is not a list
            self.shift = shift
            self.shiftmin = -2000
            self.shiftmax = 2000

    def results(self, tau, dtau, shift, amp=1):
        """Handle results after fitting

        After fitting, the results are processed. Chi-squared is calculated
        and optional plotting is done.

        Parameters:
        -----------
        tau : ndarray or float
            Fitted lifetime(s)
        dtau : ndarray or float
            Error in fitted lifetimes
        shift : float
            Fitted colourshift value
        amp : ndarray or float
            Fitted amplitude(s)

        """

        self.tau = tau
        self.dtau = dtau
        self.amp = amp
        self.shift = shift*self.channelwidth

        residuals = self.convd - self.measured
        residuals = residuals / np.sqrt(np.abs(self.measured))
        residualsnotinf = residuals != np.inf
        residuals = residuals[residualsnotinf]  # For some reason this is the only way i could find that works
        chisquared = np.sum((residuals ** 2)) / (np.size(self.measured) - 4 - 1)
        chisquared = chisquared * self.meas_max  # This is necessary because of normalisation
        self.chisq = chisquared
        self.t = self.t[self.startpoint:self.endpoint]
        self.residuals = residuals

        if self.ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            # ax1.set_ylim([self.measured.min() * 1000, self.measured.max() * 2])
            ax1.plot(self.t, self.measured.flatten(), color='gray', linewidth=1)
            ax1.plot(self.t, self.convd.flatten(), color='C3', linewidth=1)
            # ax1.plot(self.irf[self.startpoint:self.endpoint])
            # ax1.plot(colorshift(self.irf, shift)[self.startpoint:self.endpoint])
            textx = (self.endpoint - self.startpoint) * self.channelwidth * 1.4
            texty = self.measured.max()
            try:
                ax1.text(textx, texty, 'Tau = ' + ' '.join('{:#.3g}'.format(F) for F in tau))
                ax1.text(textx, texty / 2, 'Amp = ' + ' '.join('{:#.3g} '.format(F) for F in amp))
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
        """Makes a convolved decay using IRF and exponential model

        Parameters:
        -----------

        shift : float
            IRF colour shift
        model : ndarray
            Exponential model function

        """

        irf = self.irf
        irf = colorshift(irf, shift)
        convd = convolve(irf, model)
        convd = convd[self.startpoint:self.endpoint]
        return convd


class OneExp(FluoFit):
    """"Single exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 startpoint=None, endpoint=None, addopt=None, ploton=False):

        if tau is None:
            tau = 5
        if amp is None:
            amp = 1
        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, bg, irfbg, startpoint, endpoint, ploton)

        paramin = [self.taumin[0], self.ampmin, self.shiftmin]
        paramax = [self.taumax[0], self.ampmax, self.shiftmax]
        paraminit = [self.tau[0], self.amp, self.shift]
        if addopt is None:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit)
        else:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit, **addopt)

        tau = param[0]
        amp = param[1]
        shift = param[2]
        dtau = np.sqrt(pcov[0, 0])

        self.convd = self.fitfunc(self.t, tau, amp, shift)
        self.results(tau, dtau, shift)

    def fitfunc(self, t, tau1, a, shift):
        """Function passed to curve_fit, to be fitted to data"""

        model = a * np.exp(-t/tau1)
        return self.makeconvd(shift, model)


class TwoExp(FluoFit):
    """"Double exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 startpoint=None, endpoint=None, addopt=None, ploton=False):

        if tau is None:
            tau = [1, 5]
        if amp is None:
            amp = [1, 1]

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, bg, irfbg, startpoint, endpoint, ploton)

        paramin = self.taumin + self.ampmin + [self.shiftmin]
        paramax = self.taumax + self.ampmax + [self.shiftmax]
        paraminit = self.tau + self.amp + [self.shift]
        if addopt is None:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit)
        else:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit, **addopt)

        tau = param[0:2]
        amp = np.append(param[2], param[3])
        shift = param[4]
        dtau = np.sqrt(np.diag(pcov[0:2]))

        self.convd = self.fitfunc(self.t, tau[0], tau[1], amp[0], amp[1], shift)
        self.results(tau, dtau, shift, amp)

    def fitfunc(self, t, tau1, tau2, a1, a2, shift):
        """Function passed to curve_fit, to be fitted to data"""

        model = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
        return self.makeconvd(shift, model)


class ThreeExp(FluoFit):
    """"Triple exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 startpoint=None, endpoint=None, ploton=False):

        if tau is None:
            tau = [0.1, 1, 5]
        if amp is None:
            amp = [1, 1, 1]

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, bg, irfbg, startpoint, endpoint, ploton)

        paramin = self.taumin + self.ampmin + [self.shiftmin]
        paramax = self.taumax + self.ampmax + [self.shiftmax]
        paraminit = self.tau + self.amp + [self.shift]
        param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit)

        tau = param[0:3]
        amp = param[3:6]
        shift = param[6]
        dtau = np.diag(pcov[0:3])

        self.convd = self.fitfunc(self.t, tau[0], tau[1], tau[2], amp[0], amp[1], amp[2], shift)
        self.results(tau, dtau, shift, amp)

    def fitfunc(self, t, tau1, tau2, tau3, a1, a2, a3, shift):
        """Function passed to curve_fit, to be fitted to data"""

        model = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + a3 * np.exp(-t / tau3)
        return self.makeconvd(shift, model)


class FittingParameters:
    def __init__(self, parent):
        self.parent = parent
        self.fpd = self.parent.fitparamdialog
        self.irf = None
        self.tau = None
        self.amp = None
        self.shift = None
        self.shiftfix = None
        self.decaybg = None
        self.irfbg = None
        self.start = None
        self.end = None
        self.numexp = None
        self.addopt = None

    def getfromdialog(self):
        self.numexp = int(self.fpd.combNumExp.currentText())
        if self.numexp == 1:
            self.tau = [[self.get_from_gui(i) for i in
                         [self.fpd.line1Init, self.fpd.line1Min, self.fpd.line1Max,
                          self.fpd.check1Fix]]]
            self.amp = [[self.get_from_gui(i) for i in
                         [self.fpd.line1AmpInit, self.fpd.line1AmpMin, self.fpd.line1AmpMax,
                          self.fpd.check1AmpFix]]]

        elif self.numexp == 2:
            self.tau = [[self.get_from_gui(i) for i in
                         [self.fpd.line2Init1, self.fpd.line2Min1, self.fpd.line2Max1,
                          self.fpd.check2Fix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line2Init2, self.fpd.line2Min2, self.fpd.line2Max2,
                          self.fpd.check2Fix2]]]
            self.amp = [[self.get_from_gui(i) for i in
                         [self.fpd.line2AmpInit1, self.fpd.line2AmpMin1, self.fpd.line2AmpMax1,
                          self.fpd.check2AmpFix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line2AmpInit2, self.fpd.line2AmpMin2, self.fpd.line2AmpMax2,
                          self.fpd.check2AmpFix2]]]

        elif self.numexp == 3:
            self.tau = [[self.get_from_gui(i) for i in
                         [self.fpd.line3Init1, self.fpd.line3Min1, self.fpd.line3Max1,
                          self.fpd.check3Fix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3Init2, self.fpd.line3Min2, self.fpd.line3Max2,
                          self.fpd.check3Fix2]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3Init3, self.fpd.line3Min3, self.fpd.line3Max3,
                          self.fpd.check3Fix3]]]
            self.amp = [[self.get_from_gui(i) for i in
                         [self.fpd.line3AmpInit1, self.fpd.line3AmpMin1, self.fpd.line3AmpMax1,
                          self.fpd.check3AmpFix1]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3AmpInit2, self.fpd.line3AmpMin2, self.fpd.line3AmpMax2,
                          self.fpd.check3AmpFix2]],
                        [self.get_from_gui(i) for i in
                         [self.fpd.line3AmpInit3, self.fpd.line3AmpMin3, self.fpd.line3AmpMax3,
                          self.fpd.check3AmpFix3]]]

        self.shift = self.get_from_gui(self.fpd.lineShift)
        self.shiftfix = self.get_from_gui(self.fpd.checkFixIRF)
        self.decaybg = self.get_from_gui(self.fpd.lineDecayBG)
        self.irfbg = self.get_from_gui(self.fpd.lineIRFBG)
        self.start = self.get_from_gui(self.fpd.lineStartTime)
        self.end = self.get_from_gui(self.fpd.lineEndTime)
        # try:
        #     self.start = int(self.get_from_gui(self.fpd.lineStartTime))
        # except TypeError:
        #     self.start = self.get_from_gui(self.fpd.lineStartTime)
        # try:
        #     self.end = int(self.get_from_gui(self.fpd.lineEndTime))
        # except TypeError:
        #     self.end = self.get_from_gui(self.fpd.lineEndTime)

        if self.fpd.lineAddOpt.text() != '':
            self.addopt = self.fpd.lineAddOpt.text()
        else:
            self.addopt = None

    @staticmethod
    def get_from_gui(guiobj):
        if type(guiobj) == QLineEdit:
            if guiobj.text() == '':
                return None
            else:
                return float(guiobj.text())
        elif type(guiobj) == QCheckBox:
            return float(guiobj.isChecked())


class FittingDialog(QDialog, UI_Fitting_Dialog):
    """Class for dialog that is used to choose lifetime fit parameters."""

    def __init__(self, mainwindow, lifetime_controller):
        QDialog.__init__(self)
        UI_Fitting_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.lifetime_controller = lifetime_controller
        self.pgFitParam.setBackground(background=None)
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self.updateplot)
        for widget in self.findChildren(QCheckBox):
            widget.stateChanged.connect(self.updateplot)
        for widget in self.findChildren(QComboBox):
            widget.currentTextChanged.connect(self.updateplot)
        self.updateplot()

        # self.lineStartTime.setValidator(QIntValidator())
        # self.lineEndTime.setValidator(QIntValidator())

    def updateplot(self, *args):

        try:
            model = self.make_model()
        except Exception as err:
            logger.error('Error Occured: ' + str(err))
            return

        fp = self.lifetime_controller.fitparam
        try:
            irf = fp.irf
            irft = fp.irft
        except AttributeError:
            logger.error('No IRF!')
            return

        shift, decaybg, irfbg, start, end = self.getparams()

        channelwidth = self.mainwindow.currentparticle.channelwidth
        shift = shift / channelwidth
        start = int(start / channelwidth)
        end = int(end / channelwidth)
        # irf = tcspcfit.colorshift(irf, shift)
        irf = colorshift(irf, shift)
        convd = scipy.signal.convolve(irf, model)
        convd = convd[:np.size(irf)]
        convd = convd / convd.max()

        try:
            if self.mainwindow.current_level is None:
                histogram = self.mainwindow.currentparticle.histogram
            else:
                level = self.mainwindow.current_level
                histogram = self.mainwindow.currentparticle.levels[level].histogram
            decay = histogram.decay
            decay = decay / decay.max()
            t = histogram.t

            # decay, t = start_at_value(decay, t)
            end = min(end, np.size(t) - 1)  # Make sure endpoint is not bigger than size of t

            convd = convd[irft > 0]
            irft = irft[irft > 0]

        except AttributeError:
            logger.error('No Decay!')
        else:
            plot_item = self.pgFitParam.getPlotItem()

            plot_item.setLogMode(y=True)
            plot_pen = QPen()
            plot_pen.setWidthF(3)
            plot_pen.setJoinStyle(Qt.RoundJoin)
            plot_pen.setColor(QColor('blue'))
            plot_pen.setCosmetic(True)

            plot_item.clear()
            plot_item.plot(x=t, y=np.clip(decay, a_min=0.001, a_max=None), pen=plot_pen,
                           symbol=None)
            plot_pen.setWidthF(4)
            plot_pen.setColor(QColor('dark blue'))
            plot_item.plot(x=irft, y=np.clip(convd, a_min=0.001, a_max=None), pen=plot_pen,
                           symbol=None)
            # unit = 'ns with ' + str(currentparticle.channelwidth) + 'ns bins'
            plot_item.getAxis('bottom').setLabel('Decay time (ns)')
            # plot_item.getViewBox().setLimits(xMin=0, yMin=0.1, xMax=t[-1], yMax=1)
            # plot_item.getViewBox().setLimits(xMin=0, yMin=0, xMax=t[-1])
            # self.MW_fitparam.axes.clear()
            # self.MW_fitparam.axes.semilogy(t, decay, color='xkcd:dull blue')
            # self.MW_fitparam.axes.semilogy(irft, convd, color='xkcd:marine blue', linewidth=2)
            # self.MW_fitparam.axes.set_ylim(bottom=1e-2)

        try:
            plot_pen.setColor(QColor('gray'))
            plot_pen.setWidth(3)
            startline = pg.InfiniteLine(angle=90, pen=plot_pen, movable=False, pos=t[start])
            endline = pg.InfiniteLine(angle=90, pen=plot_pen, movable=False, pos=t[end])
            plot_item.addItem(startline)
            plot_item.addItem(endline)
            # self.MW_fitparam.axes.axvline(t[start])
            # self.MW_fitparam.axes.axvline(t[end])
        except IndexError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText('Value out of bounds!')
            msg.exec_()

    def getparams(self):
        fp = self.lifetime_controller.fitparam
        irf = fp.irf
        shift = fp.shift
        if shift is None:
            shift = 0
        decaybg = fp.decaybg
        if decaybg is None:
            decaybg = 0
        irfbg = fp.irfbg
        if irfbg is None:
            irfbg = 0
        start = fp.start
        if start is None:
            start = 0
        end = fp.end
        if end is None:
            end = np.size(irf)
        return shift, decaybg, irfbg, start, end

    def make_model(self):
        fp = self.lifetime_controller.fitparam
        t = self.mainwindow.currentparticle.histogram.t
        fp.getfromdialog()
        if fp.numexp == 1:
            tau = fp.tau[0][0]
            model = np.exp(-t / tau)
        elif fp.numexp == 2:
            tau1 = fp.tau[0][0]
            tau2 = fp.tau[1][0]
            amp1 = fp.amp[0][0]
            amp2 = fp.amp[1][0]
            # print(amp1, amp2, tau1, tau2)
            model = amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2)
        elif fp.numexp == 3:
            tau1 = fp.tau[0][0]
            tau2 = fp.tau[1][0]
            tau3 = fp.tau[2][0]
            amp1 = fp.amp[0][0]
            amp2 = fp.amp[1][0]
            amp3 = fp.amp[2][0]
            model = amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2) + amp3 * np.exp(-t / tau3)
        return model