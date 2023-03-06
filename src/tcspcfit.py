""" Module for fitting TCSPC data.

The function distfluofit is based on MATLAB code by Jörg Enderlein:
https://www.uni-goettingen.de/en/513325.html

Bertus van Heerden and Joshua Botha
University of Pretoria
"""

from __future__ import annotations

__docformat__ = 'NumPy'

import numpy as np
import pyqtgraph as pg
import scipy
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QPen, QColor, QDoubleValidator, QRegExpValidator
from PyQt5.QtWidgets import QLineEdit, QCheckBox, QDialog, QComboBox, QMessageBox
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.optimize import curve_fit, nnls
from scipy.signal import convolve
import file_manager as fm
from PyQt5 import uic
from typing import TYPE_CHECKING, Union
from settings_dialog import Settings

if TYPE_CHECKING:
    from controllers import LifetimeController

from my_logger import setup_logger

logger = setup_logger(__name__)

fitting_dialog_file = fm.path(name="fitting_dialog.ui", file_type=fm.Type.UI)
UI_Fitting_Dialog, _ = uic.loadUiType(fitting_dialog_file)


BACKGOURD_SECTION_LENGTH = 50


def moving_avg(vector: Union[list, np.ndarray], window_length: int,
               pad_same_size: bool = True) -> np.ndarray:
    vector_size = vector.size
    left_window = int(np.floor(window_length/2))
    right_window = int(np.ceil(window_length/2))
    offset = 0
    if pad_same_size:
        start_pad = np.zeros(left_window)
        end_pad = np.zeros(right_window)
        vector = np.concatenate([start_pad, vector, end_pad])
    else:
        offset = window_length

    new_vector = np.array([
        np.mean(vector[i - left_window: i + right_window])
        for i in range(left_window, left_window + vector_size - offset)
    ])
    return new_vector


def max_continuous_zeros(vector: np.ndarray) -> int:
    if type(vector) is list:
        vector = np.array(vector)
    is_zero = vector == 0
    continous_zeros = np.diff(np.where(np.concatenate(([is_zero[0]], is_zero[:-1] != is_zero[1:],
                                                       [True])))[0])[::2]
    return np.max(continous_zeros) if len(continous_zeros) > 0 else 0


def makerow(vector):
    """Reshape 1D vector into a 2D row"""
    return np.reshape(vector, (1, -1))


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
    new_index_left = np.fmod(np.fmod(t - np.floor(shift), irflength) + irflength, irflength).astype(int)
    new_index_right = np.fmod(np.fmod(t - np.ceil(shift), irflength) + irflength, irflength).astype(int)
    integer_left_shift = irf[new_index_left]
    integer_right_shift = irf[new_index_right]
    irs = (1 - shift + np.floor(shift)) * integer_left_shift + (shift - np.floor(shift)) * integer_right_shift
    return irs


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
    ploton : bool
        Whether to automatically plot the irf_data

    """

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 boundaries=None, ploton=False, fwhm=None, numexp=None):

        self.numexp = numexp
        self.channelwidth = channelwidth
        self.ploton = ploton
        self.t = t

        if fwhm is not None:
            self.simulate_irf = True
        else:
            self.simulate_irf = False

        self.tau = []
        self.taumin = []
        self.taumax = []
        self.amp = []
        self.ampmin = []
        self.ampmax = []
        self.shift = None
        self.shiftmin = None
        self.shiftmax = None
        self.fwhm = None
        self.fwhmmin = None
        self.fwhmmax = None
        self.stds = []
        self.setup_params(amp, shift, tau, fwhm)
        self.settings = Settings()
        self.load_settings()

        self.bg = None
        self.irfbg = None
        self.calculate_bg(bg, irf, irfbg, measured)

        if not self.simulate_irf:
            self.irf = irf - self.irfbg
            # self.irf = self.irf / np.sum(self.irf)  # Normalize IRF
            self.irf = self.irf / self.irf.max()  # Normalize IRF

        self.startpoint = None
        self.endpoint = None
        self.startpoint, self.endpoint = self.calculate_boundaries(measured, boundaries, self.bg,
                                                                   self.settings, channelwidth)

        self.meas_bef_bg = measured.copy()
        measured = measured - self.bg  # This will result in negative counts, and can't see where it's dealt with
        measured[measured <= 0] = 0

        self.measured_unbounded = measured
        measured = measured[self.startpoint:self.endpoint]
        self.measured_not_normalized = self.meas_bef_bg[self.startpoint:self.endpoint]
        self.meas_max = np.nansum(measured)
        meas_std = np.sqrt(np.abs(measured))
        self.bg_n = None
        self.meas_std = None
        if self.meas_max != 0:
            self.measured = measured / self.meas_max  # Normalize measured
            self.bg_n = self.bg / self.meas_max  # Normalized background
            self.meas_std = meas_std / self.meas_max
        self.dtau = None
        self.chisq = None
        self.residuals = None
        self.dw = None
        self.dw_bound = None
        self.convd = None
        self.is_fit = False

    def load_settings(self):
        settings_file_path = fm.path('settings.json', fm.Type.ProjectRoot)
        with open(settings_file_path, 'r') as settings_file:
            if not hasattr(self, 'settings'):
                self.settings = Settings()
            self.settings.load_settings_from_file(file_or_path=settings_file)

    @staticmethod
    def calculate_boundaries(measured, boundaries, bg, settings, channel_width):
        """Set the start and endpoints

        Sets the values to the given ones or automatically find good ones.
        The start value is chosen as earliest point that is at least 80%
        of the maximum point of the measured decay, and the end
        value is chosen as either the point where the decay is 1 % of
        maximum or 20 times the background value, whichever is highest.
        These values can be modified in the settings dialog.

        Parameters
        ----------
        measured : ndarray
            The measured decay data
        boundaries : list
                [startpoint, endpoint, autostart, autoend]
        bg : float
            Calculated decay background
        settings : settings_dialog.Settings() object
            Contains config settings
        channel_width : float
            Duration of a single channel
        """
        if boundaries is None:
            boundaries = [None, None, 'Manual', False]

        startpoint = boundaries[0]
        endpoint = boundaries[1]
        autostart = boundaries[2]
        autoend = boundaries[3]

        startmax = None
        if settings.lt_use_moving_avg:
            measured = moving_avg(vector=measured,
                                  window_length=min(settings.lt_moving_avg_window,
                                                    int(0.1 * measured.size)),
                                  pad_same_size=True)
        if autostart == 'Manual':
            if startpoint is None:
                startpoint = 0
        else:
            maxpoint = measured.max()
            close_percentage = settings.lt_start_percent / 100
            close_to_max, = np.where(measured > close_percentage * maxpoint)
            startmax = close_to_max[0]
            startmin = FluoFit.estimate_bg(measured, return_bglim=True)
            if autostart == '(Close to) max':
                startpoint = startmax
            elif autostart == 'Rise middle':
                startpoint = int(0.5 * (startmin + startmax))
            elif autostart == 'Rise start':
                startpoint = startmin
            elif autostart == 'Safe rise start':
                startpoint = startmin - min((startmax - startmin), 10)

        if autoend:
            if settings is not None:
                end_multiple = settings.lt_end_multiple
                end_percent = settings.lt_end_percent / 100
            else:
                end_multiple = 20
                end_percent = 0.01
            min_val = max(end_multiple * bg, end_percent * measured.max())
            if not np.isnan(min_val):
                great_than_bg, = np.where(measured[startpoint:] > min_val)
                great_than_bg += startpoint
                if great_than_bg.size != 0 and \
                        great_than_bg[-1] == measured.size - 1:
                    great_than_bg = great_than_bg[:-1]
                if not great_than_bg.size == 0:
                    endpoint = great_than_bg.max()
                else:
                    endpoint = startpoint + int(np.round(
                        settings.lt_minimum_decay_window/channel_width))
            if settings.lt_use_moving_avg:
                endpoint += int(np.round(settings.lt_moving_avg_window/2))
        elif endpoint is not None:
            endpoint = min(endpoint, measured.size - 1)
        else:
            endpoint = measured.size - 1

        if settings is not None:
            if channel_width*(endpoint - startpoint) < settings.lt_minimum_decay_window:
                start = startmax if startmax is not None else startpoint
                endpoint = start + int(np.round(
                    settings.lt_minimum_decay_window/channel_width))
                if autostart == 'Safe rise start':
                    startpoint -= int(np.round(
                        0.3*settings.lt_minimum_decay_window/channel_width))

        return startpoint, endpoint

    def calculate_bg(self, bg, irf, irf_bg, measured):
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
        if irf_bg is None:
            if self.simulate_irf:
                self.irfbg = 0
            else:  # TODO: replace with estimate_bg
                bglim = None
                maxind = np.argmax(irf)
                for i in range(maxind):
                    reverse = maxind - i
                    if np.int(irf[reverse]) == np.int(np.mean(irf[:20])):
                        bglim = reverse
                        break

                if bglim is not None:
                    self.irfbg = np.mean(irf[:bglim])
                else:
                    self.irfbg = 0
        else:
            self.irfbg = irf_bg
        if bg is None:
            bg_est = self.estimate_bg(measured, settings=self.settings)
            self.bg = bg_est
        else:
            self.bg = bg

    @staticmethod
    def estimate_bg(measured, return_bglim=False, settings=None):
        """Estimate decay background

        Optionally returns only the index of rise start.

        Parameters
        ----------

        measured : ndarray
            measured decay data
        return_bglim : bool
            whether to return index of rise start instead of bg
        settings : settings_dialog.Settings() object
            Contains config settings

        """
        maxind = np.argmax(measured)

        meas_real_start = np.nonzero(measured)[0][0]
        bg_section = measured[meas_real_start:meas_real_start + BACKGOURD_SECTION_LENGTH]

        # Attempt to remove low `island` of counts before real start
        if max_continuous_zeros(bg_section)/BACKGOURD_SECTION_LENGTH >= 0.5 \
                and len(measured[meas_real_start:]) > 2*BACKGOURD_SECTION_LENGTH:
            original_start = meas_real_start.copy()
            while max_continuous_zeros(bg_section)/BACKGOURD_SECTION_LENGTH >= 0.5:
                next_seg_start = meas_real_start + np.argmax(bg_section == 0) + 1
                if len(measured[meas_real_start + next_seg_start:]) < 2*BACKGOURD_SECTION_LENGTH:
                    logger.warning('No reasonable background estimate could be made')
                    meas_real_start = original_start
                    bg_section = measured[
                                 meas_real_start:meas_real_start + BACKGOURD_SECTION_LENGTH]
                    break
                meas_real_start = np.nonzero(measured[next_seg_start:])[0][0] + next_seg_start
                bg_section = measured[meas_real_start:meas_real_start + BACKGOURD_SECTION_LENGTH]

        bg_section_mean = np.mean(bg_section)
        found = False
        for i in reversed(range(meas_real_start, maxind)):
            if measured[i] <= bg_section_mean:
                bglim = i
                found = True
                break
        if not found:
            bglim = meas_real_start + 50
        bg_est = np.mean(measured[meas_real_start:bglim])
        if settings is not None:
            bg_percent = settings.lt_bg_percent / 100
        else:
            bg_percent = 0.05
        bg_est = min(bg_est, bg_percent * measured.max())  # bg shouldn't be more than given % of measured max
        if np.isnan(bg_est):  # bg also shouldn't be NaN
            bg_est = 0
        if return_bglim:
            return bglim
        else:
            return bg_est

    def setup_params(self, amp, shift, tau, fwhm=None):
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
        fwhm : list or float
            simulated IRF fwhm

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

        if self.simulate_irf:
            try:
                self.fwhm = fwhm[0]
                if fwhm[-1]:
                    self.fwhmmin = fwhm[0] - 0.0001
                    self.fwhmmax = fwhm[0] + 0.0001
                else:
                    self.fwhmmin = fwhm[1]
                    self.fwhmmax = fwhm[2]
            except (TypeError, IndexError) as e:  # If fwhm is not a list
                self.fwhm = fwhm
                self.fwhmmin = 0.05
                self.fwhmmax = 2

    @staticmethod
    def df_len(test_obj) -> int:
        df_num = 0
        if type(test_obj) in [list, np.ndarray]:
            df_num = len(test_obj)
        elif test_obj is not None:
            df_num = 1
        return df_num

    def results(self, tau, stds, avtaustd, shift, amp=1, fwhm=None):
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
        self.amp = amp
        self.shift = shift*self.channelwidth
        self.fwhm = fwhm
        self.stds = stds
        self.avtaustd = avtaustd

        param_df = self.df_len(tau) + (self.df_len(amp) - 1) + self.df_len(shift)

        measured = self.meas_bef_bg
        measured = measured[self.startpoint:self.endpoint]
        measured = measured / self.meas_max

        convd = self.convd + self.bg_n

        residuals = (convd - measured) * self.meas_max

        residuals = residuals / np.sqrt(np.abs(convd * self.meas_max))

        residualsnotinf = np.abs(residuals) != np.inf
        residuals = residuals[residualsnotinf]  # For some reason this is the only way I could find that works
        chisquared = np.sum((residuals ** 2)) / (np.size(measured) - param_df - 1)
        self.chisq = chisquared
        self.t = self.t[self.startpoint:self.endpoint]
        self.residuals = residuals
        self.dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)  # Durbin-Watson parameter
        self.dw_bound = self.durbinwatson()

        if self.ploton:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_yscale('log')
            ax1.plot(self.t, measured.flatten(), color='gray', linewidth=1)
            ax1.plot(self.t, self.convd.flatten(), color='C3', linewidth=1)
            textx = (self.endpoint - self.startpoint) * self.channelwidth * 1.4
            texty = measured.max()
            try:
                ax1.text(textx, texty, 'Tau = ' + ' '.join('{:#.3g}'.format(F) for F in tau))
                ax1.text(textx, texty / 2, 'Amp = ' + ' '.join('{:#.3g} '.format(F) for F in amp))
            except TypeError:  # only one component
                ax1.text(textx, texty, 'Tau = {:#.3g}'.format(tau))
            ax2.plot(self.t[residualsnotinf], residuals, '.', markersize=2)
            ax2.text(textx, residuals.max() / 1.1, r'$\chi ^2 = $ {:3.4f}'.format(chisquared))
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Weighted residual')
            ax1.set_ylabel('Number of photons in channel')
            plt.show()

    def makeconvd(self, shift, model, fwhm=None):
        """Makes a convolved decay using IRF and exponential model

        Parameters:
        -----------

        shift : float
            IRF colour shift
        model : ndarray
            Exponential model function

        """

        if fwhm is None:
            irf = self.irf
        else:  # Simulate gaussian irf with max at max of measured data
            irf, irft = self.sim_irf(self.channelwidth, fwhm, self.measured_unbounded)

        irf = colorshift(irf, shift)
        convd = convolve(irf, model)
        convd = convd[self.startpoint:self.endpoint]
        convd = convd / convd.sum()
        return convd

    @staticmethod
    def sim_irf(channelwidth, fwhm, measured):
        fwhm = fwhm / channelwidth
        c = fwhm / 2.35482
        t = np.arange(np.size(measured))
        maxind = np.argmax(measured)
        gauss = np.exp(-(t - maxind) ** 2 / (2 * c ** 2))
        irf = gauss * measured.max() / gauss.max()
        irft = t * channelwidth
        return irf, irft

    def durbinwatson(self):
        """Calculates Durbin-Watson lower bound.

        Based on Turner 2020 https://doi.org/10.1080/13504851.2019.1691711
        We use 1% or 5% critical bound for lower bound d_L of DW parameter.
        """
        numpoints = np.size(self.residuals)
        # for 5% and 1% calculate lower bound on critical value.
        for conf in [1, 5]:
            if self.numexp == 1:  # 2 params = lifetime + shift
                if conf == 5:
                    beta1 = -3.312097
                    beta2 = -3.332536
                    beta3 = -3.632166
                    beta4 = 19.31135
                else:
                    beta1 = -4.642915
                    beta2 = -4.052984
                    beta3 = 5.966592
                    beta4 = 14.91894
            elif self.numexp == 2:  # 4 params = lifetimes + shift + amplitude 1
                if conf == 5:
                    beta1 = -3.447993
                    beta2 = -4.229294
                    beta3 = -28.91627
                    beta4 = 80.00972
                else:
                    beta1 = -4.655069
                    beta2 = -7.296073
                    beta3 = -5.300441
                    beta4 = 60.11130
            elif self.numexp == 3:  # 6 params = lifetimes + shift + amplitudes 1+2
                # Currently we use the values for 5 parameters since Turner only computed for max 5!
                # The difference between 4 and 5 is not big for large n so between 5 and 6 should not be large either.
                # It's straightforward to compute values for 6 parameters but will just take some time.
                if conf == 5:
                    beta1 = -3.535331
                    beta2 = -4.085190
                    beta3 = -47.63654
                    beta4 = 127.7127
                else:
                    beta1 = -4.675041
                    beta2 = -8.518908
                    beta3 = -15.25711
                    beta4 = 96.32291

            dw = 2 + beta1 / np.sqrt(numpoints) + beta2 / numpoints + \
                 beta3 / (np.sqrt(numpoints) ** 3) + beta4 / numpoints ** 2
            dw = np.round(dw, 3)
            if conf == 5:
                dw5 = dw
            else:
                dw1 = dw

        # For < 1% use normal distribution
        var = (4 * numpoints ** 2 * (numpoints - 2)) / ((numpoints + 1) * (numpoints - 1) ** 3)
        std = np.sqrt(var)
        dw03 = np.round(2 - 3 * std, 3)
        dw01 = np.round(2 - 3.28 * std, 3)

        return dw5, dw1, dw03, dw01


class OneExp(FluoFit):
    """"Single exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 boundaries=None, addopt=None, ploton=False, fwhm=None):

        if tau is None:
            tau = 5
        if amp is None:
            amp = 1
        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, bg, irfbg,
                         boundaries, ploton, fwhm, numexp=1)

        if self.simulate_irf:
            paramin = [self.taumin[0], self.ampmin, self.shiftmin, self.fwhmmin]
            paramax = [self.taumax[0], self.ampmax, self.shiftmax, self.fwhmmax]
            paraminit = [self.tau[0], self.amp, self.shift, self.fwhm]
        else:
            paramin = [self.taumin[0], self.ampmin, self.shiftmin]
            paramax = [self.taumax[0], self.ampmax, self.shiftmax]
            paraminit = [self.tau[0], self.amp, self.shift]

        try:
            if addopt is None:
                param, pcov = curve_fit(self.fitfunc, self.t,#, self.t[self.startpoint: self.endpoint],
                                        self.measured, bounds=(paramin, paramax), p0=paraminit)
            else:
                param, pcov = curve_fit(self.fitfunc, self.t[self.startpoint: self.endpoint],
                                        self.measured, bounds=(paramin, paramax), p0=paraminit, **addopt)
        except ValueError as error:
            logger.error('Fitting failed')
        else:
            tau = param[0]
            amp = param[1]
            shift = param[2]
            stds = np.sqrt(np.diag(pcov))
            avtaustd = stds[0]

            if self.simulate_irf:
                fwhm = param[3]
            else:
                fwhm = None

            self.convd = self.fitfunc(self.t, tau, amp, shift, fwhm)
            self.results(tau, stds, avtaustd, shift, amp=1, fwhm=fwhm)

    def fitfunc(self, t, tau1, a, shift, fwhm=None):
        """Function passed to curve_fit, to be fitted to data"""

        model = a * np.exp(-t/tau1)
        return self.makeconvd(shift, model, fwhm)


class TwoExp(FluoFit):
    """"Double exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 boundaries=None, addopt=None, ploton=False, fwhm=None):

        if tau is None:
            tau = [1, 5]
        if amp is None:
            amp = [1, 1]

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, bg, irfbg, boundaries, ploton,
                         fwhm, numexp=2)

        if self.simulate_irf:
            paramin = self.taumin + self.ampmin + [self.shiftmin] + [self.fwhmmin]
            paramax = self.taumax + self.ampmax + [self.shiftmax] + [self.fwhmmax]
            paraminit = self.tau + self.amp + [self.shift] + [self.fwhm]
        else:
            paramin = self.taumin + self.ampmin + [self.shiftmin]
            paramax = self.taumax + self.ampmax + [self.shiftmax]
            paraminit = self.tau + self.amp + [self.shift]

        if addopt is None:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit)
        else:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit,
                                    **addopt)

        tau = param[0:2]
        amp = np.append(param[2], 1 - param[2])
        shift = param[4]
        stds = np.sqrt(np.diag(pcov))
        stds[3] = stds[2]  # second amplitude std is same as that of the first
        avtaustd = np.sqrt(tau[0] * amp[0] * np.sqrt((stds[0] / tau[0]) ** 2 + (stds[2] / amp[0])) +
                           tau[1] * amp[1] * np.sqrt((stds[1] / tau[1]) ** 2 + (stds[3] / amp[1])))

        if self.simulate_irf:
            fwhm = param[5]
        else:
            fwhm = None

        self.convd = self.fitfunc(self.t, tau[0], tau[1], amp[0], amp[1], shift, fwhm)
        self.results(tau, stds, avtaustd, shift, amp, fwhm)

    def fitfunc(self, t, tau1, tau2, a1, a2, shift, fwhm=None):
        """Function passed to curve_fit, to be fitted to data"""

        model = a1 * np.exp(-t / tau1) + (1 - a1) * np.exp(-t / tau2)
        return self.makeconvd(shift, model, fwhm)


class ThreeExp(FluoFit):
    """"Triple exponential fit. Takes exact same arguments as Fluofit"""

    def __init__(self, irf, measured, t, channelwidth, tau=None, amp=None, shift=None, bg=None, irfbg=None,
                 boundaries=None, addopt=None, ploton=False, fwhm=None):

        if tau is None:
            tau = [0.1, 1, 5]
        if amp is None:
            amp = [1, 1, 1]

        FluoFit.__init__(self, irf, measured, t, channelwidth, tau, amp, shift, bg, irfbg, boundaries, ploton,
                         fwhm, numexp=3)

        if self.simulate_irf:
            paramin = self.taumin + self.ampmin + [self.shiftmin] + [self.fwhmmin]
            paramax = self.taumax + self.ampmax + [self.shiftmax] + [self.fwhmmax]
            paraminit = self.tau + self.amp + [self.shift] + [self.fwhm]
        else:
            paramin = self.taumin + self.ampmin + [self.shiftmin]
            paramax = self.taumax + self.ampmax + [self.shiftmax]
            paraminit = self.tau + self.amp + [self.shift]

        if addopt is None:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit)
        else:
            param, pcov = curve_fit(self.fitfunc, self.t, self.measured, bounds=(paramin, paramax), p0=paraminit, **addopt)

        tau = param[0:3]
        amp = np.append(param[3:5], 1 - param[3] - param[4])
        shift = param[6]
        stds = np.sqrt(np.diag(pcov))
        stds[5] = np.sqrt(stds[3] ** 2 + stds[4] ** 2)  # third amp std is based on first two
        avtaustd = np.sqrt(tau[0] * amp[0] * np.sqrt((stds[0] / tau[0]) ** 2 + (stds[3] / amp[0])) +
                           tau[1] * amp[1] * np.sqrt((stds[1] / tau[1]) ** 2 + (stds[4] / amp[1])) +
                           tau[2] * amp[2] * np.sqrt((stds[2] / tau[2]) ** 2 + (stds[5] / amp[2])))

        if self.simulate_irf:
            fwhm = param[7]
        else:
            fwhm = None

        self.convd = self.fitfunc(self.t, tau[0], tau[1], tau[2], amp[0], amp[1], amp[2], shift, fwhm)
        self.results(tau, stds, avtaustd, shift, amp, fwhm)

    def fitfunc(self, t, tau1, tau2, tau3, a1, a2, a3, shift, fwhm=None):
        """Function passed to curve_fit, to be fitted to data"""

        model = a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + (1 - a1 - a2) * np.exp(-t / tau3)
        return self.makeconvd(shift, model, fwhm)


class FittingParameters:
    def __init__(self, parent: LifetimeController):
        self.parent = parent
        self.fpd = self.parent.fitparamdialog
        self.irf = None
        self.tau = None
        self.amp = None
        self.shift = None
        self.decaybg = None
        self.irfbg = None
        self.start = None
        self.autostart = 'Manual'
        self.end = None
        self.autoend = False
        self.numexp = None
        self.addopt = None
        self.fwhm = None

    def getfromdialog(self):
        self.numexp = int(self.fpd.combNumExp.currentText())
        if self.numexp == 1:
            self.tau = [[self.get_from_gui(i) for i in
                         [self.fpd.line1Init, self.fpd.line1Min, self.fpd.line1Max,
                          self.fpd.check1Fix]]]
            self.amp = [[self.get_from_gui(i) for i in
                         [self.fpd.line1AmpInit, self.fpd.line1AmpMin, self.fpd.line1AmpMax,
                          self.fpd.check1AmpFix]]]
            self.amp[0][0] = 1

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
            self.amp[1][0] = 1 - self.amp[0][0]

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
            self.amp[2][0] = 1 - self.amp[0][0] - self.amp[1][0]

        self.shift = [self.get_from_gui(i) for i in [self.fpd.lineShift, self.fpd.lineShiftMin, self.fpd.lineShiftMax,
                                                     self.fpd.checkFixIRF]]
        self.decaybg = self.get_from_gui(self.fpd.lineDecayBG)
        self.irfbg = self.get_from_gui(self.fpd.lineIRFBG)
        self.start = self.get_from_gui(self.fpd.lineStartTime)
        self.autostart = self.get_from_gui(self.fpd.comboAutoStart)
        self.end = self.get_from_gui(self.fpd.lineEndTime)
        self.autoend = bool(self.get_from_gui(self.fpd.checkAutoEnd))

        if self.fpd.lineAddOpt.text() != '':
            self.addopt = self.fpd.lineAddOpt.text()
        else:
            self.addopt = None

        if self.fpd.checkSimIRF.isChecked():
            self.fwhm = [self.get_from_gui(i) for i in [self.fpd.fwhmInit, self.fpd.fwhmMin,
                                                        self.fpd.fwhmMax, self.fpd.checkfwhmFix]]
        else:
            self.fwhm = None

    # @staticmethod
    def get_from_gui(self, guiobj):
        if type(guiobj) == QLineEdit:
            invalid = 0
            if guiobj in [*self.fpd.tau_edits, *self.fpd.amp_edits]:
                invalid = 0.0001
            elif guiobj is self.fpd.time_edits[1]:
                invalid = None
            elif guiobj in self.fpd.bg_edits:
                invalid = None

            if guiobj.text() == '':
                return invalid
            else:
                text = guiobj.text()
                num_chs = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',']
                if all([ch in num_chs for ch in text]):
                    if text.count('-') > 1 or text.count(',') > 1 or text.count('.') > 1:
                        return invalid
                    if text[0] in ['.', ',']:
                        text = '0' + text
                    if text[0] == '-':
                        if len(text) == 1:
                            text = 0
                        elif text[1] in ['.', ',']:
                            text = text[0] + '0' + text[1:]
                    float_text = float(text)
                    if float_text == 0:
                        float_text = invalid
                    return float_text
                else:
                    return invalid
        elif type(guiobj) == QCheckBox:
            return float(guiobj.isChecked())
        elif type(guiobj) == QComboBox:
            return guiobj.currentText()


class FittingDialog(QDialog, UI_Fitting_Dialog):
    """Class for dialog that is used to choose lifetime fit parameters."""

    def __init__(self, mainwindow, lifetime_controller):
        QDialog.__init__(self)
        UI_Fitting_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.lifetime_controller = lifetime_controller
        self.pgFitParam.setBackground(background=None)

        self.settings = mainwindow.settings
        # self.load_settings()

        self.checkSimIRF.stateChanged.connect(self.enable_sim_vals)

        self.tau_edits = [self.line1Init, self.line2Init1, self.line2Init2, self.line3Init1,
                          self.line3Init2, self.line3Init3]
        self.tau_min_edits = [self.line1Min, self.line2Min1, self.line2Min2, self.line3Min1,
                              self.line3Min2, self.line3Min3]
        self.tau_max_edits = [self.line1Max, self.line2Max1, self.line2Max2, self.line3Max1,
                              self.line3Max2, self.line3Max3]

        self.amp_edits = [self.line1AmpInit, self.line2AmpInit1, self.line2AmpInit2,
                          self.line3AmpInit1, self.line3AmpInit2, self.line3AmpInit3]
        self.amp_min_edits = [self.line1AmpMin, self.line2AmpMin1, self.line2AmpMin2,
                              self.line3AmpMin1, self.line3AmpMin2, self.line3AmpMin3]
        self.amp_max_edits = [self.line1AmpMax, self.line2AmpMax1, self.line2AmpMax2,
                              self.line3AmpMax1, self.line3AmpMax2, self.line3AmpMax3]

        self.bg_edits = [self.lineDecayBG, self.lineIRFBG]
        self.time_edits = [self.lineStartTime, self.lineEndTime]

        self.irf_shift_edits = [self.lineShift, self.lineShiftMin, self.lineShiftMax]

        reg_exp = QRegExp("[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?")
        reg_val = QRegExpValidator(reg_exp)

        self._reg_exp_validator = reg_val
        for tau_edit in self.tau_edits:
            tau_edit.setValidator(self._reg_exp_validator)
            tau_edit.textChanged.connect(self.updateplot)

        for tau_min_max_edit in [*self.amp_min_edits, *self.tau_max_edits]:
            tau_min_max_edit.setValidator(self._reg_exp_validator)

        for amp_edit in self.amp_edits:
            amp_edit.setValidator(self._reg_exp_validator)
            amp_edit.textChanged.connect(self.updateplot)

        for amp_min_max_edit in [*self.amp_min_edits, *self.amp_max_edits]:
            amp_min_max_edit.setValidator(self._reg_exp_validator)

        for bg_edit in self.bg_edits:
            bg_edit.setValidator(self._reg_exp_validator)
        for time_edit in self.time_edits:
            time_edit.setValidator(self._reg_exp_validator)
            time_edit.textEdited.connect(self.updateplot)

        for irf_shift_edit in self.irf_shift_edits:
            irf_shift_edit.setValidator(self._reg_exp_validator)
            irf_shift_edit.textEdited.connect(self.updateplot)

        self.combNumExp.currentIndexChanged.connect(self.updateplot)
        #  TODO: regexvalidator for sim irf parameters

        self.comboAutoStart.currentTextChanged.connect(self.updateplot)
        self.checkAutoEnd.stateChanged.connect(self.updateplot)

        # for widget in self.findChildren(QLineEdit):
        #     widget.textChanged.connect(self.text_changed)
        # for widget in self.findChildren(QCheckBox):
        #     widget.stateChanged.connect(self.text_changed)
        # for widget in self.findChildren(QComboBox):
        #     widget.currentTextChanged.connect(self.text_changed)
        # self.updateplot()

        # self.lineStartTime.setValidator(QIntValidator())
        # self.lineEndTime.setValidator(QIntValidator())

    def load_settings(self):
        settings_file_path = fm.path('settings.json', fm.Type.ProjectRoot)
        with open(settings_file_path, 'r') as settings_file:
            if not hasattr(self, 'settings'):
                self.settings = Settings()
            self.settings.load_settings_from_file(file_or_path=settings_file)

    def enable_sim_vals(self, enable):
        self.fwhmInit.setEnabled(enable)
        self.fwhmMin.setEnabled(enable)
        self.fwhmMax.setEnabled(enable)
        self.checkfwhmFix.setEnabled(enable)
        self.updateplot()

    def updateplot(self, *args):
        if not hasattr(self.lifetime_controller, 'fitparam'):
            return
        else:
            self.lifetime_controller.fitparam.getfromdialog()
            crit_params = list()
            crit_params.extend(self.lifetime_controller.fitparam.tau)
            crit_params.extend(self.lifetime_controller.fitparam.amp[:-1])
            for param in crit_params:
                if None in param:
                    return

        try:
            model = self.make_model()
        except Exception as err:
            logger.error('Error Occured: ' + str(err))
            return

        channelwidth = self.mainwindow.current_particle.channelwidth
        fp = self.lifetime_controller.fitparam

        #  TODO: try should contain as little code as possible
        try:
            if self.mainwindow.current_particle.level_selected is None:
                histogram = self.mainwindow.current_particle.histogram
            else:
                level = self.mainwindow.current_particle.level_selected
                if level <= self.mainwindow.current_particle.num_levels - 1:
                    histogram = self.mainwindow.current_particle.levels[level].histogram
                else:
                    group = level - self.mainwindow.current_particle.num_levels
                    histogram = self.mainwindow.current_particle.groups[group].histogram
            decay = histogram.decay
            decay = decay / decay.sum()
            t = histogram.t

        except AttributeError:
            logger.error('No Decay!')
        else:
            if fp.irf is not None or fp.fwhm is not None:
                try:
                    if fp.fwhm is None:
                        irf = fp.irf
                        irft = fp.irft
                    else:
                        irf, irft = FluoFit.sim_irf(channelwidth, fp.fwhm[0], decay)
                except AttributeError:
                    logger.error('No IRF!')
                    return

                shift, decaybg, irfbg, start, autostart, end, autoend = self.getparams()

                shift = shift / channelwidth
                # irf = tcspcfit.colorshift(irf, shift)
                irf = colorshift(irf, shift)
                convd = scipy.signal.convolve(irf, model)
                convd = convd[:np.size(irf)]
                convd = convd / convd.sum()

                bg = FluoFit.estimate_bg(decay, settings=self.settings)
                start = int(start / channelwidth)
                end = int(end / channelwidth) if end is not None else None
                print(end)
                start, end = FluoFit.calculate_boundaries(decay, [start, end, autostart, autoend],
                                                          bg, self.settings, channelwidth)
                print(end)
                if autostart != 'Manual':
                    self.lineStartTime.setText(f'{start * channelwidth:.3g}')
                if autoend:
                    self.lineEndTime.setText(f'{end * channelwidth:.3g}')

                convd = convd[irft > 0]
                irft = irft[irft > 0]

                plot_item = self.pgFitParam.getPlotItem()

                plot_item.setLogMode(y=True)
                plot_pen = QPen()
                plot_pen.setWidthF(1.5)
                plot_pen.setJoinStyle(Qt.RoundJoin)
                plot_pen.setColor(QColor('blue'))
                plot_pen.setCosmetic(True)

                plot_item.clear()
                plot_item.plot(x=t, y=np.clip(decay, a_min=0.000001, a_max=None), pen=plot_pen,
                               symbol=None)

                plot_pen = QPen()
                plot_pen.setWidthF(2)
                plot_pen.setJoinStyle(Qt.RoundJoin)
                plot_pen.setCosmetic(True)
                plot_pen.setColor(QColor('dark blue'))
                plot_item.plot(x=irft, y=np.clip(convd, a_min=0.000001, a_max=None), pen=plot_pen,
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
                    plot_pen = QPen()
                    plot_pen.setWidthF(2.5)
                    plot_pen.setJoinStyle(Qt.RoundJoin)
                    plot_pen.setCosmetic(True)
                    plot_pen.setColor(QColor('gray'))
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
        shift = fp.shift[0]
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
        autostart = fp.autostart
        end = fp.end
        autoend = fp.autoend
        if end is None and irf is not None:
            end = np.size(irf)
        return shift, decaybg, irfbg, start, autostart, end, autoend

    def make_model(self):
        fp = self.lifetime_controller.fitparam
        t = self.mainwindow.current_particle.histogram.t
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