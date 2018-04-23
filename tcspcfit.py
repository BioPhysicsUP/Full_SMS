""" Module for fitting TCSPC data.

Based on MATLAB code by Jörg Enderlein
https://www.uni-goettingen.de/en/513325.html
"""

# from math import *
# from statistics import *
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.optimize import minimize, curve_fit
from matplotlib import pyplot as plt


class InputError(Exception):
    """Raised when input is incompatible with a given function"""
    pass


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

    mm = np.mean(irf[-11:])
    decaylength = np.size(decay, 1)
    irflength = np.size(irf, 1)
    if decaylength > irflength:
        irf = np.append(irf, mm * np.ones(decaylength - irflength))
    else:
        irf = irf[:, :decaylength]

    # Duplicate rows of irf so dimensions are the same as decay and convolve.
    y = np.float_(np.real(ifft(np.outer(np.ones(np.size(decay, 0)), fft(irf)) * fft(decay))))

    # t needs to have the same length as irf (irflength) in each case:
    if irflength <= decaylength:
        t = np.arange(0, irflength)
    else:
        t = np.concatenate((np.arange(0, decaylength), np.arange(0, irflength - decaylength)))

    # Either remove from y or add from start to end so that y has same length as irf.
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
    new_index_left = np.fmod(np.fmod(t - np.floor(shift) - 1, irflength) + irflength, irflength).astype(int)
    new_index_right = np.fmod(np.fmod(t - np.ceil(shift) - 1, irflength) + irflength, irflength).astype(int)
    integer_left_shift = irf[new_index_left]
    integer_right_shift = irf[new_index_right]
    irs = (1 - shift + np.floor(shift)) * integer_left_shift + (shift - np.floor(shift)) * integer_right_shift
    return makerow(irs)


def lsfit(param, irf, measured, period):
    """Returns the Least-Squares deviation between measured and computed values.

    Assumes a function of the form:

    measured =  yoffset + A1*convol(irf,exp(-t/tau1/(1-exp(-p/tau1))) + ...

    The only use case of lsfit is in the call to
    scipy.optimize.minimize(). minimize() minimizes the output of lsfit by
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
    cshift = param[0]
    offset = param[1]
    tau = param[2:]

    # Create data from parameters
    calculated, irs = create_exp(tp, tau, period, irf, irflength, t)

    # Calculate least-squares error between calculated and measured data
    amplitudes, residuals, rank, s = np.linalg.lstsq(calculated.T, measured, rcond=None)
    calculated = np.matmul(calculated.T, amplitudes)
    err = np.sum(((calculated - measured) ** 2) / np.abs(calculated)) / (irflength - np.size(tau, 0))
    # ind = np.nonzero(measured > 0)
    # print(calculated[ind])
    # err = np.sum(measured[ind] * np.log(np.abs(measured[ind] / calculated[ind])) - measured[ind] + calculated[ind]) / (irflength - np.size(tau));
    return err#, amplitudes, calculated


def model(model_in, tau1, tau2):

    period = model_in[-1:].astype(int)[0]
    irflength = model_in[-2:-1].astype(int)[0]
    irf = model_in[:irflength]
    measured = model_in[irflength:-2]

    irf = irf.flatten()

    t = np.arange(irflength)
    tp = np.arange(1, period)
    tau = np.array([tau1, tau2])

    # Create data from parameters
    calculated, irs = create_exp(tp, tau, period, irf, irflength, t)
    amplitudes, residuals, rank, s = np.linalg.lstsq(calculated.T, measured, rcond=None)
    calculated = np.matmul(calculated.T, amplitudes)
    return calculated.flatten()


def fluofit(irf, measured, period, channelwidth, tau, taubounds=None, init=0, ploton=False):
    """Fit of a multi-exponential decay curve.

    Arguments:
    irf -- Instrumental Response Function
    measured -- Fluorescence decay data
    period -- Time between laser exciation pulses (in nanoseconds)
    channelwidth -- Time width of one TCSPC channel (in nanoseconds)
    tau -- Initial guess times
    taubounds -- limits for the lifetimes guess times - defaults to 0<tau<100
                 format: [[tau1_min, tau1_max], [tau2_min, tau2_max], ...]
    init -- Whether to use a initial guess routine or not

    Output:
    c -- Color Shift (time shift of the IRF w.r.t. the fluorescence curve)
    offset -- Offset
    amplitudes -- Amplitudes of the different decay components
    tau	-- Decay times of the different decay components
    dc -- Color shift error
    doffset -- Offset error
    dtau -- Decay times error
    irs -- IRF, shifted by the value of the colorshift
    separated_decays -- Fitted fluorecence component curves
    t -- time axis
    chisquared -- chi squared value
    """

    irf = irf.flatten()
    measured = measured.flatten()
    irflength = np.size(irf)
    init = 0
    offset = 0

    if init>0:
        # [cx, tau, ~, c] = DistFluofit(irf, measured, period, channelwidth, [-3 3], 0, 0)
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
        cshift = 0

    if taubounds is None:
        taubounds = np.concatenate((np.zeros((np.size(tau, 1), 1)), 100 * np.ones((np.size(tau, 1), 1))), axis=1)
    taubounds = taubounds / channelwidth
    taubounds = tuple(map(tuple, taubounds))  # convert to tuple as required by minimize()
    # tau_lower = taubounds[:, 0]
    # tau_upper = taubounds[:, 1]

    period = period / channelwidth
    tau = tau / channelwidth
    t = np.arange(np.size(measured))
    tp = np.arange(1, period)
    taulength = np.size(tau)
    # decay = np.matmul(np.exp(np.matmul(-(tp-1), (1/tau))), np.diagflat(1/(1-np.exp(-period/tau))))
    # decay = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1-np.exp(-period / tau)))))

    # irs = (1-c+np.floor(c))*irf[(np.fmod(np.fmod(t-np.floor(c)-1, irflength)+irflength, irflength)).astype(int)] \
        # + (c-np.floor(c))*irf[(np.fmod(np.fmod(t-np.ceil(c)-1, irflength)+irflength, irflength)).astype(int)]
    # irs = np.reshape(irs, (1, -1))
    # calculated = convol(np.reshape(irs, (1, -1)), decay.T)
    # calculated = np.reshape([np.ones([np.size(calculated,1), 1]), calculated], (-1, 1))
    # decay = np.exp(np.matmul(np.outer(-(tp-1), (1/tau)), np.diag(1 / (1-np.exp(-period / tau)))))
    # irs = [(1 - c + np.floor(c)) * irf[0, np.fmod(np.fmod(t - np.floor(c) - 1, irflength) + irflength, irflength).astype(int)]\
    #     + (c - np.floor(c)) * irf[0, np.fmod(np.fmod(t - np.ceil(c) - 1, irflength) + irflength, irflength).astype(int)]]

    # calculated = convol(irs, decay.T)
    # print(np.shape(calculated), np.shape(measured))
    # calculated = np.concatenate((np.ones((np.size(calculated, 0), 1)), calculated), axis=1)
    # print(np.shape(irf), np.shape(measured))
    # amplitudes, residuals, rank, s = np.linalg.lstsq(calculated.T, measured.T)
    # calculated = np.matmul(calculated.T, amplitudes)

    # param = np.insert(tau, 0, 999)
    param = np.array([cshift, offset])
    param = np.append(param, tau)

    # Decay times and offset are assumed to be positive.
    offs_lower = np.array([-10])
    offs_upper = np.array([10])
    cshift_lower = np.array([-1])
    cshift_upper = np.array([1])

    # lowerbounds = np.concatenate((offs_lower, cshift_lower, tau_lower))
    # upperbounds = np.concatenate((offs_upper, cshift_upper, tau_upper))
    bounds = (((-1/channelwidth, 1/channelwidth), (0, None)) + taubounds)
    params = []
    result = minimize(lsfit, param, args=(irf, measured, period), method='Nelder-Mead', tol=1e-16)#, bounds=bounds)
    param = result.x
    tau = param[2:]
    # paramvariance = np.diag(result.hess_inv.matmat(np.identity(4)))
    # print(paramvariance)

    model_in = np.append(measured.flatten(), [irflength, period])
    model_in = np.concatenate((irf.flatten(), model_in))

    # print(tau)
    # param, pcov = curve_fit(model, model_in, measured.flatten(), tau.flatten(), bounds=(tau_lower, tau_upper))
    # dtau = np.sqrt(np.diag(pcov))
    # cshift = param[0]
    # # dc = dparam(1)  # TODO: Get errors out of minimisation
    # tau = param
    # # dtau = dparam[2:np.shape(param)]

    # Calculate values from parameters
    calculated, irs = create_exp(tp, tau, period, irf, irflength, t)

    calculated = calculated.T/np.ones((irflength, 1))*np.sum(calculated)  # Normalize
    amplitudes, residuals, rank, s = np.linalg.lstsq(calculated, measured, rcond=None)

    # print(np.sqrt((residuals/(irflength - 4)) * paramvariance))

    # Put individual decay curves into rows
    separated_decays = calculated * np.matmul(np.ones(np.size(calculated, 1)), amplitudes.T)
    total_decay = np.matmul(calculated, amplitudes).T

    # plt.figure(dpi=800)
    plt.plot(total_decay)
    plt.plot(measured)
    # plt.show()
    #     dtau = dtau
    #     dc = channelwidth*dc
    chisquared = sum((measured - total_decay) ** 2. / abs(total_decay)) / (irflength - taulength)
    t = channelwidth * t
    tau = channelwidth * tau.T
    cshift = channelwidth * cshift
    offset = separated_decays[0]
    return cshift, offset, amplitudes, tau, 0, 0, irs, separated_decays, t, chisquared
