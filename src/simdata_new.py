import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
import h5py
import tcspcfit

# matplotlib.use('TkAgg')

# file = h5py.File('IRF SPCM 680nm.h5', 'r')
file = h5py.File('../IRF 680nm.h5', 'r')
# file = h5py.File('IRF_AQR_EMN.h5', 'r')
# file = h5py.File('IRF.h5', 'r')
# file = h5py.File('IRF NIM.h5', 'r')
# file = h5py.File('beads_both_cards.h5', 'r')
# file = h5py.File('IRF FAST 680nm.h5', 'r')
datadict = file['Particle 1']
microtimes = datadict['Micro Times (s)']
microtimes = microtimes - np.min(microtimes)
differences = np.diff(np.sort(microtimes[:]))
channelwidth = np.unique(differences)[1]
t = np.arange(0, 25, channelwidth)

irf, t = np.histogram(microtimes, bins=t)
halfmaxind = np.where(irf > 0.5*irf.max())
fwhm = t[halfmaxind[0][-1]] - t[halfmaxind[0][0]]
# print(halfmaxind[-1])
# print(fwhm)
t = t[:-1]

# plt.plot(t, irf)
# plt.show()

# file1 = h5py.File('IRF SPCM 680nm.h5', 'r')
# datadict1 = file1['Particle 1']
# microtimes1 = datadict['Micro Times (ns)']
# microtimes1 = microtimes1 - np.min(microtimes1)
# t1 = np.arange(0, 25, channelwidth)
# irf1, t1 = np.histogram(microtimes1, bins=t1)
#
centre = t[np.argmax(irf)]
sigma = fwhm / 2.355
gauss = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (t - centre) ** 2 / (2 * sigma ** 2))
gauss = gauss * np.trapz(irf, t)
irf = gauss
# # irf = irf + 10  # bg of 1
#
# plt.figure()
# plt.plot(t, gauss)
# plt.plot(t, irf)
# plt.show()
#
# fastamp = 0.1
#
#


def fitfunc1(t, tau1):
    model = np.exp(-t / tau1)
    irs = tcspcfit.colorshift(irf, 20)
    convd = convolve(irs, model)
    # convd = model
    convd = convd * (10 / convd.max())  # peak of 100 is about what you get at low excitation power
    return convd


def fitfunc(t, amp1, tau1, tau2):
    model = amp1 * np.exp(-t / tau1) + (1 - amp1) * np.exp(-t / tau2)
    irs = tcspcfit.colorshift(irf, 0.1)
    convd = convolve(irs, model)
    convd = convd * (100 / convd.max())  # peak of 100 is about what you get at low excitation power
    return convd


# convd = fitfunc1(t,1.3)
# tau = [[1.3, 1.2, 1.4, 0]]
convd = fitfunc(t,0.5, 0.08, 3.4)
tau = [[0.08, 0.05, 0.1, 0], [3.4, 3.2, 3.6, 0]]
amp = [[0.5, 0.4, 0.6, 0]]
# amp = [[0.1, 0, 1, 1], [0.9, 0, 1, 1]]
shift = [20, -300, 300, 0]


convd = convd + 5
convdsum = convd.sum()
convdnoise = np.random.poisson(convd)
# convdnoise[convdnoise <= 0] = 0
# convdsum = convdnoise.sum()
# convdnoise = convdnoise / convdsum
# convdnoise = convd

bg_est = tcspcfit.FluoFit.estimate_bg(convdnoise)
# print(bg_est * convdsum)
print(bg_est)

# fit = tcspcfit.OneExp(irf, convdnoise, t, channelwidth, tau=tau, shift=shift, bg=0, amp=bg_est*convdsum)
# fit = tcspcfit.OneExp(irf, convdnoise, t, channelwidth, tau=tau, shift=shift, bg=0, amp=bg_est)
fit = tcspcfit.TwoExp(irf, convdnoise, t, channelwidth, tau=tau, shift=shift, bg=0, amp=bg_est)
print('# photons: ', convdnoise.sum())
print('# fit photos: ', fit.convd.sum())
print('Tau: ', fit.tau)
print('Amp: ', fit.amp)
print('IRFbg: ', fit.irfbg)
print('bg: ', fit.bg)
print('shift: ', fit.shift)

resid = (fit.convd - fit.measured) #/ np.sqrt(np.abs(fit.measured))
acf = np.correlate(resid, resid, 'full')
chisq = np.sum(resid ** 2)
print(chisq)
# plt.figure()
# plt.plot(resid, '.')
# plt.show()
plt.figure()
plt.plot(convdnoise)
plt.plot(convd)
# plt.plot(irf)
# plt.plot(fit.measured)
plt.plot(fit.convd)
# plt.plot(fit.irf)
plt.show()
#
# # param, pcov = curve_fit(fitfunc, t, convdnoise, p0=[0.01, 0.08, 3.4])
# # print(param)
# # plt.plot(fitfunc(t, param[0], param[1], param[2]))
# # plt.plot(convdnoise)
# # plt.show()
#
