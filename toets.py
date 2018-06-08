import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from matplotlib import rc
rc('text', usetex=True)

data = np.loadtxt('/home/bertus/temp/gendata10_25.txt')
# simdata = np.loadtxt('/home/bertus/temp/convd.txt')

a1 = 0.624
a2 = 0.423

t = data[:, 0]
measured = data[:, 1]
irf = data[:, 2]
scale = np.max(measured)

startpoint = 300
measured = measured[startpoint:4096]
irf = irf[startpoint:4096]


def fitfunc(t, tau1, tau2, scale, a1, a2):
    model = a1 * np.exp(-t/tau1) + a2 * np.exp(-t/tau2)

    convd = convolve(irf, model)
    convd = convd * scale/np.max(convd)
    convd = convd[:4096-startpoint]
    return convd


popt, pcov = curve_fit(fitfunc, t, measured, bounds=([1, 1, 0, 0, 0], [100, 100, 11000, 1, 1]),
                       p0=[10, 25, scale, 0.7, 0.3])
print(popt)
convd = fitfunc(t, popt[0], popt[1], popt[2], popt[3], popt[4])

residuals = convd - measured
sumresiduals = sum((convd[measured>0] - measured[measured>0]) ** 2 / np.abs(measured[measured>0]), 0.001) /np.size(measured[measured>0])
print(sumresiduals)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_yscale('log')
ax1.set_ylim([1, 50000])
ax1.plot(measured)
ax1.plot(convd)
ax1.plot(irf)
ax1.text(1500, 20000, 'Tau = %5.3f,     %5.3f' %(popt[0], popt[1]))
ax1.text(1500, 8000, 'Amp = %5.3f,     %5.3f' %(popt[3], popt[4]))

ax2.plot(residuals, '.')
ax2.text(2500, 200, r'$\chi ^2 = $ %4.3f' %sumresiduals)
plt.show()

# plt.figure(dpi=600)
# plt.plot(measured)
# plt.plot(convd)
# plt.plot(irf)
# plt.show()
