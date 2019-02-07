from matplotlib import rc
from tcspcfit import *
import h5py

rc('text', usetex=True)

irf_file = h5py.File('IRF_680nm.h5', 'r')
meas_file = h5py.File('LHCII_630nW.h5', 'r')
# meas_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/Johanette/40.h5', 'r')
irf_data = irf_file['Particle 1/Micro Times (s)'][:]
meas_data = meas_file['Particle 7/Micro Times (s)'][:]
# meas_data += 10
irf_data = irf_data - np.sort(meas_data)[1]
meas_data = meas_data - np.sort(meas_data)[1]

# Determine the channelwidth. I think this should rather be saved as a data attribute.
differences = np.diff(np.sort(irf_data))
channelwidth = np.unique(differences)[1]
assert channelwidth == np.unique(np.diff(np.sort(meas_data)))[1]

# Make histograms out of the microtimes
tmin = min(irf_data.min(), meas_data.min())
tmax = max(irf_data.max(), meas_data.max())
window = tmax - tmin
numpoints = int(window // channelwidth)
t = np.linspace(0, window, numpoints)
print(t)
irf, t = np.histogram(irf_data, bins=t)
measured, t = np.histogram(meas_data, bins=t)
irf = irf[:-20]
measured = measured[:-20]
t = t[:-20]
# plt.plot(measured)
# plt.plot(irf)
# plt.show()

# How to specify initial values
# [init, min, max, fix]
tau = [[3.333, 0.01, 10, 1],
      [0.015, 0.01, 10, 1]]
# tau = [[3.652, 0.01, 10, 0],
#        [1.037, 0.01, 10, 0]]

# shift = [-0.0487/channelwidth, -100, 2000, 0]  # Units are number of channels
shift = [3.1018 / channelwidth, -100, 2000, 1]  # Units are number of channels

# Only need to specify one amplitude as they sum to 100%
# [init (amp1 %), fix]
amp = [0.0064, 0.198, 0]

# Object orientated interface: Each fit is an object
fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=amp, shift=shift, startpoint=0, bg=2.602, irfbg=0.6, ploton=True)
# fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=amp, shift=shift, ploton=True)
# fit1 = OneExp(irf, measured, t, channelwidth, tau=3, shift=shift, startpoint=800, ploton=True)
# Initial guess not necessarily needed:
# fit2 = OneExp(irf, measured, t, channelwidth, ploton=True)

tau = fit.tau
dtau = fit.dtau
amp = fit.amp
shift = fit.shift
chisq = fit.chisq
bg = fit.bg
irfbg = fit.irfbg
relamp = amp * 100 / np.sum(amp)

print('Lifetimes: {:4.2f} ns, {:5.3f} ns'.format(tau[0], tau[1]))
print('Lifetime err: {:4.2f} ns, {:5.3f} ns'.format(dtau[0], dtau[1]))
print('Amplitudes: {:6.4f}, {:6.4f}'.format(amp[0], amp[1]))
print('Relative Amplitudes: {:6.4f} %, {:6.4f} %'.format(relamp[0], relamp[1]))
print('IRF Shift: {:6.4f} ns'.format(shift))
print('Chi sq: {:6.4f}'.format(chisq))
print('Decay bg: {:6.4f}'.format(bg))
print('IRF bg: {:6.4f}'.format(irfbg))
# print('scale factor: {}'.format(fit.scalefactor))

# fast_data = np.loadtxt('particle 15 - fit.txt', skiprows=6, comments='*')
# plt.figure()
# plt.plot(fast_data[:, 2])
# # plt.plot(fast_data[:, 1], '.')
# # plt.plot(fit.measured[234:])
# # plt.plot(measured[1034:], '.')
# # plt.plot(fit.convd[234:])
# fast_model = 0.0149 * np.exp(-t / 3.65) + 0.0342 * np.exp(-t / 1.037)
# irs = colorshift(irf, -0.0487 / channelwidth)
# fast_convd = np.convolve(irs, fast_model)
# plt.plot(fast_convd[1034:])
#
# # plt.show()
#
# plt.figure()
#
# # measured = measured[1034:]
# fastres = fast_data[:, 2] - measured[:-1]
# res = fit.convd - measured[:-3]
# fastres = fastres / np.sqrt(np.abs(measured[:-1]))
# res = res / np.sqrt(np.abs(measured[:-3]))
# plt.plot(fastres, '.')
# # plt.show()
# plt.figure()
# plt.plot(res, '.')
#
# chisquared = np.sum((res ** 2)) / (np.size(measured) - 4 - 1)
# chisquared_fast = np.sum((fastres ** 2)) / (np.size(measured) - 4 - 1)
# print(chisquared)
# print(chisquared_fast)

# plt.show()




