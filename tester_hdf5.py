from matplotlib import rc
from tcspcfit import *
import h5py

rc('text', usetex=True)

irf_file = h5py.File('IRF_680nm.h5', 'r')
meas_file = h5py.File('LHCII_630nW.h5', 'r')
irf_data = irf_file['Particle 1/Micro Times (s)'][:]

particlename = 'Particle 7'
meas_data = meas_file[particlename + '/Micro Times (s)'][:]
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
print(channelwidth)
# plt.plot(measured)
# plt.plot(irf)
# plt.show()

# How to specify initial values
# [init, min, max, fix]
tau = [[3, 0.01, 10, 0],
       [0.1, 0.01, 10, 0]]

amp = [[0.01, 0, 10, 0],
       [0.01, 0, 10, 0]]

shift = [3, -100, 2000, 0]  # Units are number of channels

# Object orientated interface: Each fit is an object
# fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=amp, shift=shift, startpoint=800, ploton=True)
fit = TwoExp(irf, measured, t, channelwidth, tau=tau, amp=amp, shift=shift, startpoint=0, ploton=True)
fit1 = OneExp(irf, measured, t, channelwidth, tau=3, shift=shift, startpoint=0, ploton=True)
# Initial guess not necessarily needed:
fit2 = OneExp(irf, measured, t, channelwidth, ploton=True)

tau = fit.tau
amp = fit.amp
shift = fit.shift
chisq = fit.chisq
bg = fit.bg
irfbg = fit.irfbg

print('Lifetimes: {:4.2f} ns, {:5.3f} ns'.format(tau[0], tau[1]))
print('Amplitudes: {:4.2f} %, {:4.2f} %'.format(amp[0], amp[1]))
print('IRF Shift: {:6.4f} ns'.format(shift * channelwidth))
print('Chi sq: {:6.4f}'.format(chisq))
print('Decay bg: {:6.4f}'.format(bg))
print('IRF bg: {:6.4f}'.format(irfbg))

savedata = np.column_stack((fit.t, fit.measured, fit.convd, fit.residuals))
np.savetxt(particlename + '-decay.txt', savedata, fmt='%6f', header='time (ns)  measured    fitted  residuals')


