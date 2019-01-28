from matplotlib import rc
from tcspcfit import *
import h5py

rc('text', usetex=True)

irf_file = h5py.File('IRF_680nm.h5', 'r')
meas_file = h5py.File('LHCII_630nW.h5', 'r')
# meas_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/Johanette/40.h5', 'r')

i = 1
while True:
    irf_data = irf_file['Particle 1/Micro Times (s)'][:]
    meas_data = meas_file['Particle {}/Micro Times (s)'.format(i)][:]
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
    t = t[:-21]

    np.savetxt('fast/Particle {}.fst'.format(i), np.column_stack((t, irf, measured)))
    i += 1

