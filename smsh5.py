"""Module for handling SMS data from HDF5 files

Bertus van Heerden
University of Pretoria
2018
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt


class H5dataset:

    def __init__(self, filename):

        self.file = h5py.File(filename, 'r')
        self.particles = []
        for particlename in self.file.keys():
            self.particles.append(Particle(particlename, self))
        self.numpart = len(self.particles)
        assert self.numpart == self.file.attrs['# Particles']
        self.channelwidth = None

    def makehistograms(self):
        """Put the arrival times into histograms"""

        for particle in self.particles:
            particle.makehistogram()

    def binints(self, binsize):
        """Bin the absolute times into traces using binsize"""

        for particle in self.particles:
            particle.binints(binsize)


class Particle:

    def __init__(self, name, dataset, tmin, tmax, channelwidth):#, number, irf, tmin, tmax, channelwidth=None):

        self.name = name
        self.dataset = dataset
        self.datadict = self.dataset.file[self.name]
        self.microtimes = self.datadict['Micro Times (s)']
        self.abstimes = self.datadict['Absolute Times (ns)']
        self.spectra = self.datadict['Spectra (counts\s)']
        self.wavelengths = self.datadict['Spectra (counts\s)'].attrs['Wavelengths']
        self.spectratimes = self.datadict['Spectra (counts\s)'].attrs['Spectra Abs. Times (s)']
        # fig, ax = plt.subplots()
        # ax.imshow(self.spectra.T)
        # ax.set_aspect(0.01)
        # plt.plot(self.wavelengths, np.sum(self.spectra, axis=0))
        # plt.show()
        try:
            self.rasterscan = self.datadict['Raster Scan']
        except KeyError:
            print("Problem loading raster scan for " + self.name)
        self.description = self.datadict.attrs['Discription']
        # self.irf = irf
        if channelwidth is None:
            differences = np.diff(np.sort(self.microtimes[:]))
            channelwidth = np.unique(differences)[1]
        self.channelwidth = channelwidth
        self.tmin = tmin
        self.tmax = tmax
        self.measured = None
        self.t = None
        self.ignore = False
        self.bg = False
        self.histogram = None
        self.binnedtrace = None

    def makehistogram(self):
        """Put the arrival times into a histogram"""

        self.histogram = Histogram(self)

    def binints(self, binsize):
        """Bin the absolute times into a trace using binsize"""

        self.binnedtrace = Trace(self, binsize)


class Trace:
    """Binned intensity trace

    Parameters
    ----------
    particle: Particle
        The Particle which creates the Trace.

    binsize: float
        Size of time bin in ms.
    """

    def __init__(self, particle, binsize):

        self.particle = particle
        self.binsize = binsize
        data = self.particle.abstimes[:]

        binsize = binsize * 1000000  # Convert ms to ns
        endbin = np.int(np.max(data) / binsize)

        binned = np.zeros(endbin)
        for step in range(endbin):
            binned[step] = np.size(data[((step + 1) * binsize > data) * (data > step * binsize)])

        # binned *= (1000 / 100)
        self.intdata = binned


class Histogram:

    def __init__(self, particle):

        self.particle = particle
        tmin = min(self.particle.tmin, self.particle.microtimes[:].min())
        tmax = max(self.particle.tmax, self.particle.microtimes[:].max())
        window = tmax - tmin
        numpoints = int(window // self.particle.channelwidth)

        t = np.linspace(0, window, numpoints)
        # particle.microtimes -= particle.microtimes.min()

        self.decay, self.t = np.histogram(self.particle.microtimes[:], bins=t)


class RasterScan:
    pass


class Spectra:
    pass


class Levels:

    def __init__(self):

        pass  # Change points code called here?
