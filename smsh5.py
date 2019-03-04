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
        """Put the arrival times into a histogram"""

        for particle in self.particles:
            tmin = min(self.tmin, particle.microtimes[:].min())
            tmax = max(self.tmax, particle.microtimes[:].max())
            window = tmax - tmin
            numpoints = int(window // self.channelwidth)

            t = np.linspace(0, window, numpoints)
            # particle.microtimes -= particle.microtimes.min()

            particle.measured, particle.t = np.histogram(particle.microtimes[:], bins=t)

    def binints(self, binsize):

        for particle in self.particles:
            data = particle.abstimes[:]

            binsize = binsize * 1000000
            endbin = np.int(np.max(data) / binsize)

            binned = np.zeros(endbin)
            for step in range(endbin):
                binned[step] = np.size(data[((step + 1) * binsize > data) * (data > step * binsize)])

            binned *= (1000 / 100)
            particle.binned = binned


class Particle:

    def __init__(self, name, dataset):#, number, irf, tmin, tmax, channelwidth=None):

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
        # if channelwidth is None:
        #     differences = np.diff(np.sort(self.microtimes[:]))
        #     channelwidth = np.unique(differences)[1]
        # self.channelwidth = channelwidth
        # self.tmin = tmin
        # self.tmax = tmax
        self.measured = None
        self.t = None
        self.ignore = False
        self.bg = False


class Trace:
    pass


class RasterScan:
    pass


class Spectra:
    pass


class Histogram:
    pass


class Levels:
    pass
