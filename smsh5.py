"""Module for handling SMS data from HDF5 files

Bertus van Heerden
University of Pretoria
2018
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt


class Particle:

    def __init__(self, name, file, number, irf, tmin, tmax, channelwidth=None):

        self.name = name
        self.file = file
        self.datadict = file[self.name]
        self.microtimes = self.datadict['Micro Times (s)'][:]
        self.abstimes = self.datadict['Absolute Times (ns)'][:]
        self.spectra = self.datadict['Spectra (counts\s)'][:, :]
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
        self.irf = irf
        if channelwidth is None:
            differences = np.diff(np.sort(self.microtimes))
            channelwidth = np.unique(differences)[1]
        self.channelwidth = channelwidth
        self.tmin = tmin
        self.tmax = tmax
        self.measured = None
        self.t = None
        self.ignore = False
        self.bg = False

    def makehistogram(self):
        """Put the arrival times into a histogram"""

        self.tmin = min(self.tmin, self.microtimes.min())
        self.tmax = max(self.tmax, self.microtimes.max())
        window = self.tmax - self.tmin
        numpoints = int(window // self.channelwidth)

        t = np.linspace(0, window, numpoints)
        self.microtimes -= self.microtimes.min()

        self.measured, self.t = np.histogram(self.microtimes, bins=t)