"""Module for handling SMS data from HDF5 files

Bertus van Heerden
University of Pretoria
"""
import h5py
import numpy as np


class Particle:

    def __init__(self, file, number, irf, tmin, tmax, channelwidth=None):

        self.file = file
        self.datadict = file['Particle %d' % number]
        self.microtimes = self.datadict['Micro Times (s)'][:]
        self.abstimes = self.datadict['Absolute Times (ns)'][:]
        self.irf = irf
        if channelwidth is None:
            differences = np.diff(np.sort(self.microtimes))
            channelwidth = np.unique(differences)[1]
        self.channelwidth = channelwidth
        self.tmin = tmin
        self.tmax = tmax
        self.measured = None
        self.t = None

    def makehistogram(self):
        """Put the arrival times into a histogram"""

        self.tmin = min(self.tmin, self.microtimes.min())
        self.tmax = max(self.tmax, self.microtimes.max())
        window = self.tmax - self.tmin
        numpoints = int(window // self.channelwidth)

        t = np.linspace(0, window, numpoints)
        self.microtimes -= self.microtimes.min()

        self.measured, self.t = np.histogram(self.microtimes, bins=t)