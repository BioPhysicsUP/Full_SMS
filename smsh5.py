"""Module for handling SMS data from HDF5 files

Bertus van Heerden
University of Pretoria
2018
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from change_point import ChangePoints
import re


class H5dataset:

    def __init__(self, filename, progress_sig=None):

        if progress_sig is not None:
            self.progress_sig = progress_sig
        # self.main_signals.progress.connect()
        self.name = filename
        self.file = h5py.File(self.name, 'r')
        try:
            self.version = self.file.attrs['Version']
        except KeyError:
            self.version = '0.1'

        unsorted_names = list(self.file.keys())
        natural_p_names = [None] * len(unsorted_names)
        natural_key = []
        for name in unsorted_names:
            for seg in re.split('(\d+)', name):
                if seg.isdigit():
                    natural_key.append(int(seg))
        for num, key_num in enumerate(natural_key):
            natural_p_names[key_num-1] = unsorted_names[num]

        self.particles = []
        for particlename in natural_p_names:
            self.particles.append(Particle(particlename, self))
        self.numpart = len(self.particles)
        assert self.numpart == self.file.attrs['# Particles']
        self.channelwidth = None

    def makehistograms(self):
        """Put the arrival times into histograms"""

        for particle in self.particles:
            particle.makehistogram()
            if hasattr(self, 'progress_sig'):
                self.progress_sig.emit()  # Increments the progress bar on the MainWindow GUI

    def binints(self, binsize):
        """Bin the absolute times into traces using binsize
            binsize is in ms
        """

        for particle in self.particles:
            particle.binints(binsize)
            if hasattr(self, 'progress_sig'):
                self.progress_sig.emit()  # Increments the progress bar on the MainWindow GUI
        print("done binning")


class Particle:

    def __init__(self, name, dataset, tmin=None, tmax=None, channelwidth=None):#, number, irf, tmin, tmax, channelwidth=None):

        self.name = name
        self.dataset = dataset
        self.datadict = self.dataset.file[self.name]
        self.microtimes = self.datadict['Micro Times (s)']
        self.abstimes = self.datadict['Absolute Times (ns)']
        self.num_photons = len(self.abstimes)
        self.cpts = ChangePoints(self)  # Added by Josh: creates an object for Change Point Analysis (cpa)
        self.has_levels = False
        self.levels = None
        self.num_levels = None

        self.spectra = Spectra(self)
        self.rasterscan = RasterScan(self)
        self.description = self.datadict.attrs['Discription']
        # self.irf = irf
        if channelwidth is None:
            differences = np.diff(np.sort(self.microtimes[:]))
            channelwidth = np.unique(differences)[1]
        self.channelwidth = channelwidth
        if tmin is None:
            self.tmin = 0
        else:
            self.tmin = tmin
        if tmax is None:
            self.tmax = 25
        else:
            self.tmax = tmax
        self.measured = None
        self.t = None
        self.ignore = False
        self.bg = False
        self.histogram = None
        self.binnedtrace = None
        self.bin_size = None

    def get_levels(self):
        assert self.cpts.cpa_has_run, "Particle:\tChange point analysis needs to run before levels can be defined."
        self.add_levels(self.cpts.get_levels())

    def add_levels(self, levels=None, num_levels=None):
        assert levels is not None and num_levels is not None, "Particle:\tBoth arguments need to be non-None to add level."
        self.levels = levels
        self.num_levels = num_levels
        self.has_levels = True

    def makehistogram(self):
        """Put the arrival times into a histogram"""

        self.histogram = Histogram(self)

    def binints(self, binsize):
        """Bin the absolute times into a trace using binsize"""

        self.bin_size = binsize
        self.binnedtrace = Trace(self, self.bin_size)


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
        self.t = self.t[:-1]  # Remove last value so the arrays are the same size


class RasterScan:

    def __init__(self, particle):

        self.particle = particle
        try:
            self.image = self.particle.datadict['Raster Scan']
        except KeyError:
            print("Problem loading raster scan for " + self.name)
            self.image = None


class Spectra:

    def __init__(self, particle):

        self.particle = particle
        self.spectra = self.particle.datadict['Spectra (counts\s)']
        self.wavelengths = self.spectra.attrs['Wavelengths']
        self.spectratimes = self.spectra.attrs['Spectra Abs. Times (s)']


# Level class has been defined in change_point.py
# class Levels:
#
#     def __init__(self):
#
#         pass  # Change points code called here?
