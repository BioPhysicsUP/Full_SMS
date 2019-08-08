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
from generate_sums import CPSums
from PyQt5.QtCore import pyqtSignal
import dbg
from joblib import Parallel, delayed


class H5dataset:

    def __init__(self, filename, progress_sig: pyqtSignal = None,
                 auto_prog_sig: pyqtSignal = None):

        self.use_parallel = True
        self.progress_sig = progress_sig
        self.auto_prog_sig = auto_prog_sig
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

        self.all_sums = CPSums(n_min=10, n_max=1000, auto_prog_sig=self.auto_prog_sig)
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

    def binints(self, binsize, progress_sig=None):
        """Bin the absolute times into traces using binsize
            binsize is in ms
        """
        if progress_sig is not None:
            self.progress_sig = progress_sig

        if self.use_parallel:
            self.bintsize_parallel = binsize
            if hasattr(self, 'progress_sig'):
                self.prog_sig_parallel = progress_sig
            Parallel(n_jobs=-1, backend='threading')(
                delayed(self.run_binints_parallel)(particle) for particle in self.particles
            )
            del self.bintsize_parallel, self.prog_sig_parallel
        else:
            for particle in self.particles:
                particle.binints(binsize)
                if hasattr(self, 'progress_sig'):
                    self.progress_sig.emit()  # Increments the progress bar on the MainWindow GUI
        dbg.p('Binning all done', 'H5Dataset')

    def run_binints_parallel(self, particle):
        particle.binints(self.bintsize_parallel)
        if hasattr(self, 'self.prog_sig_parallel'):
            self.prog_sig_parallel.emit()

class Particle:

    def __init__(self, name, dataset, tmin=None, tmax=None, channelwidth=None):#, number, irf, tmin, tmax, channelwidth=None):

        self.name = name
        self.dataset = dataset
        self.datadict = self.dataset.file[self.name]
        self.microtimes = self.datadict['Micro Times (s)']
        self.abstimes = self.datadict['Absolute Times (ns)']
        self.num_photons = len(self.abstimes)
        self.cpts = ChangePoints(self)  # Added by Josh: creates an object for Change Point Analysis (cpa)
        self.cpt_inds = None
        self.num_cpts = None
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
        assert levels is not None and num_levels is not None, \
            "Particle:\tBoth arguments need to be non-None to add level."
        self.levels = levels
        self.num_levels = num_levels
        self.has_levels = True
        
    def levels2data(self) -> [np.ndarray, np.ndarray]:
        assert self.has_levels, 'ChangePointAnalysis:\tNo levels to convert to data.'
        levels_data = np.empty(shape=self.num_levels+1)
        times = np.empty(shape=self.num_levels+1)
        accum_time = 0
        for num, level in enumerate(self.levels):
            times[num] = accum_time
            accum_time += level.dwell_time/1E9
            levels_data[num] = level.int
            if num+1 == self.num_levels:
                levels_data[num+1] = accum_time
                times[num+1] = level.int
                
        return levels_data, times

    def makehistogram(self):
        """Put the arrival times into a histogram"""

        self.histogram = Histogram(self)

    def binints(self, binsize):
        """Bin the absolute times into a trace using binsize"""

        self.bin_size = binsize
        self.binnedtrace = Trace(self, self.bin_size)

    def remove_cpa_results(self):
        self.cpt_inds = None
        self.levels = None
        self.num_cpts = None
        self.num_levels = None
        self.has_levels = False


class Trace:
    """Binned intensity trace

    Parameters
    ----------
    particle: Particle
        The Particle which creates the Trace.

    binsize: float
        Size of time bin in ms.
    """

    def __init__(self, particle, binsize: int):

        # self.particle = particle
        self.binsize = binsize
        data = particle.abstimes[:]

        binsize_ns = binsize * 1E6  # Convert ms to ns
        endbin = np.int(np.max(data) / binsize_ns)

        binned = np.zeros(endbin+1, dtype=np.int)
        for step in range(endbin):
            binned[step+1] = np.size(data[((step+1)*binsize_ns > data)*(data > step*binsize_ns)])
            if step == 0:
                binned[step] = binned[step+1]

        # binned *= (1000 / 100)
        self.intdata = binned
        self.inttimes = np.array(range(0, binsize+(endbin*binsize), binsize))

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
