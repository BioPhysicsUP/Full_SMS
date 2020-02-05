"""Module for handling SMS data from HDF5 files

Bertus van Heerden
University of Pretoria
2018
"""
import traceback
import h5py
import numpy as np
import tcspcfit
# from main.MainWindow import start_at_nonzero
import dbg
from matplotlib import pyplot as plt
from ChangePoint import ChangePoints
import re
from generate_sums import CPSums
from PyQt5.QtCore import pyqtSignal
import dbg
from ClusteringGrouping import AHCA
from joblib import Parallel, delayed


class H5dataset:

    def __init__(self, filename, progress_sig: pyqtSignal = None,
                 auto_prog_sig: pyqtSignal = None):

        self.cpa_has_run = False
        self.use_parallel = False
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
            natural_p_names[key_num - 1] = unsorted_names[num]

        self.all_sums = CPSums(n_min=10, n_max=1000,
                               auto_prog_sig=self.auto_prog_sig)
        self.particles = []
        for particlename in natural_p_names:
            self.particles.append(Particle(particlename, self))
        self.numpart = len(self.particles)
        assert self.numpart == self.file.attrs['# Particles']
        self.channelwidth = None

    def makehistograms(self, progress=True):
        """Put the arrival times into histograms"""

        for particle in self.particles:
            particle.makehistogram()
            particle.makelevelhists()
            if progress and hasattr(self, 'progress_sig'):  # TODO: this is a hack and should be make cleaner
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
                delayed(self.run_binints_parallel)(particle) for particle in
                self.particles
            )
            del self.bintsize_parallel, self.prog_sig_parallel
        else:
            for particle in self.particles:
                particle.binints(binsize)
                if hasattr(self, 'progress_sig'):
                    self.progress_sig.emit()  # Increments the progress bar
                    # on the MainWindow GUI
        dbg.p('Binning all done', 'H5Dataset')

    def run_binints_parallel(self, particle):
        particle.binints(self.bintsize_parallel)
        if hasattr(self, 'self.prog_sig_parallel'):
            self.prog_sig_parallel.emit()


class Particle:
    """
    Class for particle in H5dataset.
    """

    def __init__(self, name, dataset, tmin=None, tmax=None,
                 channelwidth=None):  # , number, irf, tmin, tmax,
        # channelwidth=None):
        """
        Creates an instance of Particle

        Parameters
        ----------
        name: str
            The name of the particle
        dataset: H5dataset
            The instance of the dataset to which this particle belongs
        tmin: int, Optional
            TODO
        tmax: int, Optional
            TODO
        channelwidth: TODO
        """
        self.name = name
        self.dataset = dataset
        self.datadict = self.dataset.file[self.name]
        self.microtimes = self.datadict['Micro Times (s)']
        self.abstimes = self.datadict['Absolute Times (ns)']
        self.num_photons = len(self.abstimes)
        self.cpts = ChangePoints(
            self)  # Added by Josh: creates an object for Change Point
        # Analysis (cpa)
        self.ahca = AHCA(
            self)  # Added by Josh: creates an object for Agglomerative
        # Hierarchical Clustering Algorithm
        self.cpt_inds = None
        self.num_cpts = None
        self.has_levels = False
        self.levels = None
        self.num_levels = None
        self.avg_int_weighted = None
        self.int_std_weighted = None
        self.burst_std_factor = 1.5

        self.spectra = Spectra(self)
        self.rasterscan = RasterScan(self)
        self.description = self.datadict.attrs['Discription']
        self.irf = None
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
        assert self.cpts.cpa_has_run, "Particle:\tChange point analysis " \
                                      "needs to run before levels can be " \
                                      "defined."
        self.add_levels(self.cpts.get_levels())

    def add_levels(self, levels=None, num_levels=None):
        assert levels is not None and num_levels is not None,\
            "Particle:\tBoth arguments need to be non-None to add level."
        self.levels = levels
        self.num_levels = num_levels
        self.has_levels = True

    def levels2data(self, plot_type: str = 'line') -> [np.ndarray, np.ndarray]:
        """
        Uses the Particle objects' levels to generate two arrays for
        plotting the levels.
        Parameters
        ----------
        plot_type: str, {'line', 'step'}

        Returns
        -------
        [np.ndarray, np.ndarray]
        """
        assert self.has_levels, 'ChangePointAnalysis:\tNo levels to convert ' \
                                'to data.'

        # ############## Old, for Matplotlib ##############
        # levels_data = np.empty(shape=self.num_levels+1)
        # times = np.empty(shape=self.num_levels+1)
        # accum_time = 0
        # for num, level in enumerate(self.levels):
        #     times[num] = accum_time
        #     accum_time += level.dwell_time/1E9
        #     levels_data[num] = level.int
        #     if num+1 == self.num_levels:
        #         levels_data[num+1] = accum_time
        #         times[num+1] = level.int

        levels_data = np.empty(shape=self.num_levels * 2)
        times = np.empty(shape=self.num_levels * 2)
        accum_time = 0
        for num, level in enumerate(self.levels):
            times[num * 2] = accum_time
            accum_time += level.dwell_time / 1E9
            times[num * 2 + 1] = accum_time
            levels_data[num * 2] = level.int
            levels_data[num * 2 + 1] = level.int

        return levels_data, times

    def current2data(self, num, plot_type: str = 'line') -> [np.ndarray, np.ndarray]:
        """
        Uses the Particle objects' levels to generate two arrays for plotting level num.
        Parameters
        ----------
        plot_type: str, {'line', 'step'}

        Returns
        -------
        [np.ndarray, np.ndarray]
        """
        # TODO: Cleanup this function anc the one above it
        assert self.has_levels, 'ChangePointAnalysis:\tNo levels to convert to data.'

        # ############## Old, for Matplotlib ##############
        # levels_data = np.empty(shape=self.num_levels+1)
        # times = np.empty(shape=self.num_levels+1)
        # accum_time = 0
        # for num, level in enumerate(self.levels):
        #     times[num] = accum_time
        #     accum_time += level.dwell_time/1E9
        #     levels_data[num] = level.int
        #     if num+1 == self.num_levels:
        #         levels_data[num+1] = accum_time
        #         times[num+1] = level.int

        level = self.levels[num]
        times = np.array(level.times) / 1E9
        levels_data = np.array([level.int, level.int])

        return levels_data, times

    def makelevelhists(self):
        """Make level histograms"""

        if not self.has_levels:
            print('No levels.')
            return

        for level in self.levels:
            level.histogram = Histogram(self, level)

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
        self.num_levels = None
        self.has_levels = False
        self.has_burst = False
        self.burst_levels = np.array([])


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

        binned = np.zeros(endbin + 1, dtype=np.int)
        for step in range(endbin):
            binned[step + 1] = np.size(data[
                                           ((step + 1) * binsize_ns > data) * (
                                                       data > step *
                                                       binsize_ns)])
            if step == 0:
                binned[step] = binned[step + 1]

        # binned *= (1000 / 100)
        self.intdata = binned
        self.inttimes = np.array(
            range(0, binsize + (endbin * binsize), binsize))


class Histogram:

    def __init__(self, particle, level=None):
        self.particle = particle
        self.level = level
        if level is None:
            self.microtimes = self.particle.microtimes[:]
        else:
            self.microtimes = self.level.microtimes[:]

        if self.microtimes.size == 0:
            self.decay = np.empty(1)
            self.t = np.empty(1)
        else:
            print(self.microtimes)
            tmin = min(self.particle.tmin, self.microtimes.min())
            tmax = max(self.particle.tmax, self.microtimes.max())
            window = tmax-tmin
            numpoints = int(window//self.particle.channelwidth)

            t = np.linspace(0, window, numpoints)

            self.decay, self.t = np.histogram(self.microtimes, bins=t)
            self.t = self.t[:-1]  # Remove last value so the arrays are the same size
            self.decay = self.decay[self.t > 0]
            self.t = self.t[self.t > 0]

        self.convd = None
        self.convd_t = None
        self.fitted = False

    def fit(self, numexp, tauparam, ampparam, shift, decaybg, irfbg, start, end, addopt, irf):

        # Todo: This should probably happen somewhere else:
        try:
            self.decay, self.t = start_at_nonzero(self.decay, self.t, neg_t=False)
        except IndexError:  # Empty decay
            return False
        irf, irft = start_at_nonzero(irf, self.t, neg_t=False)

        # TODO: debug option that would keep the fit object (not done normally to conserve memory)
        try:
            if numexp == 1:
                fit = tcspcfit.OneExp(irf, self.decay, self.t, self.particle.channelwidth, tauparam, None, shift,
                                      decaybg, irfbg, start, end)
            elif numexp == 2:
                fit = tcspcfit.TwoExp(irf, self.decay, self.t, self.particle.channelwidth, tauparam, ampparam, shift,
                                      decaybg, irfbg, start, end)
            elif numexp == 3:
                fit = tcspcfit.ThreeExp(irf, self.decay, self.t, self.particle.channelwidth, tauparam, ampparam, shift,
                                        decaybg, irfbg, start, end)
        except:
            dbg.p('Error while fitting lifetime:', debug_from='smsh5')
            traceback.print_exc()
            return False

        else:
            self.fit_decay = fit.measured
            self.convd = fit.convd
            self.convd_t = fit.t
            self.tau = fit.tau
            self.amp = fit.amp
            self.shift = fit.shift
            self.bg = fit.bg
            self.irfbg = fit.irfbg
            self.fitted = True

        return True

    def levelhist(self, level):
        levelobj = self.particle.levels[level]
        tmin = levelobj.microtimes[:].min()
        tmax = levelobj.microtimes[:].max()
        window = tmax-tmin
        numpoints = int(window//self.particle.channelwidth)
        t = np.linspace(0, window, numpoints)

        decay, t = np.histogram(levelobj.microtimes[:], bins=t)
        t = t[:-1]  # Remove last value so the arrays are the same size
        return decay, t


class RasterScan:

    def __init__(self, particle):

        self.particle = particle
        try:
            self.image = self.particle.datadict['Raster Scan']
        except KeyError:
            print("Problem loading raster scan for " + self.name)
            print("Problem loading raster scan for " + self.particle.name)
            self.image = None


class Spectra:

    def __init__(self, particle):
        self.particle = particle
        self.spectra = self.particle.datadict['Spectra (counts\s)']
        self.wavelengths = self.spectra.attrs['Wavelengths']
        self.spectratimes = self.spectra.attrs['Spectra Abs. Times (s)']

# Level class has been defined in ChangePoint.py
# class Levels:
#
#     def __init__(self):
#
#         pass  # Change points code called here?


def start_at_nonzero(decay, t, neg_t=True):
    decaystart = np.nonzero(decay)[0][0]
    if neg_t:
        t -= t[decaystart]
    t = t[decaystart:]
    decay = decay[decaystart:]
    return decay, t
