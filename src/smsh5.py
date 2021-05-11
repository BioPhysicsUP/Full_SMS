"""Module for handling SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2018
"""
from __future__ import annotations
import ast
import os
import re
import traceback
from typing import List, Union, TYPE_CHECKING
from uuid import uuid1

import h5pickle
import numpy as np
from pyqtgraph import ScatterPlotItem, SpotItem

import dbg
import tcspcfit
from change_point import ChangePoints
from generate_sums import CPSums
from grouping import AHCA
# from signals import PassSigFeedback
from my_logger import setup_logger
from processes import ProcessProgFeedback, ProcessProgress, PassSigFeedback
from tcspcfit import FittingParameters

if TYPE_CHECKING:
    from change_point import Level

logger = setup_logger(__name__)




class H5dataset:

    def __init__(self, filename, sig_fb: PassSigFeedback, prog_fb: ProcessProgFeedback):

        # self.sig_fb = sig_fb
        # self.prog_fb = prog_fb

        self.cpa_has_run = False
        self.use_parallel = False
        self.name = filename
        prog_fb.set_status(status="Reading file...")
        self.file = h5pickle.File(self.name, 'r')
        try:
            self.version = self.file.attrs['Version']
        except KeyError:
            self.version = '0.1'

        unsorted_names = list(self.file.keys())
        natural_p_names = [None]*len(unsorted_names)
        natural_key = []
        for name in unsorted_names:
            for seg in re.split('(\d+)', name):
                if seg.isdigit():
                    natural_key.append(int(seg))
        for num, key_num in enumerate(natural_key):
            natural_p_names[key_num-1] = unsorted_names[num]

        self.all_sums = CPSums(n_min=10, n_max=1000, prog_fb=prog_fb)

        self.particles = []
        for num, particlename in enumerate(natural_p_names):
            self.particles.append(Particle(name=particlename, dataset_ind=num, dataset=self))
        self.num_parts = len(self.particles)
        assert self.num_parts == self.file.attrs['# Particles']
        self.channelwidth = None

    def makehistograms(self, remove_zeros=True, startpoint=None, channel=True):
        """Put the arrival times into histograms"""

        for particle in self.particles:
            particle.startpoint = startpoint
            particle.makehistogram(channel=channel)
            particle.makelevelhists(channel=channel)
        if remove_zeros:
            maxim = 0
            for particle in self.particles:
                maxim = max(particle.histogram.decaystart, maxim)
            for particle in self.particles:
                particle.histogram.decay = particle.histogram.decay[maxim:]
                particle.histogram.t = particle.histogram.t[maxim:]

    def bin_all_ints(self, binsize:float,
                     sig_fb: PassSigFeedback = None,
                     prog_fb: ProcessProgFeedback = None):
        """Bin the absolute times into traces using binsize
            binsize is in ms
        """
        if prog_fb:
            proc_tracker = ProcessProgress(prog_fb=prog_fb, num_iterations=len(self.particles))

        # if proc_tracker:
        #     if not proc_tracker.has_num_iterations:
        #         proc_tracker.num_iterations = len(self.particles)
        for particle in self.particles:
            particle.binints(binsize)
            if prog_fb:
                proc_tracker.iterate()
        if prog_fb:
            prog_fb.end()
        # dbg.p('Binning all done', 'H5Dataset')

    def save_particles(self, file_path, selected_nums: List[int]):
        """ Save selected particle to a new or existing HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to existing file, or to file that will be created.
        selected_nums : List[int]
            Particle numbers to be written to HDF5 file.
        """

        add = os.path.exists(file_path)
        if add:
            new_h5file = h5pickle.File(file_path, mode='r+')
            num_existing = new_h5file.attrs.get('# Particles')
        else:
            new_h5file = h5pickle.File(file_path, mode='w')
            num_existing = 0

        for i, selected in enumerate(selected_nums):
            new_h5file.copy(self.file[f'/Particle {selected}'], new_h5file, name=f'/Particle {num_existing+i+1}')

        if add:
            new_h5file.attrs.modify('# Particles', num_existing+len(selected_nums))
        else:
            new_h5file.attrs.create('# Particles', len(selected_nums))
        new_h5file.close()


# TODO: This is in the incorrect file.
class BICPlotData:

    def __init__(self):
        self._scatter_plot_item = None
        self._selected_spot = None

    @property
    def has_plot(self) -> bool:
        if self._scatter_plot_item:
            return True
        else:
            return False

    @property
    def has_selected_spot(self) -> bool:
        if self._selected_spot:
            return True
        else:
            return False

    @property
    def scatter_plot_item(self):
        if self.has_plot:
            return self._scatter_plot_item

    @scatter_plot_item.setter
    def scatter_plot_item(self, scatter_plot_item: ScatterPlotItem):
        assert type(scatter_plot_item) == ScatterPlotItem, "scatter_plot_item not correct type."
        self._scatter_plot_item = scatter_plot_item

    @property
    def selected_spot(self):
        if self.has_plot:
            return self._selected_spot

    @selected_spot.setter
    def selected_spot(self, selected_spot: SpotItem):
        assert type(selected_spot) == SpotItem, "selected_spot is not correct type."
        self._selected_spot = selected_spot

    def clear_scatter_plot_item(self):
        self._scatter_plot_item = None
        self._selected_spot = None


class Particle:
    """
    Class for particle in H5dataset.
    """

    def __init__(self, name: str, dataset_ind: int, dataset: H5dataset, tmin=None, tmax=None,
                 channelwidth=None):  # , number, irf, tmin, tmax, channelwidth=None):
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
        self.uuid = uuid1()
        self.name = name
        self.dataset = dataset
        self.dataset_ind = dataset_ind
        self.datadict = self.dataset.file[self.name]
        self.microtimes = self.datadict['Micro Times (s)']
        self.abstimes = self.datadict['Absolute Times (ns)']
        self.num_photons = len(self.abstimes)
        self.cpts = ChangePoints(self)  # Added by Josh: creates an object for Change Point Analysis (cpa)
        self.ahca = AHCA(self)  # Added by Josh: creates an object for Agglomerative Hierarchical Clustering Algorithm
        self.avg_int_weighted = None
        self.int_std_weighted = None

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

        self.startpoint = None
        self.level_selected = None
        self.using_group_levels = False

    @property
    def has_spectra(self) -> bool:
        return self.spectra._has_spectra

    @property
    def has_levels(self):
        return self.cpts.has_levels

    @property
    def has_groups(self):
        return self.ahca.has_groups

    @property
    def groups(self):
        if self.has_groups:
            return self.ahca.selected_step.groups

    @property
    def num_groups(self):
        if self.has_groups:
            return self.ahca.selected_step.num_groups

    @property
    def groups_bounds(self):
        return self.ahca.selected_step.calc_int_bounds()

    @property
    def groups_ints(self):
        return self.ahca.selected_step.group_ints

    @property
    def grouping_bics(self):
        return self.ahca.bics

    @property
    def grouping_selected_ind(self):
        return self.ahca.selected_step_ind

    @property
    def best_grouping_ind(self):
        return self.ahca.best_step_ind

    @grouping_selected_ind.setter
    def grouping_selected_ind(self, ind: int):
        self.ahca.selected_step_ind = ind

    @property
    def grouping_num_groups(self):
        return self.ahca.steps_num_groups

    def reset_grouping_ind(self):
        self.ahca.reset_selected_step()

    @property
    def levels(self):
        if self.using_group_levels:
            return self.ahca.selected_step.group_levels
        else:
            return self.cpts.levels

    @property
    def num_levels(self):
        if self.using_group_levels:
            return self.ahca.selected_step.group_num_levels
        else:
            return self.cpts.num_levels

    @property
    def dwell_time(self):
        return (self.abstimes[-1] - self.abstimes[0]) / 1E9

    @property
    def level_ints(self):
        if self.using_group_levels:
            return self.ahca.selected_step.group_level_ints
        else:
            return self.cpts.level_ints

    @property
    def level_dwelltimes(self):
        if self.using_group_levels:
            return self.ahca.selected_step.group_level_dwelltimes
        else:
            return self.cpts.level_dwelltimes

    @property
    def has_burst(self) -> bool:
        return self.cpts.has_burst

    @property
    def burst_levels(self) -> np.ndarray:
        return self.cpts.burst_levels

    @has_burst.setter
    def has_burst(self, value: bool):
        self.cpts.has_burst = value

    @burst_levels.setter
    def burst_levels(self, value: np.ndarray):
        self.cpts.burst_levels = value

    @property
    def numexp(self):
        return self.histogram.numexp

    def levels2data(self, use_grouped: bool = None) -> [np.ndarray, np.ndarray]:
        """
        Uses the Particle objects' levels to generate two arrays for
        plotting the levels.
        Parameters
        ----------
        use_grouped
        plot_type: str, {'line', 'step'}

        Returns
        -------
        [np.ndarray, np.ndarray]
        """
        assert self.has_levels, 'ChangePointAnalysis:\tNo levels to convert to data.'
        levels = self.levels
        if use_grouped is not None:
            if not use_grouped:
                levels = self.cpts.levels
            else:
                levels = self.ahca.selected_step.group_levels

        num_levels = len(levels)
        levels_data = np.empty(shape=num_levels * 2)
        times = np.empty(shape=num_levels * 2)
        accum_time = 0
        for num, level in enumerate(levels):
            times[num * 2] = accum_time
            accum_time += level.dwell_time_s
            times[num * 2 + 1] = accum_time
            levels_data[num * 2] = level.int_p_s
            levels_data[num * 2 + 1] = level.int_p_s

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
        times = np.array(level.times_ns) / 1E9
        levels_data = np.array([level.int_p_s, level.int_p_s])

        return levels_data, times

    def current_group2data(self, num: int) -> [np.ndarray, np.ndarray]:
        assert self.has_groups, 'ChangePointAnalysis:\tNo groups to convert to data.'

        group = self.groups[num]
        times = np.array([self.abstimes[0], self.abstimes[-1]]) /1E9
        group_int = np.array([group.int_p_s, group.int_p_s])
        return group_int, times

    def makehistogram(self, channel=True):
        """Put the arrival times into a histogram"""

        self.histogram = Histogram(self, start_point=self.startpoint, channel=channel)
        # print(np.max(self.histogram.decay))

    def makelevelhists(self, channel: bool = True,
                       force_cpts_levels: bool = False,
                       force_group_levels: bool = False):
        """Make level histograms"""

        if self.has_levels:
            if force_cpts_levels or force_group_levels:
                levels = list()
                if force_cpts_levels:
                    levels.extend(self.cpts.levels)
                if force_group_levels:
                    levels.extend(self.ahca.selected_step.group_levels)
            else:
                levels = self.levels

            for level in levels:
                level.histogram = Histogram(self, level, self.startpoint, channel=channel)

    def makegrouphists(self, channel=True):

        if self.has_groups:
            for group in self.groups:
                group.histogram = Histogram(self, group.lvls_inds, self.startpoint, channel=channel)

    def binints(self, binsize):
        """Bin the absolute times into a trace using binsize"""

        self.bin_size = binsize
        self.binnedtrace = Trace(self, self.bin_size)

    # def fit_part_and_levels(self, channelwidth, start, end, fit_param: FittingParameters):
    #     if not self.histogram.fit(fit_param.numexp, fit_param.tau, fit_param.amp,
    #                               fit_param.shift / channelwidth, fit_param.decaybg, fit_param.irfbg,
    #                               start, end, fit_param.addopt, fit_param.irf, fit_param.shiftfix):
    #         pass  # fit unsuccessful
    #     self.numexp = fit_param.numexp
    #     # progress_sig.emit()
    #     if not self.has_levels:
    #         return
    #     levels = self.cpts.levels
    #     if self.has_groups:
    #         levels.extend(self.ahca.selected_step.group_levels)
    #         for group in self.ahca.selected_step.groups:
    #             if group.hist is None:
    #                 group.hist = Histogram(particle=self, level=group.lvls_inds,
    #                                        startpoint=self.startpoint)
    #     for level in levels:
    #         if not hasattr(level, 'histogram'):
    #             level.histogram = Histogram()
    #         try:
    #             if not level.histogram.fit(fit_param.numexp, fit_param.tau, fit_param.amp,
    #                                        fit_param.shift / channelwidth, fit_param.decaybg,
    #                                        fit_param.irfbg,
    #                                        start, end, fit_param.addopt,
    #                                        fit_param.irf, fit_param.shiftfix):
    #                 pass  # fit unsuccessful
    #         except AttributeError:
    #             print("No decay")


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
        self.binsize = binsize
        data = particle.abstimes[:]

        binsize_ns = binsize * 1E6  # Convert ms to ns
        endbin = np.int(np.max(data) / binsize_ns)

        binned = np.zeros(endbin + 1, dtype=np.int)
        for step in range(endbin):
            binned[step+1] = np.size(data[((step+1)*binsize_ns > data)*(data > step*binsize_ns)])
            if step == 0:
                binned[step] = binned[step + 1]

        # binned *= (1000 / 100)
        self.intdata = binned
        self.inttimes = np.array(
            range(0, binsize + (endbin * binsize), binsize))


class Histogram:

    def __init__(self, particle: Particle,
                 level: Union[Level, List[int]] = None,
                 start_point: float = None,
                 channel: bool = True,
                 trim_start: bool = False):
        no_sort = False
        self.particle = particle
        self.level = level
        if level is None:
            self.microtimes = self.particle.microtimes[:]
        elif type(level) is list:
            if not self.particle.has_groups:
                logger.error("Multiple levels provided, but has no groups")
                raise RuntimeError("Multiple levels provided, but has no groups")
            self.microtimes = np.array([])
            for ind in level:
                self.microtimes = np.append(self.microtimes, self.particle.cpts.levels[
                    ind].microtimes)
        else:
            self.microtimes = self.level.microtimes[:]

        if self.microtimes.size == 0:
            self.decay = np.empty(1)
            self.t = np.empty(1)
        else:
            tmin = min(self.particle.tmin, self.microtimes.min())
            tmax = max(self.particle.tmax, self.microtimes.max())
            if start_point is None:
                pass
            else:
                if channel:
                    start_point = int(start_point)
                    t = np.arange(tmin, tmax, self.particle.channelwidth)
                    tmin = t[start_point]
                    no_sort = True
                else:
                    tmin = start_point
                    tmax = max(self.particle.tmax, self.microtimes.max())

            sorted_micro = np.sort(self.microtimes)
            if not no_sort and trim_start:
                tmin = sorted_micro[np.searchsorted(sorted_micro, tmin)]  # Make sure bins align with TCSPC bins
            tmax = sorted_micro[np.searchsorted(sorted_micro, tmax) - 1]  # - 1  # Fix if max is end

            window = tmax-tmin
            numpoints = int(window//self.particle.channelwidth)

            t = np.arange(tmin, tmax, self.particle.channelwidth)

            self.decay, self.t = np.histogram(self.microtimes, bins=t)
            self.t = self.t[:-1]  # Remove last value so the arrays are the same size
            where_neg = np.where(self.t <= 0)
            self.t = np.delete(self.t, where_neg)
            self.decay = np.delete(self.decay, where_neg)

            assert len(self.t) == len(self.decay), "Time series must be same length as decay " \
                                                   "histogram"
            if start_point is None and trim_start:
                try:
                    self.decaystart = np.nonzero(self.decay)[0][0]
                except IndexError:  # Happens when there is a level with no photons
                    pass
                else:
                    if level is not None:
                        self.decay, self.t = start_at_value(self.decay, self.t, neg_t=False, decaystart=self.decaystart)
            else:
                self.decaystart = 0

            try:
                self.t -= self.t.min()
            except ValueError:
                dbg.p(f"Histogram object of {self.particle.name} does not have a valid self.t attribute", "Histogram")

        # print(f"{particle.name}: tmin={tmin}, tmax={tmax}")
        self.convd = None
        self.convd_t = None
        self.fitted = False

        self.fit_decay = None
        self.convd = None
        self.convd_t = None
        self.tau = None
        self.amp = None
        self.shift = None
        self.bg = None
        self.irfbg = None
        self.avtau = None
        self.numexp = None

    @property
    def t(self):
        return self._t.copy()

    @t.setter
    def t(self, value):
        self._t = value

    def fit(self, numexp, tauparam, ampparam, shift, decaybg, irfbg, start, end, addopt, irf, shiftfix):
        if addopt is not None:
            addopt = ast.literal_eval(addopt)

        if shiftfix:
            shift = [shift, 1]

        if start is None:
            start = 0

        self.numexp = numexp

        # TODO: debug option that would keep the fit object (not done normally to conserve memory)
        try:
            if numexp == 1:
                fit = tcspcfit.OneExp(irf, self.decay, self.t, self.particle.channelwidth,
                                      tauparam, None, shift, decaybg, irfbg, start, end, addopt)
            elif numexp == 2:
                fit = tcspcfit.TwoExp(irf, self.decay, self.t, self.particle.channelwidth, tauparam,
                                      ampparam, shift, decaybg, irfbg, start, end, addopt)
            elif numexp == 3:
                fit = tcspcfit.ThreeExp(irf, self.decay, self.t, self.particle.channelwidth,
                                        tauparam, ampparam, shift, decaybg,
                                        irfbg, start, end, addopt)
        except Exception as e:
            logger.error(e)
            # print(traceback.format_exc().split('\n')[-2])
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
            self.chisq = fit.chisq
            self.fitted = True
            if numexp == 1:
                self.avtau = self.tau
            else:
                self.avtau = sum(self.tau * self.amp) / self.amp.sum()

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


class ParticleAllHists:
    def __init__(self, particle: Particle):
        self.part_uuid = particle.uuid
        self.numexp = None
        self.part_hist = particle.histogram

        self.has_level_hists = particle.has_levels
        self.level_hists = list()
        if particle.has_levels:
            for level in particle.levels:
                if hasattr(level, 'histogram') and level.histogram is not None:
                    self.level_hists.append(level.histogram)

        self.has_group_hists = particle.has_groups
        self.group_hists = list()
        if particle.has_groups:
            for group in particle.groups:
                if group.histogram is None:
                    group.histogram = Histogram(particle=particle, level=group.lvls_inds,
                                                start_point=particle.startpoint)
                self.group_hists.append(group.histogram)

    def fit_part_and_levels(self, channelwidth, start, end, fit_param: FittingParameters):
        self.numexp = fit_param.numexp
        all_hists = [self.part_hist]
        all_hists.extend(self.level_hists)
        all_hists.extend(self.group_hists)

        for hist in all_hists:
            try:
                if not hist.fit(fit_param.numexp, fit_param.tau, fit_param.amp,
                                fit_param.shift / channelwidth, fit_param.decaybg,
                                fit_param.irfbg, start, end, fit_param.addopt,
                                fit_param.irf, fit_param.shiftfix):
                    pass  # fit unsuccessful
            except AttributeError:
                print("No decay")


class RasterScan:
    def __init__(self, particle):
        self._particle = particle
        if 'Raster Scan' in particle.datadict.keys():
            self.data = particle.datadict['Raster Scan']
            self.integration_time = self.data.attrs["Int. Time (ms/um)"]
            self.pixel_per_line = self.data.attrs["Pixels per Line"]
            self.range = self.data.attrs["Range (um)"]
            self.x_start = self.data.attrs["XStart (um)"]
            self.y_start = self.data.attrs["YStart (um)"]


class Spectra:
    def __init__(self, particle):
        self.particle = particle
        self._has_spectra = False
        if 'Spectra (counts\\s)' in particle.datadict.keys():
            self._has_spectra = True
            self.data = particle.datadict['Spectra (counts\\s)']
            self.wavelengths = self.data.attrs['Wavelengths']
            self.series_times = self.data.attrs['Spectra Abs. Times (s)']


def start_at_value(decay, t, neg_t=True, decaystart=None):
    if decaystart is None:
        decaystart = np.nonzero(decay)[0][0]
    if neg_t:
        t -= t[decaystart]
    t = t[decaystart:]
    decay = decay[decaystart:]
    return decay, t
