"""Module for handling SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2018
"""
from __future__ import annotations

__docformat__ = 'NumPy'

import ast
import os
import re
import traceback
from typing import List, Union, TYPE_CHECKING, Tuple
from uuid import uuid1

import h5pickle
import h5py
import numpy as np
from pyqtgraph import ScatterPlotItem, SpotItem

import dbg
import tcspcfit
from change_point import ChangePoints
from generate_sums import CPSums
from grouping import AHCA
from my_logger import setup_logger
from processes import ProcessProgFeedback, ProcessProgress, PassSigFeedback
from tcspcfit import FittingParameters
import smsh5_file_reader as h5_fr

if TYPE_CHECKING:
    from change_point import Level

logger = setup_logger(__name__)


class H5dataset:

    def __init__(self, filename, sig_fb: PassSigFeedback, prog_fb: ProcessProgFeedback):
        self.cpa_has_run = False
        self.use_parallel = False
        self.name = filename
        prog_fb.set_status(status="Reading file...")
        self.file = h5pickle.File(self.name, 'r')
        self.file_version = h5_fr.file_version(dataset=self)

        all_keys = self.file.keys()
        part_keys = [part_key for part_key in all_keys if 'Particle ' in part_key]

        natural_p_names = [None]*len(part_keys)
        natural_key = []
        for name in part_keys:
            for seg in re.split('(\d+)', name):
                if seg.isdigit():
                    natural_key.append(int(seg))
        for num, key_num in enumerate(natural_key):
            natural_p_names[key_num-1] = part_keys[num]

        self.all_sums = CPSums(n_min=10, n_max=1000, prog_fb=prog_fb)

        self.all_raster_scans = list()
        map_particle_indexes = self.get_all_raster_scans(particle_names=natural_p_names)
        if map_particle_indexes is not None:
            self.has_raster_scans = True
        else:
            self.has_raster_scans = False

        self.particles = []
        for num, particle_name in enumerate(natural_p_names):
            if map_particle_indexes is not None:
                this_raster_scan_index = map_particle_indexes[num]
                this_raster_scan = self.all_raster_scans[map_particle_indexes[num]]
            else:
                this_raster_scan_index = None
                this_raster_scan = None
            self.particles.append(
                Particle(name=particle_name, dataset_ind=num, dataset=self,
                         raster_scan_dataset_index=this_raster_scan_index,
                         raster_scan=this_raster_scan))
        self.num_parts = len(self.particles)
        assert self.num_parts == h5_fr.num_parts(dataset=self)
        self.channelwidth = None
        self.save_selected = None
        self.has_levels = False
        self.has_groups = False
        self.has_lifetimes = False
        self.irf = None
        self.irf_t = None
        self.has_irf = False

    def get_all_raster_scans(self, particle_names: List[str]) -> list:
        raster_scans = list()
        file_keys = self.file.keys()
        for num, particle_name in enumerate(particle_names):
            if particle_name in file_keys:
                particle = h5_fr.particle(particle_num=num+1, dataset=self)
                if h5_fr.has_raster_scan(particle=particle):
                    raster_scans.append((h5_fr.raster_scan(particle=particle), num))

        if len(raster_scans) != 0:
            prev_raster_scan = None
            group_indexes = list()
            map_particle_index = list()
            raster_scan_counter = 0
            for raster_scan, num in raster_scans:
                if raster_scan == prev_raster_scan and num != len(raster_scans) - 1:
                    # Must not add new RasterScan, because same raster scan as previous and not last
                    group_indexes.append(num)
                    raster_scan_num = len(self.all_raster_scans)
                else:  # Must add new RasterScan, because raster scan different from previous, or last
                    if num != 0:  # Not first one
                        if num != len(raster_scans) - 1:  # Not last one
                            # Save previous raster scan with previous group indexes
                            self.all_raster_scans.append(
                                RasterScan(raster_scan_dataset=prev_raster_scan,
                                           particle_indexes=group_indexes,
                                           dataset_index=raster_scan_num))
                            raster_scan_counter += 1
                            raster_scan_num = len(self.all_raster_scans)
                        else:  # Last one
                            # Save this raster scan with updated group indexes
                            if raster_scan == prev_raster_scan:
                                # Last one is part of previous group
                                group_indexes.append(num)
                                raster_scan_num = len(self.all_raster_scans)
                            else:
                                # Last one part of new group
                                self.all_raster_scans.append(
                                    RasterScan(raster_scan_dataset=prev_raster_scan,
                                               particle_indexes=group_indexes,
                                               dataset_index=raster_scan_num))
                                raster_scan_counter += 1
                                group_indexes = [num]
                                raster_scan_num = len(self.all_raster_scans)
                            self.all_raster_scans.append(RasterScan(raster_scan_dataset=raster_scan,
                                                                    particle_indexes=group_indexes,
                                                                    dataset_index=raster_scan_counter))
                            raster_scan_counter += 1
                    else:
                        raster_scan_num = 0
                        if len(raster_scans) == 1:
                            group_indexes = [0]
                            self.all_raster_scans.append(
                                RasterScan(raster_scan_dataset=raster_scan,
                                           particle_indexes=group_indexes,
                                           dataset_index=raster_scan_counter))
                    group_indexes = [num]
                    prev_raster_scan = raster_scan

                map_particle_index.append(raster_scan_num)
            return map_particle_index
        else:
            return None

    def makehistograms(self, remove_zeros=True, startpoint=None, channel=True):
        """Put the arrival times into histograms"""

        for particle in self.particles:
            particle.startpoint = startpoint
            particle.makehistogram(channel=channel, add_roi=True)
            particle.makelevelhists(channel=channel)
        if remove_zeros:
            maxim = 0
            for particle in self.particles:
                try:
                    maxim = max(particle.histogram.decaystart, maxim)
                except AttributeError:
                    maxim = 0
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
            new_h5file.copy(self.file[f'/Particle {selected}'],
                            new_h5file, name=f'/Particle {num_existing+i+1}')

        if add:
            new_h5file.attrs.modify('# Particles', num_existing+len(selected_nums))
        else:
            new_h5file.attrs.create('# Particles', len(selected_nums))
        new_h5file.close()


# TODO: This is in the incorrect file.
# class BICPlotData:
#
#     def __init__(self):
#         self._scatter_plot_item = None
#         self._selected_spot = None
#
#     @property
#     def has_plot(self) -> bool:
#         if self._scatter_plot_item:
#             return True
#         else:
#             return False
#
#     @property
#     def has_selected_spot(self) -> bool:
#         if self._selected_spot:
#             return True
#         else:
#             return False
#
#     @property
#     def scatter_plot_item(self):
#         if self.has_plot:
#             return self._scatter_plot_item
#
#     @scatter_plot_item.setter
#     def scatter_plot_item(self, scatter_plot_item: ScatterPlotItem):
#         assert type(scatter_plot_item) == ScatterPlotItem, "scatter_plot_item not correct type."
#         self._scatter_plot_item = scatter_plot_item
#
#     @property
#     def selected_spot(self):
#         if self.has_plot:
#             return self._selected_spot
#
#     @selected_spot.setter
#     def selected_spot(self, selected_spot: SpotItem):
#         assert type(selected_spot) == SpotItem, "selected_spot is not correct type."
#         self._selected_spot = selected_spot
#
#     def clear_scatter_plot_item(self):
#         self._scatter_plot_item = None
#         self._selected_spot = None


class Particle:
    """
    Class for particle in H5dataset.
    """

    def __init__(self, name: str, dataset_ind: int, dataset: H5dataset,
                 raster_scan_dataset_index: int = None, raster_scan: RasterScan = None,
                 tmin=None, tmax=None, channelwidth=None):
        """
        Creates an instance of Particle

        Parameters
        ----------
        name: str
            The name of the particle
        dataset: H5dataset
            The instance of the dataset to which this particle belongs
        tmin: int, Optional
            TODO: Update docsring
        tmax: int, Optional
        channelwidth:
        """
        self.uuid = uuid1()
        self.name = name
        # self.dataset = dataset
        self.dataset_ind = dataset_ind
        self.file = dataset.file
        self.file_version = h5_fr.file_version(dataset=dataset)
        self.datadict = self.file[self.name]
        self.microtimes = h5_fr.microtimes(particle=self)
        self.abstimes = h5_fr.abstimes(particle=self)
        self.num_photons = len(self.abstimes)
        self.cpts = ChangePoints(self)  # Added by Josh: creates an object for Change Point Analysis (cpa)
        self.ahca = AHCA(self)  # Added by Josh: creates an object for Agglomerative Hierarchical Clustering Algorithm
        self.avg_int_weighted = None
        self.int_std_weighted = None

        self.spectra = Spectra(self)
        self._raster_scan_dataset_index = raster_scan_dataset_index
        self.raster_scan = raster_scan
        self.has_raster_scan = raster_scan is not None
        self.description = h5_fr.description(particle=self)
        self.irf = None
        try:
            if channelwidth is None:
                differences = np.diff(np.sort(self.microtimes[:]))
                channelwidth = np.unique(differences)[1]
        except IndexError as e:
            logger.error(f"channelwidth could not be detemined. Inspect {self.name}.")
            channelwidth = 0.01220703125
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
        self._histogram = None
        self._histogram_roi = None
        self.use_roi_for_histogram = False
        self.binnedtrace = None
        self.bin_size = None
        try:
            self.roi_region = (0, self.abstimes[-1]/1E9)
        except IndexError:
            self.roi_region = (0, 0)

        self.startpoint = None
        self.level_selected = None
        self.using_group_levels = False

        self.has_fit_a_lifetime = False
        self.has_exported = False

    @property
    def histogram(self) -> Histogram:
        if not self.use_roi_for_histogram:
            hist = self._histogram
        else:
            hist = self._histogram_roi
        return hist

    def set_histgram(self, histogram: Histogram) -> None:
        self._histogram = histogram

    def set_histgram_roi(self, histogram: Histogram) -> None:
        self._histogram_roi = histogram

    @property
    def use_roi_for_grouping(self) -> bool:
        return self.ahca.use_roi_for_grouping

    @property
    def grouped_with_roi(self) -> bool:
        return self.ahca.grouped_with_roi

    @property
    def roi_region_photon_inds(self) -> Tuple[int, int]:
        first_photon = 0
        last_photon = self.num_photons - 1
        roi_start = self.roi_region[0]
        roi_end = self.roi_region[1]
        epsilon_t = 0.1
        end_t = self.abstimes[-1]/1E9
        if roi_start >= 0 + epsilon_t/2 or not roi_end - epsilon_t/2 <= end_t <= roi_end + epsilon_t/2:
            times = self.abstimes[:]/1E9
            first_photon = np.argmin(roi_start > times)
            last_photon = np.argmin(roi_end > times)
        return first_photon, last_photon

    @property
    def num_photons_roi(self) -> int:
        first_photon, last_photon = self.roi_region_photon_inds
        return last_photon - first_photon

    @property
    def has_spectra(self) -> bool:
        return self.spectra._has_spectra

    # @property
    # def raster_scan(self) -> RasterScan:
    #     if self.has_raster_scan and self.dataset is not None:
    #         return self.dataset.all_raster_scans[self._raster_scan_dataset_index]

    @property
    def mircrotimes_roi(self) -> np.ndarray:
        times = np.array(self.abstimes)/1E9
        if self.roi_region[0] == 0:
            first_ind = 0
        else:
            where_start = np.where(self.roi_region[0] >= times)
            if len(where_start[0]):
                first_ind = where_start[0][0]
            else:
                first_ind = 0
        where_end = np.where(self.roi_region[1] <= times)
        if len(where_end[0]):
            last_ind = where_end[0][0] + 1
        else:
            last_ind = len(times)
        return self.microtimes[first_ind:last_ind]

    @property
    def first_level_ind_in_roi(self):
        if self.has_levels:
            end_times = np.array([level.times_s[1] for level in self.cpts.levels])
            first_roi_ind = np.argmax(end_times > self.roi_region[0])
            return int(first_roi_ind)

    @property
    def last_level_ind_in_roi(self):
        if len(self.roi_region) == 3:
            last_roi_ind = self.roi_region[2]
        else:
            end_times = np.array([level.times_s[1] for level in self.cpts.levels])
            last_roi_ind = np.argmax(np.round(end_times, 3) >= np.round(self.roi_region[1], 3))
        return int(last_roi_ind)

    @property
    def first_group_level_ind_in_roi(self):
        if self.has_groups and self.group_levels is not None:
            end_times = np.array([level.times_s[1] for level in self.group_levels])
            first_group_roi_ind = np.argmax(end_times > self.roi_region[0])
            return int(first_group_roi_ind)

    @property
    def last_group_level_ind_in_roi(self):
        # if self.has_groups and self.group_levels is not None:
        if self.group_levels is not None:
            # if len(self.roi_region) == 3:
            #     last_roi_ind = self.roi_region[2]
            # else:
            end_times = np.array([level.times_s[1] for level in self.group_levels])
            last_group_roi_ind = np.argmax(np.round(end_times, 3) >= np.round(self.roi_region[1], 3))
            return int(last_group_roi_ind)

    @property
    def raster_scan_coordinates(self) -> tuple:
        particle_attr_keys = self.datadict.attrs.keys()
        if self.has_raster_scan:
            coords = h5_fr.raster_scan_coord(particle=self)
            return coords[1], coords[0]
        else:
            return None, None

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
        if self.has_groups and self.using_group_levels:
            return self.group_levels
        else:
            return self.cpts.levels

    @property
    def levels_roi(self):
        if self.has_groups and self.using_group_levels:
            return self.group_levels
        else:
            return self.cpts.levels[self.first_level_ind_in_roi: self.last_level_ind_in_roi + 1]

    @property
    def group_levels(self) -> List[Level]:
        if self.has_groups:
            return self.ahca.selected_step.group_levels

    @property
    def group_levels_roi(self) -> List[Level]:
        if self.has_groups:
            group_levels = self.group_levels
            return group_levels[self.first_group_level_ind_in_roi: self.last_group_level_ind_in_roi + 1]

    @property
    def num_levels(self):
        if self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_num_levels
        else:
            return self.cpts.num_levels

    @property
    def num_levels_roi(self):
        return (self.last_level_ind_in_roi - self.first_level_ind_in_roi) + 1

    @property
    def dwell_time(self):
        return (self.abstimes[-1] - self.abstimes[0]) / 1E9

    @property
    def dwell_time_roi(self):
        if self.has_levels:
            return self.levels_roi[-1].times_s[1] - self.levels_roi[0].times_s[0]
        else:
            first_photon_ind, last_photon_ind = self.roi_region_photon_inds
            return (self.abstimes[last_photon_ind] - self.abstimes[first_photon_ind])/1E9

    @property
    def level_ints(self):
        if self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_ints
        else:
            return self.cpts.level_ints

    @property
    def level_ints_roi(self):
        if self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_ints
        else:
            return self.cpts.level_ints[self.first_level_ind_in_roi:self.last_level_ind_in_roi]

    @property
    def level_dwelltimes(self):
        if self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_dwelltimes
        else:
            return self.cpts.level_dwelltimes

    @property
    def level_dwelltimes_roi(self):
        if self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_dwelltimes
        else:
            return self.cpts.level_dwelltimes[self.first_level_ind_in_roi:self.last_level_ind_in_roi]

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

    # @property
    # def icon(self):
    #     return ParticleIcons.test_icon

    def remove_and_reset_grouping(self):
        self.ahca = AHCA(particle=self)
        self.using_group_levels = False

    def levels2data(self, use_grouped: bool = None, use_roi: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the Particle objects' levels to generate two arrays for
        plotting the levels.
        Parameters
        ----------
        use_grouped
        use_roi

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
        """
        assert self.has_levels, 'ChangePointAnalysis:\tNo levels to convert to data.'
        if use_grouped is None:
            use_grouped = self.has_groups and self.using_group_levels

        if not use_grouped:
            if not use_roi:
                levels = self.cpts.levels
            else:
                levels = self.levels_roi
        else:
            if not use_roi:
                levels = self.group_levels
            else:
                levels = self.group_levels_roi

        times = np.array([[level.times_s[0], level.times_s[1]] for level in levels])
        times = times.flatten()

        ints = np.array([[level.int_p_s, level.int_p_s] for level in levels])
        ints = ints.flatten()

        return ints, times

    def current2data(self, level_ind: int, use_roi: bool = False) -> [np.ndarray, np.ndarray]:
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

        if not use_roi:
            level = self.levels[level_ind]
        else:
            level = self.levels_roi[level_ind]
        times = np.array(level.times_ns) / 1E9
        levels_data = np.array([level.int_p_s, level.int_p_s])

        return levels_data, times

    def current_group2data(self, group_ind: int) -> [np.ndarray, np.ndarray]:
        assert self.has_groups, 'ChangePointAnalysis:\tNo groups to convert to data.'

        group = self.groups[group_ind]
        times = np.array([self.abstimes[0], self.abstimes[-1]]) /1E9
        group_int = np.array([group.int_p_s, group.int_p_s])
        return group_int, times

    def makehistogram(self, channel=True, add_roi: bool = False):
        """Put the arrival times into a histogram"""

        self._histogram = Histogram(self, start_point=self.startpoint, channel=channel)
        if add_roi:
            self._histogram_roi = Histogram(self, start_point=self.startpoint, channel=channel, is_for_roi=True)
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
                    levels.extend(self.group_levels)
            else:
                levels = self.levels

            for level in levels:
                level.histogram = Histogram(self, level, self.startpoint, channel=channel)

    def makegrouplevelhists(self):
        if self.has_groups and self.ahca.selected_step.groups_have_hists:
            if self.ahca.num_steps == 1:
                self.groups[0].histogram = self.ahca.steps[0].groups[0].histogram
            else:
                groups = self.groups
                for group_level in self.group_levels:
                    g_ind = group_level.group_ind
                    group_level.histogram = groups[g_ind].histogram

    def makegrouphists(self, channel=True):

        if self.has_groups:
            for group in self.groups:
                group.histogram = Histogram(self, group.lvls_inds, self.startpoint, channel=channel)
            self.ahca.selected_step.groups_have_hists = True

    def binints(self, binsize):
        """Bin the absolute times into a trace using binsize"""

        self.bin_size = binsize
        self.binnedtrace = Trace(self, self.bin_size)

    def trim_trace(self, min_level_dwell_time: float, min_level_int: int, reset_roi: bool = True):
        trimmed = None
        if self.has_levels and self.level_ints[-1] < min_level_int:
            trimmed = False
            trim_time_total = 0
            first_valid_reversed_ind = None
            for ind_reverse in reversed(range(0, self.num_levels)):
                if self.level_ints[ind_reverse] <= min_level_int:
                    trim_time_total += self.level_dwelltimes[ind_reverse]
                    first_valid_reversed_ind = ind_reverse
                else:
                    first_valid_reversed_ind = ind_reverse
                    break
            if trim_time_total >= min_level_dwell_time and first_valid_reversed_ind > 1:
                last_active_time = self.levels[first_valid_reversed_ind].times_s[1]
                min_time = 0
                if not reset_roi:
                    if last_active_time > self.roi_region[0] and last_active_time - self.roi_region[0] > 0.5:
                        min_time = self.roi_region[0]
                    else:
                        min_time = last_active_time - 0.5
                    if last_active_time > self.roi_region[1]:
                        last_active_time = self.roi_region[1]
                if min_time >= 0:
                    self.roi_region = (min_time, last_active_time, first_valid_reversed_ind)
                    trimmed = True
        return trimmed


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
        try:
            endbin = np.int(np.max(data) / binsize_ns)
        except ValueError:
            endbin = 0

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
                 trim_start: bool = False,
                 is_for_roi: bool = False):
        assert not (level is not None and is_for_roi), "ROI can't be used for a Level"
        self.is_for_roi = is_for_roi
        self.fitted_with_roi = None
        self.roi_region_used = None
        no_sort = False
        self._particle = particle
        self.level = level
        self.original_kwargs = {'start_point': start_point,
                                'channel': channel,
                                'trim_start': trim_start}
        self.setup(level=level, use_roi=is_for_roi, **self.original_kwargs)

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
        self.residuals = None
        self.fwhm = None
        self.chisq = None
        self.dw = None
        self.dw_bound = None

    def setup(self, level: Union[Level, List[int]] = None,
              start_point: float = None,
              channel: bool = True,
              trim_start: bool = False,
              use_roi: bool = False):
        no_sort = False
        if level is None:
            if not use_roi:
                self.microtimes = self._particle.microtimes[:]
            else:
                self.microtimes = self._particle.mircrotimes_roi
                self.roi_region_used = self._particle.roi_region
        elif type(level) is list:
            if not self._particle.has_groups:
                logger.error("Multiple levels provided, but has no groups")
                raise RuntimeError("Multiple levels provided, but has no groups")
            self.microtimes = np.array([])
            for ind in level:
                self.microtimes = np.append(self.microtimes, self._particle.cpts.levels[
                    ind].microtimes)
        else:
            self.microtimes = self.level.microtimes[:]
        if self.microtimes.size == 0:
            self.decay = np.empty(1)
            self.t = np.empty(1)
        else:
            # tmin = self.microtimes.min()
            tmin = min(self._particle.tmin, self.microtimes.min())
            tmax = max(self._particle.tmax, self.microtimes.max())
            if start_point is None:
                pass
            else:
                if channel:
                    start_point = int(start_point)
                    t = np.arange(tmin, tmax, self._particle.channelwidth)
                    tmin = t[start_point]
                    no_sort = True
                else:
                    tmin = start_point
                    tmax = max(self._particle.tmax, self.microtimes.max())

            sorted_micro = np.sort(self.microtimes)
            if not no_sort and trim_start:
                tmin = sorted_micro[np.searchsorted(sorted_micro, tmin)]  # Make sure bins align with TCSPC bins
            tmax = sorted_micro[np.searchsorted(sorted_micro, tmax) - 1]  # - 1  # Fix if max is end

            window = tmax - tmin
            numpoints = int(window // self._particle.channelwidth)

            t = np.arange(tmin, tmax, self._particle.channelwidth)

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
                dbg.p(f"Histogram object of {self._particle.name} does not have a valid"
                      f" self.t attribute", "Histogram")

    def update_roi(self):
        self.setup(level=self.level, use_roi=True, **self.original_kwargs)

    @property
    def t(self):
        return self._t.copy()

    @t.setter
    def t(self, value):
        self._t = value

    def fit(self, numexp, tauparam, ampparam, shift, decaybg, irfbg, boundaries, addopt, irf, fwhm=None):
        if addopt is not None:
            addopt = ast.literal_eval(addopt)

        self.numexp = numexp

        # TODO: debug option that would keep the fit object (not done normally to conserve memory)
        try:
            if numexp == 1:
                fit = tcspcfit.OneExp(irf, self.decay, self.t, self._particle.channelwidth,
                                      tauparam, None, shift, decaybg, irfbg, boundaries, addopt, fwhm=fwhm)
            elif numexp == 2:
                fit = tcspcfit.TwoExp(irf, self.decay, self.t, self._particle.channelwidth,
                                      tauparam, ampparam, shift, decaybg, irfbg, boundaries, addopt, fwhm=fwhm)
            elif numexp == 3:
                fit = tcspcfit.ThreeExp(irf, self.decay, self.t, self._particle.channelwidth,
                                        tauparam, ampparam, shift, decaybg,
                                        irfbg, boundaries, addopt, fwhm=fwhm)
        except Exception as e:
            trace_string = ''
            for trace_part in traceback.format_tb(e.__traceback__):
                trace_string = trace_string + trace_part
            logger.error(e.args[0] + '\n\n' + trace_string[:-1])
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
            self.fwhm = fit.fwhm
            self.chisq = fit.chisq
            self.dw = fit.dw
            self.dw_bound = fit.dw_bound
            self.residuals = fit.residuals
            self.fitted = True
            if numexp == 1:
                self.avtau = self.tau
            else:
                self.avtau = sum(self.tau * self.amp) / self.amp.sum()

        return True

    def levelhist(self, level):
        levelobj = self._particle.levels[level]
        tmin = levelobj.microtimes[:].min()
        tmax = levelobj.microtimes[:].max()
        window = tmax-tmin
        numpoints = int(window // self._particle.channelwidth)
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
            for level in particle.cpts.levels:
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
        shift = fit_param.shift[:-1] / channelwidth
        shiftfix = fit_param.shift[-1]
        shift = [*shift, shiftfix]
        boundaries = [start, end, fit_param.autostart, fit_param.autoend]

        for hist in all_hists:
            try:
                if not hist.fit(fit_param.numexp, fit_param.tau, fit_param.amp,
                                shift, fit_param.decaybg,
                                fit_param.irfbg, boundaries, fit_param.addopt,
                                fit_param.irf, fit_param.fwhm):
                    pass  # fit unsuccessful
            except AttributeError:
                print("No decay")


class RasterScan:
    def __init__(self, raster_scan_dataset: h5py.Dataset, particle_indexes: List[int],
                 dataset_index: int = None):
        self.dataset = raster_scan_dataset
        self.dataset_index = dataset_index
        self.particle_indexes = particle_indexes
        self.integration_time = h5_fr.rs_integration_time(part_or_rs=self)
        self.pixel_per_line = h5_fr.rs_pixels_per_line(part_or_rs=self)
        self.range = h5_fr.rs_range(part_or_rs=self)
        self.x_start = h5_fr.rs_x_start(part_or_rs=self)
        self.y_start = h5_fr.rs_y_start(part_or_rs=self)

        self.x_axis_pos = np.linspace(self.x_start, self.x_start + self.range, self.pixel_per_line)
        self.y_axis_pos = np.linspace(self.y_start, self.y_start + self.range, self.pixel_per_line)


class Spectra:
    def __init__(self, particle: Particle):
        self._particle = particle
        self._has_spectra = h5_fr.has_spectra(particle=particle)
        if self._has_spectra:
            self.data = h5_fr.spectra(particle=particle)
            self.wavelengths = h5_fr.spectra_wavelengths(particle=particle)
            self.series_times = h5_fr.spectra_abstimes(particle=particle)


def start_at_value(decay, t, neg_t=True, decaystart=None):
    if decaystart is None:
        decaystart = np.nonzero(decay)[0][0]
    if neg_t:
        t -= t[decaystart]
    t = t[decaystart:]
    decay = decay[decaystart:]
    return decay, t
