"""Module for handling SMS data from HDF5 files

Bertus van Heerden and Joshua Botha
University of Pretoria
2018
"""
from __future__ import annotations

__docformat__ = "NumPy"

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
import grouping
import tcspcfit
from change_point import ChangePoints
from generate_sums import CPSums
from grouping import AHCA, GlobalLevel
from my_logger import setup_logger
from processes import ProcessProgFeedback, ProcessProgress, PassSigFeedback
from tcspcfit import FittingParameters
import smsh5_file_reader as h5_fr
from antibunching import AntibunchingAnalysis

if TYPE_CHECKING:
    from change_point import Level
    from grouping import GlobalLevel

logger = setup_logger(__name__)


class H5dataset:
    """Represents an entire HDF5 dataset

    Parameters
    ----------
    filename : str
        HDF5 file path
    sig_fb : PassSigFeedback, optional
        feedback queue for signals
    prog_fb : ProcessProgFeedback, optional
        feedback queue for updating progress bar
    """

    def __init__(
        self,
        filename,
        sig_fb: PassSigFeedback = None,
        prog_fb: ProcessProgFeedback = None,
    ):
        self.cpa_has_run = False
        self.use_parallel = False
        self.name = filename
        if prog_fb is not None:
            prog_fb.set_status(status="Reading file...")
        self._file = h5pickle.File(self.name, "r")
        self.file_version = h5_fr.file_version(dataset=self)

        all_keys = self.file.keys()
        part_keys = [part_key for part_key in all_keys if "Particle " in part_key]

        natural_p_names = [None] * len(part_keys)
        natural_key = []
        for name in part_keys:
            for seg in re.split("(\d+)", name):
                if seg.isdigit():
                    natural_key.append(int(seg))
        for num, key_num in enumerate(natural_key):
            natural_p_names[key_num - 1] = part_keys[num]

        self.all_sums = CPSums(n_min=10, n_max=1000, prog_fb=prog_fb)

        self.all_raster_scans = list()
        map_particle_indexes = self.get_all_raster_scans(particle_names=natural_p_names)
        if map_particle_indexes is not None:
            self.has_raster_scans = True
        else:
            self.has_raster_scans = False

        self.particles = []
        self.num_parts = 0
        for num, particle_name in enumerate(natural_p_names):
            if map_particle_indexes is not None:
                this_raster_scan_index = map_particle_indexes[num]
                this_raster_scan = self.all_raster_scans[map_particle_indexes[num]]
            else:
                this_raster_scan_index = None
                this_raster_scan = None
            prim_part = Particle(
                name=particle_name,
                dataset_ind=num,
                dataset=self,
                raster_scan_dataset_index=this_raster_scan_index,
                raster_scan=this_raster_scan,
            )
            if (
                h5_fr.abstimes2(prim_part) is not None
            ):  # if second card data exists, create secondary particle
                # print(particle_name)
                sec_part = Particle(
                    name=particle_name,
                    dataset_ind=num,
                    dataset=self,
                    raster_scan_dataset_index=this_raster_scan_index,
                    raster_scan=this_raster_scan,
                    is_secondary_part=True,
                    prim_part=prim_part,
                )
                prim_part.sec_part = sec_part
                self.particles.append(prim_part)
                self.particles.append(sec_part)
            else:
                self.particles.append(prim_part)
            self.num_parts += 1
        assert self.num_parts == h5_fr.num_parts(dataset=self)
        self.channelwidth = None
        self.save_selected = None
        self.has_levels = False
        self.has_groups = False
        self.has_lifetimes = False
        self.irf = None
        self.irf_t = None
        self.has_irf = False
        self.has_spectra = False
        self.has_corr = False
        self.global_particle = None

    @property
    def file(self):
        """HDF5 file."""
        if self._file is not None and self._file.__bool__() is True:
            return self._file
        else:
            raise Warning("File not set")

    def unload_file(self, should_close: bool = True, should_delete: bool = True):
        """Remove file reference and close and/or delete the file."""
        if should_close:
            if self._file is not None and self._file.__bool__() is True:
                self._file.close()
                if should_delete:
                    del self._file
        self._file = None

    def load_file(self, new_file: h5pickle.File):
        """Load pickled HDF file."""
        if type(new_file) is h5pickle.File and new_file.__bool__() is True:
            self._file = new_file
            self.name = new_file.filename
            return True
        else:
            if type(new_file) is h5pickle.File and new_file.__bool__() is False:
                logger.error("Provided H5 file file is closed.")
            else:
                logger.error("Provided H5 file invalid.")

    def get_all_raster_scans(self, particle_names: List[str]) -> list:
        """Get all the raster scans from the file and handle appropriately.

        Each raster scan has one or more particles associated with it. In the HDF5
        file, this is manifested as having the same raster scan duplicated across
        particle objects. This function reads out all unique raster scans and assigns
        them each their particles. The return value is a map from particle numbers to
        'raster scan numbers - if there's a raster scan for particles 1 and 2
        and another one for particles 3-5, the map would be: [1, 1, 2, 2, 2]

        Arguments
        ---------
        particle_names : list[str]
            list of particle names to read raster scan data from

        Returns
        -------
        map_particle_index : list or None
            maps particle numbers to raster scan numbers.
        """
        raster_scans = list()
        file_keys = self.file.keys()
        for num, particle_name in enumerate(particle_names):
            if particle_name in file_keys:
                particle = h5_fr.particle(particle_num=num + 1, dataset=self)
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
                                RasterScan(
                                    h5dataset=self,
                                    particle_num=num,
                                    h5dataset_index=raster_scan_num,
                                    particle_indexes=group_indexes,
                                )
                            )
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
                                    RasterScan(
                                        h5dataset=self,
                                        particle_num=num,
                                        h5dataset_index=raster_scan_num,
                                        particle_indexes=group_indexes,
                                    )
                                )
                                raster_scan_counter += 1
                                group_indexes = [num]
                                raster_scan_num = len(self.all_raster_scans)
                            self.all_raster_scans.append(
                                RasterScan(
                                    h5dataset=self,
                                    particle_num=num,
                                    h5dataset_index=raster_scan_counter,
                                    particle_indexes=group_indexes,
                                )
                            )
                            raster_scan_counter += 1
                    else:
                        raster_scan_num = 0
                        if len(raster_scans) == 1:
                            group_indexes = [0]
                            self.all_raster_scans.append(
                                RasterScan(
                                    h5dataset=self,
                                    particle_num=num,
                                    h5dataset_index=raster_scan_counter,
                                    particle_indexes=group_indexes,
                                )
                            )
                    group_indexes = [num]
                    prev_raster_scan = raster_scan

                map_particle_index.append(raster_scan_num)
            return map_particle_index
        else:
            return None

    def makehistograms(self, remove_zeros=True, startpoint=None, channel=True):
        """Put the (micro) arrival times into histograms."""

        for particle in self.particles:
            particle.makehistograms(remove_zeros, startpoint, channel)

    def bin_all_ints(
        self,
        binsize: float,
        sig_fb: PassSigFeedback = None,
        prog_fb: ProcessProgFeedback = None,
    ):
        """Bin the absolute times into intensity traces.

        Arguments
        ---------
        binsize : float
            Time bin size in ms
        sig_fb : PassSigFeedback, optional
            feedback queue for signals
        prog_fb : ProcessProgFeedback, optional
            feedback queue for updating progress bar
        """
        if prog_fb:
            proc_tracker = ProcessProgress(
                prog_fb=prog_fb, num_iterations=len(self.particles)
            )

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
        """Save selected particle to a new or existing HDF5 file.

        Arguments
        ---------
        file_path : str
            Path to existing file, or to file that will be created.
        selected_nums : List[int]
            Particle numbers to be written to HDF5 file.
        """

        add = os.path.exists(file_path)
        if add:
            new_h5file = h5pickle.File(file_path, mode="r+")
            num_existing = new_h5file.attrs.get("# Particles")
        else:
            new_h5file = h5pickle.File(file_path, mode="w")
            num_existing = 0

        for i, selected in enumerate(selected_nums):
            new_h5file.copy(
                self.file[f"/Particle {selected}"],
                new_h5file,
                name=f"/Particle {num_existing + i + 1}",
            )

        if add:
            new_h5file.attrs.modify("# Particles", num_existing + len(selected_nums))
        else:
            new_h5file.attrs.create("# Particles", len(selected_nums))
        new_h5file.close()


class Particle:
    """Represents a particle in an `H5dataset`.

    Parameters
    ----------
    name : str
        The name of the particle
    dataset_ind : H5dataset
        The index of the particle in the dataset
    dataset : H5dataset
        The instance of the dataset to which this particle belongs
    raster_scan_dataset_index : int
        The index of the raster scan connected to the particle
    raster_scan : RasterScan
        The raster scan object this particle is connected to
    is_secondary_part : bool
        Whether this is a "secondary particle" that contains the data from a second TCSCPC card
    prim_part : Particle
        If this particle is a secondary particle, the corresponding primary particle
    sec_part : Particle
        If this particle is a primary particle, the corresponding secondary particle
    tmin : int, optional
        Minimum photon micro time in ns
    tmax : int, optional
        Maximum photon micro time in ns
    channelwidth : float, optional
        TCSPC histogram channelwidth in ns. Normally automatically determined.
    is_global : bool = False
        TODO: What is this?
    """

    def __init__(
        self,
        name: str,
        dataset_ind: int,
        dataset: H5dataset,
        raster_scan_dataset_index: int = None,
        raster_scan: RasterScan = None,
        is_secondary_part: bool = False,
        prim_part: Particle = None,
        sec_part: Particle = None,
        tmin=None,
        tmax=None,
        channelwidth=None,
        is_global: bool = False,
    ):
        self.uuid = uuid1()
        self.name = name
        self.dataset = dataset
        self.dataset_ind = dataset_ind
        # self.get_file = lambda: dataset.get_file()
        self.file_version = h5_fr.file_version(dataset=dataset)
        # self.dataset_particle = self.file[self.name]
        if not is_global:
            self.is_secondary_part = is_secondary_part
            self.prim_part = prim_part
            self.sec_part = sec_part
            if not self.is_secondary_part:
                # self.microtimes = h5_fr.microtimes(particle=self)
                # self.abstimes = h5_fr.abstimes(particle=self)
                self.num_photons = len(self.abstimes)
            else:
                # self.microtimes = h5_fr.microtimes2(particle=self)
                # self.abstimes = h5_fr.abstimes2(particle=self)
                self.num_photons = len(self.abstimes)
            self.tcspc_card = h5_fr.tcspc_card(particle=self)
            self.int_trace = h5_fr.int_trace(particle=self)
            self.cpts = ChangePoints(
                self
            )  # Added by Josh: creates an object for Change Point Analysis (cpa)
            self.ahca = AHCA(
                self
            )  # Added by Josh: creates an object for Agglomerative Hierarchical Clustering Algorithm

            self.avg_int_weighted = None
            self.int_std_weighted = None

            if self.is_secondary_part:
                self.spectra = self.prim_part.spectra
                self._raster_scan_dataset_index = (
                    self.prim_part._raster_scan_dataset_index
                )
                self.raster_scan = self.prim_part.raster_scan
                self.has_raster_scan = self.prim_part.has_raster_scan
                self.description = self.prim_part.description
            # self.ab_analysis = self.prim_part.ab_analysis
            else:
                self.spectra = Spectra(self)
                self._raster_scan_dataset_index = raster_scan_dataset_index
                self.raster_scan = raster_scan
                self.has_raster_scan = raster_scan is not None
                self.description = h5_fr.description(particle=self)
                self._ab_analysis = AntibunchingAnalysis(self)

            self.irf = None
            if channelwidth is None and not (
                len(self.microtimes) == 0 and len(self.abstimes) == 0
            ):
                differences = np.diff(np.sort(self.microtimes[:]))
                possible_channelwidths = np.unique(np.diff(np.unique(differences)))
                if len(possible_channelwidths) != 1:
                    channelwidth = 0.01220703125
                    logger.warning(
                        f"Channel width could not be determined. Inspect {self.name}. A default of {channelwidth} used."
                    )
                else:
                    channelwidth = possible_channelwidths[0]
            else:
                channelwidth = 0.01220703125
                logger.warning(
                    f"Channel width could not be determined. Inspect {self.name}. A default of {channelwidth} used."
                )
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
                self.roi_region = (0, self.abstimes[-1] / 1e9)
            except IndexError:
                self.roi_region = (0, 0)

            self.startpoint = None
            self.level_or_group_selected = None
            self.using_group_levels = False

            self.has_fit_a_lifetime = False
            self.has_exported = False
            self.is_global = False
        else:
            self.cpts = ChangePoints(
                self
            )  # Added by Josh: creates an object for Change Point Analysis (cpa)
            self.ahca = AHCA(
                self
            )  # Added by Josh: creates an object for Agglomerative Hierarchical Clustering Algorithm
            self.is_global = False

    @property
    def file(self):
        """HDF5 file."""
        return self.dataset.file

    @property
    def file_group(self):
        """The particle's group in the HDF5 file."""
        if self.file is not None:
            return self.file[self.name]

    @property
    def microtimes(self) -> h5pickle.Dataset:
        """The particle's microtimes."""
        if self.file is not None and self.file.__bool__() is True:
            if not self.is_secondary_part:
                return h5_fr.microtimes(particle=self)
            else:
                return h5_fr.microtimes2(particle=self)

    @property
    def abstimes(self) -> h5pickle.Dataset:
        """The particle's absolute times."""
        if self.file is not None and self.file.__bool__() is True:
            if not self.is_secondary_part:
                return h5_fr.abstimes(particle=self)
            else:
                return h5_fr.abstimes2(particle=self)

    @property
    def histogram(self) -> Histogram:
        """The particle's `Histogram`."""
        if not self.use_roi_for_histogram:
            hist = self._histogram
        else:
            hist = self._histogram_roi
        return hist

    #  TODO: These 2 functions have no usages
    def set_histgram(self, histogram: Histogram) -> None:
        self._histogram = histogram

    def set_histgram_roi(self, histogram: Histogram) -> None:
        self._histogram_roi = histogram

    @property
    def use_roi_for_grouping(self) -> bool:
        """Whether to use the particle's ROI for grouping."""
        if not self.is_secondary_part:
            return self.ahca.use_roi_for_grouping

    @property
    def grouped_with_roi(self) -> bool:
        """Whether the particle's `ahca` used ROI."""
        if not self.is_secondary_part:
            return self.ahca.grouped_with_roi

    @property
    def roi_region_photon_inds(self) -> Tuple[int, int]:
        """Indices of photons at ROI boundaries."""
        first_photon = 0
        last_photon = self.num_photons - 1
        roi_start = self.roi_region[0]
        roi_end = self.roi_region[1]
        epsilon_t = 0.1
        end_t = self.abstimes[-1] / 1e9
        if (
            roi_start >= 0 + epsilon_t / 2
            or not roi_end - epsilon_t / 2 <= end_t <= roi_end + epsilon_t / 2
        ):
            times = self.abstimes[:] / 1e9
            first_photon = np.argmin(roi_start > times)
            last_photon = np.argmin(roi_end > times)
        return first_photon, last_photon

    @property
    def num_photons_roi(self) -> int:
        """Number of photons in the particle ROI."""
        first_photon, last_photon = self.roi_region_photon_inds
        return last_photon - first_photon

    @property
    def has_spectra(self) -> bool:
        """Whether the particle has spectra."""
        return self.spectra._has_spectra

    @property
    def microtimes_roi(self) -> np.ndarray:
        """Microtimes from the particle ROI."""
        times = np.array(self.abstimes) / 1e9
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
        """Index of the first level in the particle's ROI."""
        if self.has_levels:
            end_times = np.array([level.times_s[1] for level in self.cpts.levels])
            first_roi_ind = np.argmax(end_times > self.roi_region[0])
            return int(first_roi_ind)

    @property
    def last_level_ind_in_roi(self):
        """Index of the last level in the particle's ROI."""
        if len(self.roi_region) == 3:
            last_roi_ind = self.roi_region[2]
        else:
            end_times = np.array([level.times_s[1] for level in self.cpts.levels])
            last_roi_ind = np.argmax(
                np.round(end_times, 3) >= np.round(self.roi_region[1], 3)
            )
        return int(last_roi_ind)

    @property
    def first_group_level_ind_in_roi(self):
        """Index of the first grouped level in the particle's ROI."""
        if not self.is_secondary_part:
            if self.has_groups and self.group_levels is not None:
                end_times = np.array([level.times_s[1] for level in self.group_levels])
                first_group_roi_ind = np.argmax(end_times > self.roi_region[0])
                return int(first_group_roi_ind)

    @property
    def last_group_level_ind_in_roi(self):
        """Index of the last grouped level in the particle's ROI."""
        # if self.has_groups and self.group_levels is not None:
        if not self.is_secondary_part:
            if self.group_levels is not None:
                # if len(self.roi_region) == 3:
                #     last_roi_ind = self.roi_region[2]
                # else:
                end_times = np.array([level.times_s[1] for level in self.group_levels])
                last_group_roi_ind = np.argmax(
                    np.round(end_times, 3) >= np.round(self.roi_region[1], 3)
                )
                return int(last_group_roi_ind)

    @property
    def raster_scan_coordinates(self) -> tuple:
        """The particle's RS coordinates."""
        if self.has_raster_scan:
            coords = h5_fr.raster_scan_coord(particle=self)
            return coords[1], coords[0]
        else:
            return None, None

    @property
    def has_levels(self):
        """Whether the particle has levels."""
        return self.cpts.has_levels

    @property
    def has_groups(self):
        """Whether the particle has groups."""
        if not self.is_secondary_part:
            return self.ahca.has_groups

    @property
    def has_corr(self):
        """Whether the particle has a second-order correlation."""
        return self.ab_analysis.has_corr

    @property
    def ab_analysis(self):
        if self.is_secondary_part:
            return self.prim_part._ab_analysis
        else:
            return self._ab_analysis

    @ab_analysis.setter
    def ab_analysis(self, ab_analysis):
        if not self.is_secondary_part:
            self._ab_analysis = ab_analysis


    @property
    def groups(self) -> List[grouping.Group]:
        """The particle's grouped levels."""
        if not self.is_secondary_part and self.has_groups:
            return self.ahca.selected_step.groups

    @property
    def num_groups(self):
        """The number of level groups."""
        if not self.is_secondary_part and self.has_groups:
            return self.ahca.selected_step.num_groups

    @property
    def groups_bounds(self):
        """The particle's groups' bounds."""
        if not self.is_secondary_part:
            return self.ahca.selected_step.calc_int_bounds()

    @property
    def groups_ints(self):
        """The particle's group intensities."""
        if not self.is_secondary_part:
            return self.ahca.selected_step.group_ints

    @property
    def grouping_bics(self):
        """The particle's group BIC's."""
        if not self.is_secondary_part:
            return self.ahca.bics

    @property
    def grouping_selected_ind(self):
        """The currently selected group index."""
        if not self.is_secondary_part:
            return self.ahca.selected_step_ind

    @property
    def best_grouping_ind(self):
        """The index of the grouping step with max BIC."""
        if not self.is_secondary_part:
            return self.ahca.best_step_ind

    @grouping_selected_ind.setter
    def grouping_selected_ind(self, ind: int):
        """The index of the selected grouping step."""
        if not self.is_secondary_part:
            self.ahca.selected_step_ind = ind

    @property
    def grouping_num_groups(self):
        """The number of groups in each AHCA step"""
        if not self.is_secondary_part:
            return self.ahca.steps_num_groups

    def reset_grouping_ind(self):
        """Reset the grouping step selection."""
        if not self.is_secondary_part:
            self.ahca.reset_selected_step()

    @property
    def levels(self):
        """The particle's raw or grouped levels."""
        if not self.is_secondary_part and self.has_groups and self.using_group_levels:
            return self.group_levels
        else:
            return self.cpts.levels

    #  TODO: The following 4 functions don't seem to be entirely sensible
    @property
    def levels_roi(self):
        """The particle's raw or grouped levels, from ROI."""
        if not self.is_secondary_part and self.has_groups and self.using_group_levels:
            return self.group_levels
        else:
            return self.cpts.levels[
                self.first_level_ind_in_roi : self.last_level_ind_in_roi + 1
            ]

    @property
    def levels_roi_force(self):
        """The particle's raw levels, from ROI."""
        return self.cpts.levels[
            self.first_level_ind_in_roi : self.last_level_ind_in_roi + 1
        ]

    @property
    def group_levels(self) -> List[Union[Level, GlobalLevel]]:
        """The particle's grouped levels."""
        if not self.is_secondary_part and self.has_groups:
            return self.ahca.selected_step.group_levels

    @property
    def group_levels_roi(self) -> List[Level]:
        """The particle's grouped levels, from ROI."""
        if not self.is_secondary_part and self.has_groups:
            group_levels = self.group_levels
            return group_levels[
                self.first_group_level_ind_in_roi : self.last_group_level_ind_in_roi + 1
            ]

    @property
    def num_levels(self):
        """Number of raw or grouped particle levels."""
        if self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_num_levels
        else:
            return self.cpts.num_levels

    @property
    def num_levels_roi(self):
        """Number of raw particle levels in ROI."""
        return (self.last_level_ind_in_roi - self.first_level_ind_in_roi) + 1

    @property
    def dwell_time_s(self):
        """The particle's total measurement time."""
        return (self.abstimes[-1] - self.abstimes[0]) / 1e9

    @property
    def dwell_time_roi(self):
        """The particle's ROI measurement time."""
        if self.has_levels:
            return self.levels_roi[-1].times_s[1] - self.levels_roi[0].times_s[0]
        else:
            first_photon_ind, last_photon_ind = self.roi_region_photon_inds
            return (
                self.abstimes[last_photon_ind] - self.abstimes[first_photon_ind]
            ) / 1e9

    @property
    def level_ints(self):
        """The particle's raw or grouped level intensities."""
        if not self.is_secondary_part and self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_ints
        else:
            return self.cpts.level_ints

    @property
    def level_ints_roi(self):
        """The particle's ROI raw or grouped level intensities."""
        #  TODO: shouldn't grouped levels also use ROI?
        if not self.is_secondary_part and self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_ints
        else:
            return self.cpts.level_ints[
                self.first_level_ind_in_roi : self.last_level_ind_in_roi
            ]

    @property
    def level_dwelltimes(self):
        """The particle's raw or grouped level dwelltimes."""
        if not self.is_secondary_part and self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_dwelltimes
        else:
            return self.cpts.level_dwelltimes

    @property
    def level_dwelltimes_roi(self):
        """The particle's ROI raw or grouped level dwelltimes."""
        if not self.is_secondary_part and self.has_groups and self.using_group_levels:
            return self.ahca.selected_step.group_level_dwelltimes
        else:
            return self.cpts.level_dwelltimes[
                self.first_level_ind_in_roi : self.last_level_ind_in_roi
            ]

    @property
    def has_burst(self) -> bool:
        """Whether the particle has a 'photon burst'."""
        return self.cpts.has_burst

    @property
    def burst_levels(self) -> np.ndarray:
        """Levels containing a 'photon burst'."""
        return self.cpts.burst_levels

    @has_burst.setter
    def has_burst(self, value: bool):
        self.cpts.has_burst = value

    @burst_levels.setter
    def burst_levels(self, value: np.ndarray):
        self.cpts.burst_levels = value

    @property
    def numexp(self):
        """Number of exponents in fitted decay model."""
        return self.histogram.numexp

    @property
    def unique_name(self):
        """Unique particle name in the case of dual-channel measurement."""
        if self.is_secondary_part:
            return self.name + "_2"
        else:
            return self.name

    @property
    def has_global_grouping(self) -> bool:
        """Whether the H5dataset has global groups."""
        if hasattr(self, "dataset") and hasattr(self.dataset, "global_particle"):
            gp = self.dataset.global_particle
            return (
                gp is not None
                and self.dataset_ind in gp.contributing_particles_dataset_inds
            )
        else:
            return False

    @property
    def global_particle(self) -> GlobalParticle:
        """The H5dataset's global particle."""
        if self.has_global_grouping:
            return self.dataset.global_particle

    @property
    def global_group_levels(self):
        """The H5dataset's globally grouped levels."""
        if self.has_global_grouping:
            gp = self.global_particle
            return list(
                filter(
                    lambda l: l.parent_particle_dataset_ind == self.dataset_ind,
                    gp.global_levels,
                )
            )

    def remove_and_reset_grouping(self):
        """Re-initialize grouping and remove current data."""
        if not self.is_secondary_part:
            self.ahca = AHCA(particle=self)
            self.using_group_levels = False

    def levels2data(
        self,
        use_grouped: bool = None,
        use_roi: bool = False,
        use_global_groups: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Level plotting data.

        Uses the Particle objects' levels to generate two arrays for
        plotting the levels.

        Arguments
        ---------
        use_grouped : bool, optional
            Whether to use grouped levels - if not provided, defaults to
            True if groups exist and false if they don't.
        use_global_groups : bool = False
            whether to use globally grouped levels
        use_roi : bool = False
            whether to use ROI

        Returns
        -------
        ints, times : Tuple[np.ndarray, np.ndarray]
            Intensities as a function of time for plotting.
        """
        assert self.has_levels, "ChangePointAnalysis:\tNo levels to convert to data."
        if self.is_secondary_part:
            return
        if use_grouped is None:
            use_grouped = self.has_groups and self.using_group_levels

        did_use_global = False
        if use_grouped or use_global_groups:
            if use_global_groups:
                levels = self.global_group_levels
                did_use_global = True
            elif not use_roi:
                levels = self.group_levels
            else:
                levels = self.group_levels_roi
        else:
            if not use_roi:
                levels = self.cpts.levels
            else:
                levels = self.levels_roi

        # if not use_global_groups:
        times = np.array([[level.times_s[0], level.times_s[1]] for level in levels])
        # else:
        #     times = np.array(
        #         [
        #             [
        #                 level.times_s[0] - level.start_time_offset_ns / 1e9,
        #                 level.times_s[1] - level.start_time_offset_ns / 1e9,
        #             ]
        #             for level in levels
        #         ]
        #     )
        times = times.flatten()

        ints = np.array([[level.int_p_s, level.int_p_s] for level in levels])
        ints = ints.flatten()

        return ints, times

    def lifetimes2data(
        self, use_grouped: bool = None, use_roi: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Level lifetime plotting data.

        Uses the Particle object's levels to generate two arrays for
        plotting the level lifetimes.

        Arguments
        ---------
        use_grouped : bool, optional
            Whether to use grouped levels - if not provided, defaults to
            True if groups exist and false if they don't.
        use_roi : bool = False
            Whether to use ROI.

        Returns
        -------
        lifetimes, times : Tuple[np.ndarray, np.ndarray]
            Lifetime as a function of time for plotting.
        """
        assert (
            self.has_fit_a_lifetime
        ), "ChangePointAnalysis:\tNo levels to convert to data."
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

        times = np.array(
            [
                [level.times_s[0], level.times_s[1]]
                for level in levels
                if level.histogram.fitted
            ]
        )
        times = times.flatten()

        lifetimes = np.array(
            [
                [level.histogram.avtau, level.histogram.avtau]
                for level in levels
                if level.histogram.fitted
            ]
        )
        lifetimes = lifetimes.flatten()

        return lifetimes, times

    def current2data(
        self, level_ind: int, use_roi: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Current level plotting data.

        Uses the Particle object's selected level to generate
        two arrays for plotting level.

        Arguments
        ---------
        level_ind : int
            Index of the level to be plotted.
        use_roi : bool = False
            Whether to use the ROI.

        Returns
        -------
        levels_data, times : Tuple[np.ndarray, np.ndarray]
            Intensity as a function of time for plotting.
        """
        # TODO: Cleanup this function anc the one above it
        assert self.has_levels, "ChangePointAnalysis:\tNo levels to convert to data."

        if not use_roi:
            level = self.levels[level_ind]
        else:
            level = self.levels_roi[level_ind]
        times = np.array(level.times_ns) / 1e9
        levels_data = np.array([level.int_p_s, level.int_p_s])

        return levels_data, times

    def current_group2data(self, group_ind: int) -> [np.ndarray, np.ndarray]:
        """Current group plotting data.

        Uses the Particle object's selected group to generate
        two arrays for plotting the group.

        Arguments
        ---------
        group_ind : int
            Index of the level to be plotted.

        Returns
        -------
        group_int, times : Tuple[np.ndarray, np.ndarray]
            Intensity as a function of time for plotting.
        """
        assert self.has_groups, "ChangePointAnalysis:\tNo groups to convert to data."
        if self.is_secondary_part:
            return

        group = self.groups[group_ind]
        times = np.array([self.abstimes[0], self.abstimes[-1]]) / 1e9
        group_int = np.array([group.int_p_s, group.int_p_s])
        return group_int, times

    def makehistograms(self, remove_zeros, startpoint, channel):
        """Make all histograms - whole trace and levels.

        Arguments
        ---------
        remove_zeros : bool
            Whether to remove zeros at the start of the decay.
        startpoint : int
            Startpoint of the decay in number of time steps.
        channel : bool
            Whether to use the hardware channelwidth
            TODO: remove this parameter including downstream as it is never used
        """
        self.startpoint = startpoint
        self.makehistogram(channel=channel, add_roi=True)
        self.makelevelhists(channel=channel)
        if remove_zeros:
            maxim = 0
            try:
                maxim = max(self.histogram.decaystart, maxim)
            except AttributeError:
                maxim = 0
            self.histogram.decay = self.histogram.decay[maxim:]
            self.histogram.t = self.histogram.t[maxim:]

    def makehistogram(self, channel=True, add_roi: bool = False):
        """Put the arrival times into a histogram.

        Arguments
        ---------
        channel : bool = True
            Whether to use the hardware channelwidth.
        add_roi : bool = True
            Whether to create the ROI Histogram as well.
        """

        self._histogram = Histogram(self, start_point=self.startpoint, channel=channel)
        if add_roi:
            self._histogram_roi = Histogram(
                self, start_point=self.startpoint, channel=channel, is_for_roi=True
            )

    def makelevelhists(
        self,
        channel: bool = True,
        force_cpts_levels: bool = False,
        force_group_levels: bool = False,
    ):
        """Make level histograms.

        Arguments
        ---------
        channel : bool = True
            Whether to use the hardware channelwidth.
        force_cpts_levels : bool = False
            Use self.cpts.levels instead of self.levels.
        force_group_levels : bool = False
            Use self.group_levels instead of self.levels.
        """

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
                level.histogram = Histogram(
                    self, level, self.startpoint, channel=channel
                )

    def makegrouplevelhists(self):
        """Make grouped level histograms."""
        if not self.is_secondary_part and self.has_groups and self.ahca.selected_step.groups_have_hists:
            if self.ahca.num_steps == 1:
                self.groups[0].histogram = self.ahca.steps[0].groups[0].histogram
            else:
                groups = self.groups
                for group_level in self.group_levels:
                    g_ind = group_level.group_ind
                    group_level.histogram = groups[g_ind].histogram

    def makegrouphists(self, channel=True):
        """Make group histograms."""
        if not self.is_secondary_part and self.has_groups:
            for group in self.groups:
                group.histogram = Histogram(
                    self, group.lvls_inds, self.startpoint, channel=channel
                )
            self.ahca.selected_step.groups_have_hists = True

    def binints(self, binsize: int):
        """Bin the absolute times into intensity trace.

        Arguments
        ---------
        binsize : int
            Size of intensity trace time bins.
        """

        self.bin_size = binsize
        self.binnedtrace = Trace(self, self.bin_size)

    def trim_trace(
        self, min_level_dwell_time: float, min_level_int: int, reset_roi: bool = True
    ):
        """Trim the intensity trace.

        This function trims the intensity trace to remove the photobleached end.

        Arguments
        ---------
        min_level_dwell_time : float
            Minimum dwell time of bleached level for trimming.
        min_level_int : int
            Minimum intensity to classify level as not bleached.
        reset_roi : bool = True
            Whether to update the ROI to the trimmed area.

        """
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
                    if (
                        last_active_time > self.roi_region[0]
                        and last_active_time - self.roi_region[0] > 0.5
                    ):
                        min_time = self.roi_region[0]
                    else:
                        min_time = last_active_time - 0.5
                    if last_active_time > self.roi_region[1]:
                        last_active_time = self.roi_region[1]
                if min_time >= 0:
                    self.roi_region = (
                        min_time,
                        last_active_time,
                        first_valid_reversed_ind,
                    )
                    trimmed = True
        return trimmed


class FakeCpts:
    """Fake ChangePoints object for GlobalParticle.

    Parameters
    ----------
    num_levels : int
        number of intensity levels
    levels : List[GlobalLevel]
        list of global levels
    """

    def __init__(self, num_levels: int, levels: list):
        self.num_levels = num_levels
        self.levels = levels
        self.num_cpts = num_levels - 1
        self.has_levels = True


class GlobalParticle:
    """Particle-like object containing levels from all particles.

    Parameters
    ----------
    particles : List[Particle]
        Particles to include in global particle.
    use_roi : bool = False
        Whether to use ROI's.
    """

    def __init__(self, particles: List[Particle], use_roi: bool = False):
        self.is_global = True
        self.name = "Global Particle"

        levels = []
        start_time_offset_ns = 0
        for p in particles:
            p_levels = p.levels_roi if use_roi else p.levels
            for l in p_levels:
                level = GlobalLevel(
                    global_particle=self,
                    parent_particle_dataset_ind=p.dataset_ind,
                    particle_levels=[l],
                    int_p_s=l.int_p_s,
                    group_ind=l.group_ind,
                    start_time_offset_ns=start_time_offset_ns,
                    dwell_time_ns=l.dwell_time_ns,
                    num_photons=l.num_photons,
                )
                levels.append(level)
            start_time_offset_ns += p_levels[-1].times_ns[1]

        self.contributing_particles_dataset_inds = [p.dataset_ind for p in particles]

        self.levels = levels
        self.num_levels = len(levels)
        self.num_levels_roi = self.num_levels
        self.dwell_time = np.sum([l.dwell_time_s for l in self.levels])
        self.num_photons = np.sum([l.num_photons for l in self.levels])

        self.uuid = uuid1()
        self.use_roi_for_grouping = False
        self.cpts = None

        self.cpts = FakeCpts(num_levels=self.num_levels, levels=self.levels)

        self.ahca = AHCA(particle=self)

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

    @property
    def level_particle_dataset_inds(self):
        return [l.parent_particle_dataset_ind for l in self.global_levels]

    @property
    def global_levels(self) -> list:
        if self.ahca.has_groups:
            return self.ahca.selected_step.group_levels

    def run_grouping(self):
        self.ahca.run_grouping()

        # all_times = []
        # all_ints = []
        # all_particle_names = []
        # for l in levels:
        #     all_times.extend(l.times_s)
        #     all_ints.extend([l.int_p_s, l.int_p_s])
        #     all_particle_names.extend([l.particle.name, l.particle.name])
        # df = pd.DataFrame(data={
        #     "times": all_times,
        #     "ints": all_ints,
        #     "particle": all_particle_names
        # })


class Trace:
    """Binned intensity trace.

    Parameters
    ----------
    particle : Particle
        The Particle which creates the Trace.
    binsize : int
        Size of time bin in ms.
    """

    def __init__(self, particle: Particle, binsize: int):
        self.binsize = binsize
        data = particle.abstimes[:]

        binsize_ns = binsize * 1e6  # Convert ms to ns
        try:
            endbin = int(np.max(data) / binsize_ns)
        except ValueError:
            endbin = 0

        binned = np.zeros(endbin + 1, dtype=int)
        for step in range(endbin):
            binned[step + 1] = np.size(
                data[((step + 1) * binsize_ns > data) * (data > step * binsize_ns)]
            )
            if step == 0:
                binned[step] = binned[step + 1]

        # binned *= (1000 / 100)
        self.intdata = binned
        self.inttimes = np.array(range(0, binsize + (endbin * binsize), binsize))


class Histogram:
    """TCSPC histogram.

    This class represents histogrammed TCSPC arrival times (micro times)
    as well as multi-exponential fits thereof.

    Parameters
    ----------
    particle : Particle
        The parent Particle of this object.
    level : Level or List = None
        The possible parent level of this object.
    start_point : float = None
        Start point for lifetime fit.
    channel : bool = True
        Whether to use hardware channel width.
    trim_start : bool = False
        Whether to trim zeros at the start of the histogram.
    is_for_roi : bool = False
        Whether this histogram is from a trace ROI.
    """

    def __init__(
        self,
        particle: Particle,
        level: Union[Level, List[int]] = None,
        start_point: float = None,
        channel: bool = True,
        trim_start: bool = False,
        is_for_roi: bool = False,
    ):
        assert not (level is not None and is_for_roi), "ROI can't be used for a Level"
        self.is_for_roi = is_for_roi
        self.fitted_with_roi = None
        self.roi_region_used = None
        no_sort = False
        self._particle = particle
        self.level = level
        self.original_kwargs = {
            "start_point": start_point,
            "channel": channel,
            "trim_start": trim_start,
        }
        self.microtimes = None
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
        self.stds = None
        self.avtaustd = None
        self.chisq = None
        self.dw = None
        self.dw_bound = None
        self.decay_roi_start_ns = None
        self.decay_roi_end_ns = None
        self.num_photons_used = None

    def setup(
        self,
        level: Union[Level, List[int]] = None,
        start_point: float = None,
        channel: bool = True,
        trim_start: bool = False,
        use_roi: bool = False,
    ):
        """Set up the object.

        This method can be called to re-do setup steps without reinitializing.

        Arguments
        ---------
        level : Level or List = None
            The possible parent level of this object.
        start_point : float = None
            Start point for lifetime fit.
        channel : bool = True
            Whether to use hardware channel width.
        trim_start : bool = False
            Whether to trim zeros at the start of the histogram.
        use_roi : bool = False
            Whether this histogram is from a trace ROI.
        """
        no_sort = False
        if level is None:
            if not use_roi:
                self.microtimes = self._particle.microtimes[:]
            else:
                self.microtimes = self._particle.microtimes_roi
                self.roi_region_used = self._particle.roi_region
        elif type(level) is list:
            if not self._particle.has_groups:
                logger.error("Multiple levels provided, but has no groups")
                raise RuntimeError("Multiple levels provided, but has no groups")
            self.microtimes = np.array([])
            for ind in level:
                self.microtimes = np.append(
                    self.microtimes, self._particle.cpts.levels[ind].microtimes
                )
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
                tmin = sorted_micro[
                    np.searchsorted(sorted_micro, tmin)
                ]  # Make sure bins align with TCSPC bins
            tmax = sorted_micro[
                np.searchsorted(sorted_micro, tmax) - 1
            ]  # - 1  # Fix if max is end

            window = tmax - tmin
            numpoints = int(window // self._particle.channelwidth)

            t = np.arange(tmin, tmax, self._particle.channelwidth)

            self.decay, self.t = np.histogram(self.microtimes, bins=t)
            self.t = self.t[:-1]  # Remove last value so the arrays are the same size
            where_neg = np.where(self.t <= 0)
            self.t = np.delete(self.t, where_neg)
            self.decay = np.delete(self.decay, where_neg)

            assert len(self.t) == len(self.decay), (
                "Time series must be same length as decay " "histogram"
            )
            if start_point is None and trim_start:
                try:
                    self.decaystart = np.nonzero(self.decay)[0][0]
                except IndexError:  # Happens when there is a level with no photons
                    pass
                else:
                    if level is not None:
                        self.decay, self.t = start_at_value(
                            self.decay, self.t, neg_t=False, decaystart=self.decaystart
                        )
            else:
                self.decaystart = 0

            try:
                self.t -= self.t.min()
            except ValueError:
                dbg.p(
                    f"Histogram object of {self._particle.name} does not have a valid"
                    f" self.t attribute",
                    "Histogram",
                )

    def update_roi(self):
        """Rerun setup to update ROI."""
        self.setup(level=self.level, use_roi=True, **self.original_kwargs)

    @property
    def t(self):
        return self._t.copy()

    @t.setter
    def t(self, value):
        self._t = value

    def fit(
        self,
        numexp,
        tauparam,
        ampparam,
        shift,
        decaybg,
        irfbg,
        boundaries,
        addopt,
        irf,
        fwhm=None,
    ):
        """Fit a multiexponential decay to the histogram.

        This function mainly calls the relevant code from `tcspcfit`.

        Arguments
        ---------
        numexp : int
            Number of exponentials in fit function (1-3).
        tauparam : array_like
            Initial guess times (in ns). This is either in the format
            [tau1, tau2, ...] or [[tau1, min1, max1, fix1], [tau2, ...], ...].
            When the "fix" value is False, the min and max values are ignored.
        ampparam : array_like
            Initial guess amplitude. Format [amp1, amp2, ...] or [[amp1, fix1],
            [amp2, fix2], ...]
        shift : array_like
            Initial guess IRF shift. Either a float, or [shift, min, max, fix].
        decaybg : float
            Background value for decay. Will be estimated if not given.
        irfbg : float
            Background value for IRF. Will be estimated if not given.
        boundaries : list
            Start and end of fitting range as well as options for automatic
            determination of the parameters as used by `FluoFit.calculate_boundaries`.
        irf : ndarray
            Instrumental Response Function
        fwhm : float = None
            Full-width at half maximum of simulated irf. IRF is not simulated if fwhm is None.
        addopt : Dict = None
            Additional options for `scipy.optimize.curve_fit` (such as optimization parameters).
        """
        if addopt is not None:
            addopt = ast.literal_eval(addopt)

        self.numexp = numexp

        # TODO: debug option that would keep the fit object (not done normally to conserve memory)
        try:
            if numexp == 1:
                fit = tcspcfit.OneExp(
                    irf,
                    self.decay,
                    self.t,
                    self._particle.channelwidth,
                    tauparam,
                    None,
                    shift,
                    decaybg,
                    irfbg,
                    boundaries,
                    addopt,
                    fwhm=fwhm,
                )
            elif numexp == 2:
                fit = tcspcfit.TwoExp(
                    irf,
                    self.decay,
                    self.t,
                    self._particle.channelwidth,
                    tauparam,
                    ampparam,
                    shift,
                    decaybg,
                    irfbg,
                    boundaries,
                    addopt,
                    fwhm=fwhm,
                )
            elif numexp == 3:
                fit = tcspcfit.ThreeExp(
                    irf,
                    self.decay,
                    self.t,
                    self._particle.channelwidth,
                    tauparam,
                    ampparam,
                    shift,
                    decaybg,
                    irfbg,
                    boundaries,
                    addopt,
                    fwhm=fwhm,
                )
        except Exception as e:
            trace_string = ""
            for trace_part in traceback.format_tb(e.__traceback__):
                trace_string = trace_string + trace_part
            logger.error(e.args[0] + "\n\n" + trace_string[:-1])
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
            self.stds = fit.stds
            self.avtaustd = fit.avtaustd
            self.chisq = fit.chisq
            self.dw = fit.dw
            self.dw_bound = fit.dw_bound
            self.residuals = fit.residuals
            self.fitted = True
            if numexp == 1:
                self.avtau = self.tau
            else:
                self.avtau = sum(np.array(self.tau) * np.array(self.amp)) / sum(self.amp)
            self.decay_roi_start_ns = fit.startpoint * self._particle.channelwidth
            self.decay_roi_end_ns = fit.endpoint * self._particle.channelwidth
            self.num_photons_used = np.sum(fit.measured_not_normalized)

        return True

    def levelhist(self, level):
        #  TODO: Remove this function as it is never used
        levelobj = self._particle.levels[level]
        tmin = levelobj.microtimes[:].min()
        tmax = levelobj.microtimes[:].max()
        window = tmax - tmin
        numpoints = int(window // self._particle.channelwidth)
        t = np.linspace(0, window, numpoints)

        decay, t = np.histogram(levelobj.microtimes[:], bins=t)
        t = t[:-1]  # Remove last value so the arrays are the same size
        return decay, t

    @staticmethod
    def start_at_value(decay, t, neg_t=True, decaystart=None):
        """Helper method for setting decay startpoint."""
        if decaystart is None:
            decaystart = np.nonzero(decay)[0][0]
        if neg_t:
            t -= t[decaystart]
        t = t[decaystart:]
        decay = decay[decaystart:]
        return decay, t


class ParticleAllHists:
    """Class containing all Histograms from Particle.

    Parameters
    ----------
    particle : Particle
        The parent particle of this object.
    """

    def __init__(self, particle: Particle):
        self.part_uuid = particle.uuid
        self.numexp = None
        self.part_hist = particle.histogram

        self.has_level_hists = particle.has_levels
        self.level_hists = list()
        if particle.has_levels:
            for level in particle.cpts.levels:
                if hasattr(level, "histogram") and level.histogram is not None:
                    self.level_hists.append(level.histogram)

        self.has_group_hists = particle.has_groups
        self.group_hists = list()
        if particle.has_groups:
            for group in particle.groups:
                if group.histogram is None:
                    group.histogram = Histogram(
                        particle=particle,
                        level=group.lvls_inds,
                        start_point=particle.startpoint,
                    )
                self.group_hists.append(group.histogram)

    def fit_part_and_levels(
        self, channelwidth, start, end, fit_param: FittingParameters
    ):
        """Fit all Histograms in this object.

        Arguments
        ---------
        channelwidth : float
            TCSPC channelwidth (time step size) in ns.
        start : int
            Fitting startpoint in number of time steps.
        end : int
            Fitting endpoint in number of time steps.
        fit_param : FittingParameters
            Object containing fit parameters.
        """
        self.numexp = fit_param.numexp
        all_hists = [self.part_hist]
        all_hists.extend(self.level_hists)
        all_hists.extend(self.group_hists)
        shift = fit_param.shift[:-1] / channelwidth
        shiftfix = fit_param.shift[-1]
        shift = [*shift, shiftfix]
        boundaries = [start, end, fit_param.autostart, fit_param.autoend]

        for hist in all_hists:
            if hist.microtimes.size > 10:
                try:
                    if not hist.fit(
                        fit_param.numexp,
                        fit_param.tau,
                        fit_param.amp,
                        shift,
                        fit_param.decaybg,
                        fit_param.irfbg,
                        boundaries,
                        fit_param.addopt,
                        fit_param.irf,
                        fit_param.fwhm,
                    ):
                        pass  # fit unsuccessful
                except AttributeError:
                    logger.info("Level or trace not fitted. No decay.")


class RasterScan:
    """Class containing raster scan data.

    A raster scan is a 2D intensity scan used to visualize particles
    before measurement.

    Parameters
    ----------
    h5dataset : H5dataset
        The parent HDF5 dataset object.
    particle_num : int
        Number of particles connected to this raster scan.
    h5dataset_index : int
        Index of this raster scan in the dataset.
    particle_indexes : List[int]
        Dataset indices of the particles in this raster scan.
    """

    def __init__(
        self,
        h5dataset: H5dataset,
        particle_num: int,
        h5dataset_index: int,
        particle_indexes: List[int],
    ):
        self.h5dataset = h5dataset
        self.particle_num = particle_num
        self.h5dataset_index = h5dataset_index
        self.particle_indexes = particle_indexes
        self.integration_time = h5_fr.rs_integration_time(part_or_rs=self)
        self.pixel_per_line = h5_fr.rs_pixels_per_line(part_or_rs=self)
        self.range = h5_fr.rs_range(part_or_rs=self)
        self.x_start = h5_fr.rs_x_start(part_or_rs=self)
        self.y_start = h5_fr.rs_y_start(part_or_rs=self)

        self.x_axis_pos = np.linspace(
            self.x_start, self.x_start + self.range, self.pixel_per_line
        )
        self.y_axis_pos = np.linspace(
            self.y_start, self.y_start + self.range, self.pixel_per_line
        )

    @property
    def dataset(self) -> h5pickle.Dataset:
        if self.h5dataset.file is not None and self.h5dataset.file.__bool__() is True:
            return h5_fr.raster_scan(h5_fr.particle(self.particle_num, self.h5dataset))


class Spectra:
    """Class containing spectral data.

    Spectra are recorded as a time scan, with a certain integration time
    over which a single spectrum is recorded using a grating and CCD.

    Parameters
    ----------
    particle : Particle
        The parent particle object.
    """

    def __init__(self, particle: Particle):
        self._particle = particle

    @property
    def _has_spectra(self) -> bool:
        if self._particle.file is not None and self._particle.file.__bool__() is True:
            return h5_fr.has_spectra(particle=self._particle)

    @property
    def data(self) -> h5pickle.Dataset:
        if self._particle.file is not None and self._particle.file.__bool__() is True:
            return h5_fr.spectra(particle=self._particle) if self._has_spectra else None

    @property
    def wavelengths(self) -> np.ndarray:
        if self._particle.file is not None and self._particle.file.__bool__() is True:
            return (
                h5_fr.spectra_wavelengths(particle=self._particle)
                if self._has_spectra
                else None
            )

    @property
    def series_times(self) -> np.ndarray:
        if self._particle.file is not None and self._particle.file.__bool__() is True:
            return (
                h5_fr.spectra_abstimes(particle=self._particle)
                if self._has_spectra
                else None
            )
