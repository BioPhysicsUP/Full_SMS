"""Module for handling analysis of change points and creation of consequent levels.

Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)

Joshua Botha
University of Pretoria
2019
"""

__docformat__ = 'NumPy'

import os
from typing import Tuple, Optional

import numpy as np
from h5py import Dataset
from statsmodels.stats.weightstats import DescrStatsW

import dbg
import file_manager as fm
from my_logger import setup_logger

MIN_PHOTONS = 10
BURST_MIN_DWELL = 0.1  # Seconds
BURST_INT_SIGMA = 3

logger = setup_logger(__name__)


class ChangePoints:
    """ Contains all the attributes to describe the found change points in an analysed particles. """

    def __init__(self, particle, confidence=None, run_levels=None):
        """
        Creates an instance of ChangePoints for the particle object provided.

        If the confidence argument is given the change point analysis will be run.
        If run_levels is set to True the analysed change points will be used to define the resulting levels.

        Parameters
        ----------
        particle: smsh5.Particle
            An the parent Particle instance.
        confidence:
        run_levels
        """
        self._particle = particle
        self.uuid = self._particle.uuid
        # self.cpa_has_run = False
        self._cpa = ChangePointAnalysis(particle, confidence)
        self.has_burst = False
        self.burst_levels = None

        if run_levels is not None:
            self._run_levels = run_levels
        else:
            self._run_levels = False

        if confidence is not None:
            self.run_cpa()

    @property
    def has_levels(self):
        return self._cpa.has_levels

    @property
    def levels(self):
        return self._cpa.levels

    @property
    def num_levels(self):
        return self._cpa.num_levels

    @property
    def level_ints(self):
        return self._cpa.level_ints

    @property
    def level_dwelltimes(self):
        return self._cpa.level_dwelltimes

    @property
    def cpa_has_run(self):
        return self._cpa.has_run

    #  Confidence property
    ######################
    @property
    def confidence(self):
        return self._cpa.confidence

    @confidence.setter
    def confidence(self, confidence):
        self._cpa.confidence = confidence

    #  Change Point Indexes property
    ################################
    @property
    def cpt_inds(self):
        return self._cpa.cpt_inds

    @cpt_inds.setter
    def cpt_inds(self, cpt_inds):
        self._cpa.cpt_inds = cpt_inds

    #  Number of change points property
    ##############################
    @property
    def num_cpts(self):
        return self._cpa.num_cpts

    @num_cpts.setter
    def num_cpts(self, num_cpts):
        self._cpa.num_cpts = num_cpts

    #  Confidence regions property
    ##############################
    @property
    def conf_regions(self):
        return self._cpa.conf_regions

    @conf_regions.setter
    def conf_regions(self, conf_regions):
        self._cpa.cpt_inds = conf_regions

    #  Time uncertainty property
    ##############################
    # @property
    # def dt_uncertainty(self):
    #     if self.cpa_has_run:
    #         return self._cpa.dt_uncertainty
    #     else:
    #         return None
    #
    # @dt_uncertainty.setter
    # def dt_uncertainty(self, dt_uncertainty):
    #     if self.cpa_has_run:
    #         self._cpa.dt_uncertainty = dt_uncertainty

    def run_cpa(self, confidence=None, run_levels=None, end_time_s=None):
        """
        Run change point analysis.

        Performs the change point analysis on the parent particle object with the confidence
            either provided as an argument here or in the __init__ method.

        Parameters
        ----------
        confidence : Confidence level with which to resolve the change points with.
         Must be 0.99, 0.95, 0.90 or 0.69.
        run_levels : If true the change point analysis will be used to add a list of levels to the
         parent particle object by running its add_levels method.
        end_time_s
        """

        if run_levels is not None:
            self._run_levels = run_levels
        if confidence is not None:
            if confidence in [69, 90, 95, 99]:
                confidence = confidence / 100
            self.confidence = confidence
        assert self.confidence is not None, "ChangePoint\tConfidence not set, can not run cpa"

        if self.cpa_has_run:
            self._cpa.reset(confidence)
        self._cpa.run_cpa(confidence, end_time_s=end_time_s)
        if self._run_levels:
            self._cpa.define_levels()
            if self.has_levels:
                self.calc_mean_std()  # self.level_ints
                self.check_burst()  # self.level_ints, self.level_dwelltimes
        logger.info(msg=f"{self._particle.name} levels resolved")

    def calc_mean_std(self):  # , intensities: np.ndarray = None
        assert self.has_levels, "ChangePoints\tNeeds to have levels to calculate mean and standard deviation."
        # num_levels = self._particle.num_levels
        # if intensities is None:
        #     intensities = np.array([level.int_p_s for level in self._particle.levels])
        dwell_weights = np.array(self.level_dwelltimes) / np.sum(self.level_dwelltimes)
        weighted_stats = DescrStatsW(self.level_ints, dwell_weights)
        self._particle.avg_int_weighted = weighted_stats.mean
        self._particle.int_std_weighted = weighted_stats.std

    def check_burst(self):  # , intensities: np.ndarray = None, dwell_times: list = None
        assert self._particle.has_levels, "ChangePoints\tNeeds to have levels to check photon bursts."
        # if intensities is None or dwell_times is None:
            # intensities, dwell_times = np.array([(level.int_p_s, level.dwell_time_s) for level in self._particle.levels])
        burst_def = self._particle.avg_int_weighted + (self._particle.int_std_weighted * BURST_INT_SIGMA)
        burst_bools = np.logical_and(self.level_ints > burst_def, np.array(self.level_dwelltimes) < BURST_MIN_DWELL)
        burst_levels = np.where(burst_bools)[0]
        if len(burst_levels):
            self.has_burst = True
            self.burst_levels = burst_levels

    def remove_bursts(self):
        assert self.has_burst, "Particle\tNo bursts to remove."
        for burst_ind in np.flip(self.burst_levels):
            # print(burst_ind)
            merge_left = bool()
            if burst_ind == self.num_levels - 1:
                merge_left = True
            elif burst_ind == 0:
                merge_left = False
            elif self.level_ints[burst_ind + 1] > self.level_ints[burst_ind - 1]:
                merge_left = False
            else:
                merge_left = True

            merge_ind = 0
            if merge_left:
                del_ind = burst_ind - 1
            else:
                del_ind = burst_ind

            self.cpt_inds = np.delete(self.cpt_inds, del_ind)
            del(self.conf_regions[del_ind])

        self.num_cpts = len(self.cpt_inds)
        self._cpa.define_levels(remove_prev=True)
        # self.cpts.num_cpts
        # self.cpts.inds
        # self.cpts.


class Err:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
        self.sum = lower + upper


class Level:
    """ Defines the start, end and intensity of a single level. """

    def __init__(self, abs_times: Dataset, microtimes: Dataset,
                 level_inds: Tuple[int, int], int_p_s:float = None):
                # conf_regions: Any[List[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]
        """
        Initiate Level

        Initiates attributes that define a single resolved level. These attributes
        are the start and end indexes, the start and end time in ns, the number of
        photons in the level, the dwell time and the intensity.

        :param abs_times: Dataset of absolute arrival times (ns) in a h5 file as read by h5py.
        :type abs_times: HDF5 Dataset
        :param level_inds: A tuples that contain the start and end of the level (ns).
        :type level_inds: tuple
        """

        assert abs_times is not None, "Levels:\tParameter 'abstimes' not given."
        assert level_inds is not None, "Levels:\tParameter 'level_inds' not given."
        assert type(level_inds) is tuple, "Level:\tLevel indexes argument is not a tuple (start, end)."
        self.level_inds = level_inds  # (first_ind, last_ind)
        self.num_photons = self.level_inds[1] - self.level_inds[0] + 1
        self.times_ns = (abs_times[self.level_inds[0]], abs_times[self.level_inds[1]])
        self.dwell_time_ns = self.times_ns[1] - self.times_ns[0]
        if int_p_s:
            self.int_p_s = int_p_s
        else:
            self.int_p_s = self.num_photons / self.dwell_time_s
        self.microtimes = microtimes[self.level_inds[0]:self.level_inds[1]]

        # TODO: Incorporate error margins
        # conf_ind_lower = conf_regions[0]
        # conf_ind_upper = conf_regions[1]
        #
        # dwell_err_lower = abs_times[conf_ind_lower] - self.times_ns[0]
        # dwell_err_upper = abs_times[conf_ind_upper] - self.times_ns[0]
        # self.dwell_err_ns = Err(lower=dwell_err_lower, upper=dwell_err_upper)
        #
        # num_ph_err_lower = conf_ind_lower - self.level_inds[0]  # + 1
        # num_ph_err_upper = conf_ind_upper - self.level_inds[0]  # + 1
        # int_err_lower = self.int_p_s - (1E9 * num_ph_err_lower / dwell_err_lower)
        # int_err_upper = self.int_p_s - (1E9 * num_ph_err_upper / dwell_err_upper)
        # self.int_err_p_s = Err(lower=int_err_lower, upper=int_err_upper)

    @property
    def times_s(self):
        return (self.times_ns[0]/1E9, self.times_ns[1]/1E9)

    @property
    def dwell_time_s(self):
        if self.dwell_time_ns is not None:
            return self.dwell_time_ns / 1e9
        else:
            return None


class TauData:
    """ Loads and stores the tau_a and tau_b files from text files for
        specific confidence and stores as attributes. """

    def __init__(self, confidence=None):
        """
        Initialise TauData instance.

        Reads local tau_a and tau_b text files and stores data in attributes for a specific confidence interval.
        :param confidence: Confidence of tau_a and tau_b to retrieve. Valid values are 0.99, 0.95, 0.90 and 0.69.
        :type confidence: float
        """
        try:
            tau_data_path = fm.folder_path(folder_name='tau_data', resource_type=fm.Type.Data)
            assert os.path.isdir(tau_data_path), "TauData:\tTau data directory not found."
            tau_data_files = {'99_a': 'Ta-99.txt',
                              '99_b': 'Tb-99.txt',
                              '95_a': 'Ta-95.txt',
                              '95_b': 'Tb-95.txt',
                              '90_a': 'Ta-90.txt',
                              '90_b': 'Tb-90.txt',
                              '69_a': 'Ta-69.txt',
                              '69_b': 'Tb-69.txt'}

            assert confidence in [0.99, 0.95, 0.90, 0.69],\
                "TauData:\tInvalid confidence provided. Can not provide confidence key."
            if confidence == 0.99:
                conf_keys = ['99_a', '99_b']
            elif confidence == 0.95:
                conf_keys = ['95_a', '95_b']
            elif confidence == 0.90:
                conf_keys = ['90_a', '90_b']
            else:
                conf_keys = ['69_a', '69_b']

            for tau_type in conf_keys:
                file_name = tau_data_files[tau_type]
                full_path = tau_data_path + os.path.sep + file_name
                assert os.path.isfile(full_path),\
                    'TauData:\tTau data file ' + file_name + ' does not exist. Look in' +\
                    full_path + '.'

                data = np.loadtxt(full_path, usecols=1)
                if tau_type[-1] == 'a':
                    self._a = data
                else:
                    self._b = data
        except Exception as err:
            logger.error(err)

    def get_tau_a(self, num_data_points=None):
        """
        Get tau_a value for n = num_data_points.

        Retrieve the a tau data for the given number of data points.

        :param num_data_points: Number of data points that the tau data is needed for.
            **Note**, only use values up to and smaller than 1000 for accuracy.
        :return: tau_a value
        :rtype: float
        """

        assert num_data_points is not None, "TauData:\tNumber of data points not given."
        return self._a[num_data_points]

    def get_tau_b(self, num_data_points=None):
        """
        Get tau_a value for n = num_data_points.

        Retrieve the a tau data for the given number of data points.

        :param num_data_points: Number of data points that the tau data is needed for.
            **Note**, only use values up to and smaller than 1000 for accuracy.
        :return: tau_a value
        :rtype: float
        """

        assert num_data_points is not None, "TauData:\tNumber of data points not given."
        return self._b[num_data_points]


class ChangePointAnalysis:
    """ Perform analysis of particle abstimes data to resolve change points. """

    def __init__(self, particle=None, confidence=None):
        """
        Initiate ChangePointAnalysis instance.
        :param particle: Object containing particle data
        :type particle: Class Particle in smsh5 module
        :param confidence: Confidence interval. Valid values are 0.99, 0.95, 0.90 and 0.69.
        :type confidence: float
        """
        # assert h5file is not None, "No HDF5 has been given"  # To ensure that a h5file is given
        # assert type(particle) is smsh5.Particle, "ChangePoints:\tNo Particle object given."
        # assert confidence is not None, "ChangePoints:\tNo confidence parameter given."

        self._particle = particle
        self._abstimes = particle.abstimes
        self._microtimes = particle.microtimes
        self.has_run = False
        self.end_at_photon = None
        self.num_photons = particle.num_photons
        self.confidence = confidence
        self.cpt_inds = np.array([], dtype=int)
        self.conf_regions = list(tuple())  # [(start, end)]
        # self.dt_uncertainty = np.array([])  # dt
        self._finding = False
        self.found_cpts = False
        self.num_cpts = None
        self.has_levels = False
        self.levels = None
        self.num_levels = None
        self._i = None
        if confidence is not None:
            self._tau = TauData(self.confidence)

    def reset(self, confidence: float = None):
        self.has_run = False
        self.confidence = confidence
        self.cpt_inds = np.array([], dtype=int)
        self.conf_regions = list(tuple())  # [(start, end)]
        # self.dt_uncertainty = np.array([])  # dt
        self.has_levels = False
        self.levels = None
        self.num_levels = None
        self._finding = False
        self.found_cpts = False
        self.num_cpts = None
        self._i = None
        if confidence is not None:
            self._tau = TauData(self.confidence)

    @property
    def level_ints(self):
        if self.has_run:
            return np.array([level.int_p_s for level in self.levels])
        else:
            return None

    @property
    def level_dwelltimes(self):
        if self.has_run:
            return [level.dwell_time_s for level in self.levels]
        else:
            return None


    def __weighted_likelihood_ratio(self, seg_inds=None) -> Tuple[bool, Optional[int]]:
        """
        Calculates the Weighted & Standardised Likelihood ratio and detects the possible change point.

        Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
        from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)
        
        If the possible change point is greater than the tau_a value for the corresponding
        confidence interval and number of data points the detected change points, it's
        confidence region (as defined by tau_b), and the corresponding uncertainty in time
        is added to this instance of ChangePointAnalysis.
        
        Parameters
        ----------
        seg_inds : (int, int), optional
            Segment indexes (start, end).
            
        Returns
        -------
        cpt_found : bool
            True if a change point was detected.
        cpt : int, Optional
            The index of the change point, if one was detected.
        """

        assert type(seg_inds) is tuple, 'ChangePointAnalysis:\tSegment index\'s not given.'
        start_ind, end_ind = seg_inds
        n = end_ind - start_ind
        assert n <= 1000, "ChangePointAnalysis:\tIndex's given result in more than a segment of more than 1000 points."
        if n < MIN_PHOTONS:
            cpt_found = False
            return cpt_found, None
        time_data = self._abstimes[start_ind:end_ind]

        ini_time = time_data[0]
        period = time_data[-1] - ini_time

        wlr = np.zeros(n, float)

        # sig_e = np.pi ** 2 / 6 - sum(1 / j ** 2 for j in range(1, (n - 1) + 1))
        sig_e = self._particle.dataset.all_sums.get_sig_e(n)

        for k in range(2, (n - 2) + 1):  # Remember!!!! range(1, N) = [1, ... , N-1]
            sum_set = self._particle.dataset.all_sums.get_set(n, k)

            cap_v_k = (time_data[k] - ini_time) / period  # Just after eq. 4

            # u_k = -sum(1 / j for j in range(k, (n - 1) + 1))  # Just after eq. 6
            u_k = sum_set['u_k']

            # u_n_k = -sum(1 / j for j in range(n - k, (n - 1) + 1))  # Just after eq. 6
            u_n_k = sum_set['u_n_k']

            l0_minus_expec_l0 = -2 * k * np.log(cap_v_k) + 2 * k * u_k - 2 * (n - k) * np.log(1 - cap_v_k) + 2 * (
                    n - k) * u_n_k  # Just after eq. 6

            # v_k2 = sum(1 / j ** 2 for j in range(k, (n - 1) + 1))  # Just before eq. 7
            v2_k = sum_set['v2_k']

            # v2_n_k = sum(1 / j ** 2 for j in range(n - k, (n - 1) + 1))  # Just before eq. 7
            v2_n_k = sum_set['v2_n_k']

            sigma_k = np.sqrt(
                4 * (k ** 2) * v2_k + 4 * ((n - k) ** 2) * v2_n_k - 8 * k * (n - k) * sig_e)  # Just before eq. 7, and note errata
            w_k = (1 / 2) * np.log((4 * k * (n - k)) / n ** 2)  # Just after eq. 6

            wlr.itemset(k, l0_minus_expec_l0 / sigma_k + w_k)  # Eq. 6 and just after eq. 6

        max_ind_local = int(wlr.argmax())

        cpt = None
        if wlr[max_ind_local] >= self._tau.get_tau_a(n):
            cpt = max_ind_local + start_ind
            self.cpt_inds = np.append(self.cpt_inds, max_ind_local + start_ind)
            tau_b_inv = wlr.max() - self._tau.get_tau_b(n)
            region_all_local = np.where(wlr >= tau_b_inv)[0]
            region = (region_all_local[0] + start_ind, region_all_local[-1] + start_ind)
            dt = self._abstimes[region[1]] - self._abstimes[region[0]]
            self.conf_regions.append(region)  # list, not ndarray
            # self.dt_uncertainty = np.append(self.dt_uncertainty, dt)
            cpt_found = True
        else:
            cpt_found = False

        return cpt_found, cpt

    def _next_seg_ind(self,
                      prev_seg_inds: Tuple[int, int] = None,
                      side: str = None,
                      rights_cpt: int = None) -> Tuple[int, int]:
        """
        Calculates the next segments indexes.

        Uses the indexes of the previous segment, as well as the latest change point
        to calculate the index values of the next segment.

        See code2flow.com for flow diagram of if statements.
        https://code2flow.com/svLn85

        Parameters
        ----------
        prev_seg_inds : (int, int)
            Contains the start and end of the previous segment (start, end)
        side: str, Optional
            If a change point was detected in the previous segment choose left
            or right of it. Possible values are 'left' or 'right'.
        rights_cpt : int, Optional
            The index of the change point for the right leg (the next_seg_start if side = 'left').
        Returns
        -------
        next_seg : (int, int)
            The next segments indexes.
        """

        if self.end_at_photon is not None:
            last_photon_ind = self.end_at_photon
        else:
            last_photon_ind = self.num_photons - 1

        if prev_seg_inds is None:
            # Data sets need to be larger than 200 photons
            assert self.num_photons >= 200, 'ChangePointAnalysis:\tData set needs to ' \
                                            'be at least 200 photons for change point detection.'
            if self.num_photons > 1000:
                next_start_ind, next_end_ind = 0, 1000
            else:
                next_start_ind, next_end_ind = 0, last_photon_ind
        else:
            prev_start_ind, prev_end_ind = prev_seg_inds
            if len(self.cpt_inds) == 0:
                if last_photon_ind >= prev_end_ind + 800:
                    next_start_ind, next_end_ind = prev_end_ind - 200, prev_end_ind + 800
                elif last_photon_ind - prev_end_ind >= 10:  # Next segment needs to be at least 10 photons large.
                    next_start_ind, next_end_ind = prev_end_ind - 200, last_photon_ind
                else:
                    next_start_ind, next_end_ind = None, None
                    dbg.p("Warning, last photon segment smaller than 10 photons and was not tested", "Change Point")
            elif side is not None:  # or prev_start_ind < self.cpts[-1] < prev_end_ind
                if side is not None:
                    assert side in ['left',
                                    'right'], "ChangePointAnalysis:\tSide of change point invalid or not specified"
                if side == 'left':
                    next_start_ind, next_end_ind = prev_start_ind, int(self.cpt_inds[-1] - 1)
                else:
                    assert rights_cpt is not None, "ChangePointAnalysis\tRight side's change point not provided."
                    next_start_ind = rights_cpt
                    # if len(self.cpt_inds) > 1:
                    #     i = -1
                    #     while self.cpt_inds[i - 1] > self.cpt_inds[i]:
                    #         i -= 1
                    #         if self.cpt_inds[i] == prev_end_ind + 1:
                    #             break
                    #         next_start_ind = self.cpt_inds[i]
                    next_end_ind = prev_end_ind
            elif last_photon_ind >= prev_end_ind + 800:
                if prev_end_ind - 200 < self.cpt_inds[-1] < prev_end_ind:
                    if last_photon_ind >= self.cpt_inds[-1] + 1000:
                        next_start_ind, next_end_ind = int(self.cpt_inds[-1]), int(self.cpt_inds[-1]) + 1000
                    else:
                        next_start_ind, next_end_ind = int(self.cpt_inds[-1]), last_photon_ind
                else:
                    next_start_ind, next_end_ind = prev_end_ind - 200, prev_end_ind + 800
            elif last_photon_ind - prev_end_ind >= 10:  # Next segment needs to be at least 10 photons large.
                if prev_end_ind - 200 < self.cpt_inds[-1] < prev_end_ind:
                    next_start_ind, next_end_ind = int(self.cpt_inds[-1]) + 1, last_photon_ind
                else:
                    next_start_ind, next_end_ind = prev_end_ind - 200, last_photon_ind
            else:
                next_start_ind, next_end_ind = None, None
                if prev_end_ind != last_photon_ind:
                    dbg.p("Warning, last photon segment smaller than 10 photons and was not tested", "Change Point")

        return next_start_ind, next_end_ind

    def _find_all_cpts(self,
                       _seg_inds: Tuple[int, int] = None,
                       _side: str = None,
                       _right_cpt: int = None):
        """
        Find all change points in particle.

        Recursive function that finds all change points that meets the confidence criteria.

        .. note::
            The first call doesn't need to be called with any parameters.

        .. note::
            The top level assigns the number of detected change points to
            the .num_cpts attribute of this instance of ChangePointAnalysis.

        Parameters
        ----------
        _right_cpt : int, Optional
            The index of the change point for the right leg (the next_seg_start if side = 'left').
        _seg_inds : (int, int), Optional
            The index of the segment that is to be searched. Calculated by _next_seg_ind method.
        _side : str, Optional
            Determines current segment is left or right of a previously
            detected change point. Valid values are 'left' or 'right'.
        """
        is_top_level = False

        if self._finding is False:
            is_top_level = True
            self._finding = True
            assert _seg_inds is None, "ChangePointAnalysis:\tDo not provide seg_inds when calling, it's used for " \
                                      "recursive calling only. "

        if is_top_level:
            _seg_inds = self._next_seg_ind()
            self._i = 0
        else:
            _seg_inds = self._next_seg_ind(prev_seg_inds=_seg_inds, side=_side, rights_cpt=_right_cpt)
        self._i += 1

        if _seg_inds != (None, None):
            cpt_found, _right_cpt = self.__weighted_likelihood_ratio(_seg_inds)

            if cpt_found:
                self._find_all_cpts(_seg_inds, _side='left')  # Left side of change point
                self._find_all_cpts(_seg_inds, _side='right', _right_cpt=_right_cpt)  # Right side of change point
                pass  # Exits if recursive

            if self.end_at_photon is not None:
                end_ind = self.end_at_photon
            else:
                end_ind = self.num_photons

            if _seg_inds[1] <= end_ind + 9 and _side is None:
                self._find_all_cpts(_seg_inds)

        if is_top_level:
            self._finding = False
            self.found_cpts = True

            sort_inds = np.argsort(self.cpt_inds)
            cpt_inds = np.zeros_like(self.cpt_inds)
            conf_regions = list()
            # dt_uncertainty = np.zeros_like(cpt_inds)
            for i, sort_i in enumerate(sort_inds):
                cpt_inds[i] = self.cpt_inds[sort_i]
                conf_regions.append(self.conf_regions[sort_i])
                # dt_uncertainty[i] = self.dt_uncertainty[sort_i]

            # TODO: Revisit necessity of duplicate removal
            cpt_inds, unique_inds = np.unique(cpt_inds, return_inverse=True)
            dups = np.append([False], [unique_inds[i] == unique_inds[i - 1] for i in range(len(unique_inds)) if i > 0])
            dups = np.flip(np.where(dups)[0].tolist())
            if len(dups) != 0:
                for i in dups:
                    del (conf_regions[i])

            self.num_cpts = len(cpt_inds)
            self.cpt_inds = cpt_inds
            self.conf_regions = conf_regions

    def define_levels(self, remove_prev: bool = None) -> None:
        """
        Creates a list of levels as defined by the detected change points.

        Uses the detected change points to create a list of Level instances
        that contain attributes that define each resolved level.

        This method populate the .levels attribute of the parent particle
        instance by using its add_levels method.

        Parameters
        ----------
        remove_prev : bool, Optional
            If true, previous levels will be removed.
        """

        if len(self.cpt_inds) != 0:
            if remove_prev is None:
                remove_prev = False

            if remove_prev:
                self.levels = None
                self.num_levels = None

            assert self.found_cpts, "ChangePointAnalysis:\tChange point analysis " \
                                    "not done, or found no change points. "
            self.num_levels = self.num_cpts + 1
            self.levels = [object()] * self.num_levels

            if self.num_cpts != 1:
                for num, cpt in enumerate(self.cpt_inds):
                    if num == 0:  # First change point
                        level_inds = (0, cpt - 1)
                    elif num == self.num_cpts - 1:  # Last change point
                        if self.end_at_photon is not None:
                            end_ind = self.end_at_photon
                        else:
                            end_ind = self.num_photons
                        level_inds = (cpt, end_ind - 1)
                        self.levels[num + 1] = Level(self._abstimes, self._microtimes,
                                                     level_inds=level_inds)

                        level_inds = (self.cpt_inds[num - 1], cpt - 1)
                    else:
                        level_inds = (self.cpt_inds[num - 1], cpt)

                    self.levels[num] = Level(self._abstimes, self._microtimes, level_inds=level_inds)
            else:
                self.levels[0] = Level(self._abstimes, self._microtimes,
                                       level_inds=(0, self.cpt_inds[0] - 1))
                self.levels[1] = Level(self._abstimes, self._microtimes,
                                       level_inds=(self.cpt_inds[0], self.num_photons - 1))

            self.has_levels = True

    # @dbg.profile
    def run_cpa(self, confidence=None, end_time_s = None):
        """
        Runs the change point analysis.

        If the ChangePointAnalysis wasn't initialised with a confidence interval, or if
        the analysis is to be rerun with a new confidence interval, this method starts said analysis.

        Parameters
        ----------
        end_time_s: float
            Time at which to end analysis. If not provided the whole trace will be used.
        confidence: float
            Confidence interval. Valid values are 0.99, 0.95, 0.90 and 0.69.

        Returns
        -------
        num_cpts: int
            Number of change points detected
        cpt_inds: ndarray
            Indexes of change points
        conf_regions: list(tuple(int, int))
            Index region corresponding to confidence interval
        dt_uncertainty: ndarray
            Array of uncertainty in time corresonding to confidence interval
        """
        if confidence is not None:
            assert confidence in [0.99, 0.95, 0.90, 0.69], "ChangePointAnalysis:\tConfidence value given not valid."
            self.confidence = confidence
            self._tau = TauData(confidence)
        else:
            assert self.confidence is not None, "ChangePointAnalysis:\tNo confidence value provided."

        if end_time_s is not None:
            self.end_at_photon = np.argmax(self._abstimes[:] > (end_time_s * 1E9))
            if self.end_at_photon == 0:
                self.end_at_photon = self.num_photons
        self._find_all_cpts()
        self.has_run = True


def main():
    """
    Tests ChangePoints init
    """

    pass


if __name__ == '__main__':
    main()
