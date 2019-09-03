"""Module for handling analysis of change points and creation of consequent levels.

Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)

Joshua Botha
University of Pretoria
2019
"""

__docformat__ = 'NumPy'

import os
import numpy as np
import dbg
from PyQt5.QtCore import pyqtSignal


# from smsh5 import Particle

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
        self.cpa = ChangePointAnalysis(particle)

        if run_levels is not None:
            self._run_levels = run_levels
        else:
            self._run_levels = False

        if confidence is not None:
            self.run_cpa()
        else:
            self.confidence = None
            self.inds = None
            self.conf_regions = None
            self.dt_uncertainty = None
            self.cpa_has_run = False
            self.num_cpts = None

    def run_cpa(self, confidence=None, run_levels=None):
        """
        Run change point analysis.

        Performs the change point analysis on the parent particle object with the confidence
            either provided as an argument here or in the __init__ method.

        :param confidence: Confidence level with which to resolve the
                change points with. Must be 0.99, 0.95, 0.90 or 0.69.
        :type confidence: int, optional
        :param run_levels: If true the change point analysis will be used to
                add a list of levels to the parent particle object by running its add_levels method.
        :type run_levels: bool, optional
        """

        if run_levels is not None:
            self._run_levels = run_levels
        if confidence > 1:
            confidence = confidence/100
        self.confidence = confidence
        if self.cpa_has_run:
            self.remove_cpa_results()
            self.cpa.prerun_setup(confidence)
        self.cpa.run_cpa(confidence)  # self.inds, self.conf_regions, self.dt_uncertainty =
        self.num_cpts = self.cpa.num_cpts
        self.cpa_has_run = True
        if self._run_levels:
            self.get_levels()

    def get_levels(self):
        """ Uses the resolved change points to define the resulting levels and adds
            them to the parent particle object by means of the add_levels method. """

        self.cpa.get_levels()

    def remove_cpa_results(self):
        self._particle.remove_cpa_results()
        self.inds = None
        self.conf_regions = None
        self.dt_uncertainty = None
        self.cpa_has_run = False
        self.num_cpts = None


class Level:
    """ Defines the start, end and intensity of a single level. """

    def __init__(self, abstimes=None, level_inds=None, microtimes=None):
        """
        Initiate Level

        Initiates attributes that define a single resolved level. These attributes
        are the start and end indexes, the start and end time in ns, the number of
        photons in the level, the dwell time and the intensity.

        :param abstimes: Dataset of absolute arrival times (ns) in a h5 file as read by h5py.
        :type abstimes: HDF5 Dataset
        :param level_inds: A tuples that contain the start and end of the level (ns).
        :type level_inds: tuple
        """
        # assert h5file is not None, "No HDF5 has been given"  # To ensure that a h5file is given
        # assert type(particle) is smsh5.Particle, "Level:\tNo Particle object given."
        # self.particle = smsh5.Particle.__copy__(particle)
        assert abstimes is not None, "Levels:\tParameter 'abstimes' not given."
        assert level_inds is not None, "Levels:\tParameter 'level_inds' not given."
        assert type(level_inds) is tuple, "Level:\tLevel indexes argument is not a tuple (start, end)."
        self.level_inds = level_inds  # (first_ind, last_ind)
        self.num_photons = self.level_inds[1]-self.level_inds[0]+1
        self.times = (abstimes[self.level_inds[0]], abstimes[self.level_inds[1]])
        self.microtimes = microtimes[self.level_inds[0]:self.level_inds[1]]
        self.dwell_time = self.times[1]-self.times[0]
        self.int = self.num_photons/(self.dwell_time/1e9)


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
        tau_data_path = os.getcwd()+os.path.sep+'tau data'
        assert os.path.isdir(tau_data_path), "TauData:\tTau data directory not found."
        tau_data_files = {'99_a': 'Ta-99.txt',
                          '99_b': 'Tb-99.txt',
                          '95_a': 'Ta-95.txt',
                          '95_b': 'Tb-95.txt',
                          '90_a': 'Ta-90.txt',
                          '90_b': 'Tb-90.txt',
                          '69_a': 'Ta-69.txt',
                          '69_b': 'Tb-69.txt'}

        assert confidence in [0.99, 0.95, 0.90,
                              0.69], "TauData:\tInvalid confidence provided. Can not provide confidence key."
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
            full_path = tau_data_path+os.path.sep+file_name
            assert os.path.isfile(
                full_path), 'TauData:\tTau data file '+file_name+' does not exist. Look in'+full_path+'.'
            data = np.loadtxt(full_path, usecols=1)
            if tau_type[-1] is 'a':
                self._a = data
            else:
                self._b = data

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
        :param particle: Object contaning particle data
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
        self.num_photons = particle.num_photons
        self.prerun_setup()

    def prerun_setup(self, confidence: float = None):
        self.confidence = confidence
        self.cpt_inds = np.array([], dtype=int)
        self.conf_regions = np.array(tuple())  # [(start, end)]
        self.dt_uncertainty = np.array([])  # dt
        self._finding = False
        self.found_cpts = False
        self.num_cpts = None
        self._i = None
        if confidence is not None:
            self._tau = TauData(self.confidence)

    def __weighted_likelihood_ratio(self, seg_inds=None) -> bool:
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
        """

        assert type(seg_inds) is tuple, 'ChangePointAnalysis:\tSegment index\'s not given.'
        start_ind, end_ind = seg_inds
        n = end_ind-start_ind
        assert n <= 1000, "ChangePointAnalysis:\tIndex's given result in more than a segment of more than 1000 points."
        # print(start_ind, end_ind)
        time_data = self._abstimes[start_ind:end_ind]

        ini_time = time_data[0]
        period = time_data[-1]-ini_time

        wlr = np.zeros(n, float)

        # sig_e = np.pi ** 2 / 6 - sum(1 / j ** 2 for j in range(1, (n - 1) + 1))
        sig_e = self._particle.dataset.all_sums.get_sig_e(n)

        for k in range(2, (n-2)+1):  # Remember!!!! range(1, N) = [1, ... , N-1]
            sum_set = self._particle.dataset.all_sums.get_set(n, k)

            cap_v_k = (time_data[k]-ini_time)/period  # Just after eq. 4

            # u_k = -sum(1 / j for j in range(k, (n - 1) + 1))  # Just after eq. 6
            u_k = sum_set['u_k']

            # u_n_k = -sum(1 / j for j in range(n - k, (n - 1) + 1))  # Just after eq. 6
            u_n_k = sum_set['u_n_k']

            l0_minus_expec_l0 = -2*k*np.log(cap_v_k)+2*k*u_k-2*(n-k)*np.log(1-cap_v_k)+2*(
                    n-k)*u_n_k  # Just after eq. 6

            # v_k2 = sum(1 / j ** 2 for j in range(k, (n - 1) + 1))  # Just before eq. 7
            v2_k = sum_set['v2_k']

            # v2_n_k = sum(1 / j ** 2 for j in range(n - k, (n - 1) + 1))  # Just before eq. 7
            v2_n_k = sum_set['v2_n_k']

            sigma_k = np.sqrt(4*(k**2)*v2_k+4*((n-k)**2)*v2_n_k-8*k*(n-k)*sig_e)  # Just before eq. 7, and note errata
            w_k = (1/2)*np.log((4*k*(n-k))/n**2)  # Just after eq. 6

            wlr.itemset(k, l0_minus_expec_l0/sigma_k+w_k)  # Eq. 6 and just after eq. 6

        max_ind_local = int(wlr.argmax())

        if wlr[max_ind_local] >= self._tau.get_tau_a(n):
            self.cpt_inds = np.append(self.cpt_inds, max_ind_local+start_ind)
            region_all_local = np.where(wlr >= self._tau.get_tau_b(n))[0]
            region_local = [region_all_local[0], region_all_local[-1]]
            region = (region_local[0]+start_ind, region_local[1]+start_ind)
            dt = self._abstimes[region_local[1]]-self._abstimes[region_local[0]]
            self.conf_regions = np.append(self.conf_regions, region)
            self.dt_uncertainty = np.append(self.dt_uncertainty, dt)
            cpt_found = True
        else:
            cpt_found = False
        return cpt_found

    def _next_seg_ind(self, prev_seg_inds=None, side=None):
        """
        Calculates the next segments indexes.

        Uses the indexes of the previous segment, as well as the latest change point
        to calculate the index values of the next segment.

        .. seealso::
            See code2flow.com for flow diagram of if statements.
            https://code2flow.com/svLn85

        :param prev_seg_inds: Contains the start and end of the previous segment (start, end)
        :type prev_seg_inds:
        :param side: If a change point was detected in the previous segment choose left
            or right of it. Possible values are 'left' or 'right'.
        :type side: str, optional
        :return: Returns the calculated indexes of the next segment (start, end)
        :rtype: (int, int)
        """

        last_photon_ind = self.num_photons-1

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
                if last_photon_ind >= prev_end_ind+800:
                    next_start_ind, next_end_ind = prev_end_ind-200, prev_end_ind+800
                elif last_photon_ind-prev_end_ind >= 10:  # Next segment needs to be at least 10 photons large.
                    next_start_ind, next_end_ind = prev_end_ind-200, last_photon_ind
                else:
                    next_start_ind, next_end_ind = None, None
                    dbg.p("Warning, last photon segment smaller than 10 photons and was not tested", "Change Point")
            elif side is not None:  # or prev_start_ind < self.cpts[-1] < prev_end_ind
                if side is not None:
                    assert side in ['left',
                                    'right'], "ChangePointAnalysis:\tSide of change point invalid or not specified"
                if side == 'left':
                    next_start_ind, next_end_ind = prev_start_ind, int(self.cpt_inds[-1]-1)
                else:
                    next_start_ind, next_end_ind = int(self.cpt_inds[-1]), prev_end_ind
            elif last_photon_ind >= prev_end_ind+800:
                if prev_end_ind-200 < self.cpt_inds[-1] < prev_end_ind:
                    if last_photon_ind >= self.cpt_inds[-1]+1000:
                        next_start_ind, next_end_ind = int(self.cpt_inds[-1]), int(self.cpt_inds[-1])+1000
                    else:
                        next_start_ind, next_end_ind = int(self.cpt_inds[-1]), last_photon_ind
                else:
                    next_start_ind, next_end_ind = prev_end_ind-200, prev_end_ind+800
            elif last_photon_ind-prev_end_ind >= 10:  # Next segment needs to be at least 10 photons large.
                if prev_end_ind-200 < self.cpt_inds[-1] < prev_end_ind:
                    next_start_ind, next_end_ind = int(self.cpt_inds[-1])+1, last_photon_ind
                else:
                    next_start_ind, next_end_ind = prev_end_ind-200, last_photon_ind
            else:
                next_start_ind, next_end_ind = None, None
                if prev_end_ind != last_photon_ind:
                    dbg.p("Warning, last photon segment smaller than 10 photons and was not tested", "Change Point")

        return next_start_ind, next_end_ind

    def _find_all_cpts(self, _seg_inds: (int, int) = None, _side: str = None) -> None:
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
        _seg_inds : (int, int), optional
            The index of the segment that is to be searched. Calculated by _next_seg_ind method.
        _side : str, optional
            Determines current segment is left or right of a previously
            detected change point. Valid values are 'left' or 'right'.
        """
        is_top_level = False
        # cpt_found = False

        if self._finding is False:
            is_top_level = True
            self._finding = True
            assert _seg_inds is None, "ChangePointAnalysis:\tDo not provide seg_inds when calling, it's used for recursive calling only."

        if is_top_level:
            _seg_inds = self._next_seg_ind()
            self._i = 0
        else:
            _seg_inds = self._next_seg_ind(prev_seg_inds=_seg_inds, side=_side)
            # print(seg_inds)
        self._i += 1
        # print(self._i)
        # print(_seg_inds)

        if _seg_inds != (None, None):
            cpt_found = self.__weighted_likelihood_ratio(_seg_inds)
            # if seg_inds[1] != self.num_photons - 1:
            # assert side.lower() in ['left', 'right', 'l', 'r'], "ChangePointAnalysis:\tSide argument needs to be 'left' or 'right'."
            if cpt_found:
                self._find_all_cpts(_seg_inds, _side='left')  # Left side of change point
                self._find_all_cpts(_seg_inds, _side='right')
                pass  # Right side of change point

            if _seg_inds[1] <= self.num_photons+9 and _side is None:
                self._find_all_cpts(_seg_inds)

        if is_top_level:
            self._finding = False
            self.found_cpts = True
            self.cpt_inds.sort()
            self._particle.cpt_inds = self.cpt_inds
            self.num_cpts = len(self.cpt_inds)
            self._particle.num_cpts = self.num_cpts

    def get_levels(self) -> None:
        """
        Creates a list of levels as defined by the detected change points.

        Uses the detected change points to create a list of Level instances
        that contain attributes that define each resolved level.

        .. note::
            This method populate the .levels attribute of the parent particle
            instance by using its add_levels method.
        """
        assert self.found_cpts, "ChangePointAnalysis:\tChange point analysis not done, or found no change points."
        num_levels = self.num_cpts+1
        levels = [None]*num_levels

        for num, cpt in enumerate(self.cpt_inds):
            # print(num)
            if num == 0:  # First change point
                start_ind = 0
                end_ind = cpt-1
                levels[num] = Level(self._abstimes, level_inds=(start_ind, end_ind), microtimes=self._microtimes)
            elif num == self.num_cpts-1:
                start_ind = self.cpt_inds[num-1]
                end_ind = cpt-1
                levels[num] = Level(self._abstimes, level_inds=(start_ind, end_ind), microtimes=self._microtimes)

                start_ind = cpt
                end_ind = self.num_photons-1
                levels[num+1] = Level(self._abstimes, level_inds=(start_ind, end_ind), microtimes=self._microtimes)
            else:
                start_ind = self.cpt_inds[num-1]
                end_ind = cpt
                levels[num] = Level(self._abstimes, level_inds=(start_ind, end_ind), microtimes=self._microtimes)

        self._particle.add_levels(levels, num_levels)
        # return levels, num_levels

    # @dbg.profile
    def run_cpa(self, confidence=None):
        """
        Runs the change point analysis.

        If the ChangePointAnalysis wasn't initialised with a confidence interval, or if
        the analysis is to be rerun with a new confidence interval, this method starts said analysis.

        :param confidence: Confidence interval. Valid values are 0.99, 0.95, 0.90 and 0.69.
        :type confidence: float
        """
        if confidence is not None:
            assert confidence in [0.99, 0.95, 0.90, 0.69], "ChangePointAnalysis:\tConfidence value given not valid."
            self.confidence = confidence
            self._tau = TauData(confidence)
        else:
            assert self.confidence is not None, "ChangePointAnalysis:\tNo confidence value provided."

        self._find_all_cpts()


def main():
    """
    Tests ChangePoints init
    """

    test = ChangePoints()


if __name__ == '__main__':
    main()
