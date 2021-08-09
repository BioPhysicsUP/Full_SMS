"""Module for handling performing Agglomerative Hierarchical Clustering Algorithm.

Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)

Joshua Botha
University of Pretoria
2018
"""

from __future__ import annotations
# from math import lgamma

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from scipy.stats import poisson
# from matplotlib import pyplot as plt
from change_point import Level
from my_logger import setup_logger

if TYPE_CHECKING:
    from smsh5 import Particle

import dbg

logger = setup_logger(__name__)


def max_jm(array: np.ndarray):
    """ Finds the j and m index for the max value. """
    max_n = np.argmax(array)
    (xD, yD) = array.shape
    if max_n >= xD:
        max_j = max_n // xD
        max_m = max_n % xD
    else:
        max_m = max_n
        max_j = 0
    return max_j, max_m


class Group:

    def __init__(self, lvls_inds: List[int_p_s] = None, particle: Particle = None):
        self.lvls_inds = lvls_inds
        self.lvls = None
        self.histogram = None

        if self.lvls_inds is not None and particle is not None:
            self.lvls = [particle.cpts.levels[i] for i in self.lvls_inds]

    @property
    def num_photons(self) -> int_p_s:
        return int(np.sum([level.num_photons for level in self.lvls]))

    @property
    def dwell_time_s(self) -> float:
        return float(np.sum([level.dwell_time_s for level in self.lvls]))

    @property
    def int_p_s(self) -> float:
        return self.num_photons / self.dwell_time_s

    @property
    def group_times_ns(self):
        times_ns = np.array([])
        for level in self.lvls:
            times_ns = np.append(level.times_ns)
        return times_ns

    @property
    def group_microtimes(self):
        microtimes = np.array([])
        for level in self.lvls:
            times_ns = np.append(level.microtimes)
        return microtimes


# class Solution:
#
#     def __init__(self, clustering_step: ClusteringStep):
#         self._c_step = clustering_step
#
#     @property
#     def groups(self) -> List[Group]:
#         return self._c_step._seed_groups
#
#     @property
#     def num_groups(self) -> int:
#         return self._c_step._num_prev_groups
#
#     @property
#     def num_levels(self) -> int:
#         return self._c_step._num_levels


class ClusteringStep:

    def __init__(self,
                 particle: Particle,
                 first: bool = False,
                 seed_groups: List[Group] = None,
                 single_level: bool = False):

        self._particle = particle
        self._num_levels = particle.cpts.num_levels
        self.first = first
        self.single_level = single_level
        self.last = False or single_level

        self._log_l_em = None
        self._em_p_mj = None
        self._em_log_l = None
        self._ahc_p_mj = None
        self._ahc_groups = None

        self.groups = None
        self.bic = None
        self.num_groups = None
        self.level_group_ind = None

        self.group_levels = None

        if self.first or self.single_level:
            self._seed_groups = [Group([i], particle) for i in range(self._num_levels)]
            self._seed_p_mj = np.identity(n=self._num_levels)
            if self.single_level:
                self.groups = self._seed_groups
                self.group_levels = self._particle.cpts.levels
                self.num_groups = self._num_levels
                self.level_group_ind = list(range(self._num_levels))
        else:
            assert seed_groups is not None, "ClusteringStep: parameters not provided"
            self._seed_groups = seed_groups
            seed_p_mj = np.zeros(shape=(len(self._seed_groups), self._num_levels))
            for m, group in enumerate(seed_groups):
                seed_p_mj[m, group.lvls_inds] = 1
            self._seed_p_mj = seed_p_mj

        self._num_prev_groups = len(self._seed_groups)

    @property
    def group_ints(self) -> List[float]:
        if self.groups is not None:
            return [group.int_p_s for group in self.groups]

    def calc_int_bounds(self, order: str = 'descending') -> List[Tuple[float, float]]:
        """ Calculates the bounds between the groups.

        Parameters
        ----------
        order : str
            Option are 'descending' and 'ascending'
        """

        if self.groups is not None and self.single_level is False:
            assert order in ['descending', 'ascending'], "Solution: Order provided not valid"

            g_ints = self.group_ints.copy()
            g_ints.sort(reverse=(order == 'descending'))

            int_bounds = []
            for i in range(self.num_groups):

                if i == self.num_groups - 1:
                    if order == 'descending':
                        int_bounds.append((0, prev_mid_int))
                    else:
                        int_bounds.append((prev_mid_int, np.inf))
                else:
                    mid_int = (g_ints[i + 1] + g_ints[i]) / 2
                    if i == 0:
                        if order == 'descending':
                            int_bounds.append((mid_int, np.inf))
                        else:
                            int_bounds.append((0, mid_int))
                        prev_mid_int = mid_int

                    else:
                        if order == 'descending':
                            int_bounds.append((mid_int, prev_mid_int))
                        else:
                            int_bounds.append((prev_mid_int, mid_int))

                prev_mid_int = mid_int

            return int_bounds

    def ahc(self):
        merge = np.full(shape=(self._num_prev_groups, self._num_prev_groups), fill_value=-np.inf)
        for j, group_j in enumerate(self._seed_groups):  # Row
            for m, group_m in enumerate(self._seed_groups):  # Column
                if j < m:
                    n_m = group_m.num_photons
                    n_j = group_j.num_photons
                    t_m = group_m.dwell_time_s
                    t_j = group_j.dwell_time_s

                    merge[j, m] = (n_m + n_j) * np.log((n_m + n_j) / (t_m + t_j)) - n_m * np.log(n_m / t_m) - n_j * np.log(
                        n_j / t_j)
        max_j, max_m = max_jm(merge)

        new_groups = []
        p_mj = np.zeros(shape=(self._num_prev_groups - 1, self._num_levels))
        group_num = -1
        for i, group_i in enumerate(self._seed_groups):
            if i != max_m:
                group_num += 1
                if i == max_j:
                    merge_lvls = group_i.lvls_inds.copy()
                    merge_lvls.extend(self._seed_groups[max_m].lvls_inds.copy())
                    merge_lvls.sort()
                    new_groups.append(Group(lvls_inds=merge_lvls, particle=self._particle))
                else:
                    new_groups.append(group_i)

                for ind in new_groups[-1].lvls_inds:
                    p_mj[group_num, ind] = 1

        self._ahc_p_mj = p_mj
        self._ahc_groups = new_groups
        # return ClusteringStep(particle=self.particle, first=False, groups=new_groups, seed_p_mj=new_p_mj)

    def emc(self):
        """ Expectation Maximisation clustering """

        p_mj = self._ahc_p_mj.copy()
        prev_p_mj = p_mj.copy()
        levels = self._particle.cpts.levels

        i = 0
        diff_p_mj = 1
        while diff_p_mj > 1E-5 and i < 50:

            i += 1
            cap_t = self._particle.dwell_time

            t_hat = np.zeros(shape=(self._num_prev_groups - 1,))
            n_hat = np.zeros_like(t_hat)
            p_hat = np.zeros_like(t_hat)
            i_hat = np.zeros_like(t_hat)

            p_hat_g = np.zeros(shape=(self._num_prev_groups - 1, self._num_levels))
            denom = np.zeros(shape=(self._num_levels,))

            for m, group in enumerate(self._ahc_groups):

                t_hat[m] = np.sum([p_mj[m, j] * l.dwell_time_s for j, l in enumerate(levels)])
                n_hat[m] = np.sum([p_mj[m, j] * l.num_photons for j, l in enumerate(levels)])
                p_hat[m] = t_hat[m] / cap_t
                i_hat[m] = np.sum([p_mj[m, j] * l.num_photons / t_hat[m] for j, l in enumerate(levels)])

                denom_sum = 0
                for j, l in enumerate(levels):
                    p_hat_g[m, j] = p_hat[m] * poisson.pmf(l.num_photons, i_hat[m] * l.dwell_time_s)

            for j in range(self._num_levels):
                denom[j] = np.sum(p_hat_g[:, j])

                for m in range(self._num_prev_groups - 1):
                    try:
                        p_mj[m, j] = p_hat_g[m, j] / denom[j]
                    except:
                        # print('here')  # TODO: Fix div by zero
                        pass

            diff_p_mj = np.sum(np.abs(prev_p_mj - p_mj))
            prev_p_mj = p_mj.copy()

        level_p_max = np.argmax(p_mj, 0)
        eff_p_mj = np.zeros_like(p_mj)
        for j in range(self._num_levels):
            eff_p_mj[level_p_max[j], j] = 1
        self._em_p_mj = eff_p_mj

        log_l = 0
        for m in range(self._num_prev_groups - 1):
            for j in range(self._num_levels):
                if p_hat_g[m, j] > 1E-300 and eff_p_mj[m, j] != 0:  # Close to smallest value that doesn't result in -inf
                    log_l += eff_p_mj[m, j] * np.log(p_hat_g[m, j])
        self._em_log_l = log_l

        new_groups = []
        for m in range(self._num_prev_groups - 1):
            g_m_levels = list(np.nonzero(self._em_p_mj[m, :])[0])
            if len(g_m_levels):
                new_groups.append(Group(lvls_inds=g_m_levels, particle=self._particle))
        new_groups.sort(key=lambda group: group.int_p_s)
        self.groups = new_groups
        self.num_groups = len(new_groups)
        if self.num_groups == 1:
            self.last = True

        level_group_ind = [None]*self._num_levels
        for group_num, group in enumerate(self.groups):
            for level in group.lvls_inds:
                level_group_ind[level] = group_num
        self.level_group_ind = level_group_ind

    def calc_bic(self):
        num_cp = self._particle.cpts.num_cpts
        num_g = self.num_groups

        self.bic = 2 * self._em_log_l - (2 * num_g - 1) * np.log(num_cp) - num_cp * np.log(self._particle.num_photons)

    def group_2_levels(self):
        part_levels = self._particle.cpts.levels
        abs_times = self._particle.abstimes
        micro_times = self._particle.microtimes

        start_ind = 0
        group_levels = []
        for i, part_level in enumerate(part_levels):
            end_ind = part_level.level_inds[1]
            group_int = self.group_ints[self.level_group_ind[i]]
            if i < self._num_levels-1:
                if self.level_group_ind[i] != self.level_group_ind[i+1]:
                    group_levels.append(Level(abs_times=abs_times,
                                              microtimes=micro_times,
                                              level_inds=(start_ind, end_ind),
                                              int_p_s=group_int,
                                              group_ind=self.level_group_ind[i]))
                    start_ind = part_levels[i+1].level_inds[0]
            else:
                # TODO: Make sure it shouldn't be append
                group_levels.append(Level(abs_times=abs_times,
                                          microtimes=micro_times,
                                          level_inds=(start_ind, end_ind),
                                          int_p_s=group_int,
                                          group_ind=self.level_group_ind[i]))

        self.group_levels = group_levels

    @property
    def group_level_dwelltimes(self):
        return [level.dwell_time_s for level in self.group_levels]

    @property
    def group_total_dwelltime(self):
        return np.sum(self.group_level_dwelltimes)

    @property
    def group_num_levels(self):
        return len(self.group_levels)

    @property
    def group_level_ints(self):
        return np.array([level.int_p_s for level in self.group_levels])

    def setup_next_step(self) -> ClusteringStep:
        return ClusteringStep(self._particle, first=False, seed_groups=self.groups)


class AHCA:
    """
    Class for executing Agglomerative Hierarchical Clustering Algorithm and storing the results.
    """

    def __init__(self, particle):
        """

        Parameters
        ----------
        particle: smsh5.Particle
        """

        self.has_groups = False
        self._particle = particle
        self.uuid = self._particle.uuid
        self.steps = None
        self.best_step_ind = None
        self.bics = None
        self.selected_step_ind = None
        self.num_steps = None


    @property
    def selected_step(self) -> ClusteringStep:
        if self.has_groups:
            return self.steps[self.selected_step_ind]

    @selected_step.setter
    def selected_step(self, step_ind):
        assert 0 < step_ind < self.num_steps, "AHCA: Provided step index out of range."
        self.selected_step_ind = step_ind

    @property
    def best_step(self) -> ClusteringStep:
        if self.has_groups:
            return self.steps[self.best_step_ind]

    @property
    def steps_num_groups(self) -> List[int]:
        if self.has_groups:
            return [step.num_groups for step in self.steps]

    def run_grouping(self):
        """
        Run grouping

        Returns
        -------

        """

        try:
            if self._particle.has_levels:

                steps = []
                if self._particle.num_levels == 1:
                    self.steps = [ClusteringStep(self._particle, single_level=True)]
                    self.num_steps = 1
                    self.best_step_ind = 0
                    self.selected_step_ind = 0
                    self.has_groups = True
                else:
                    c_step = ClusteringStep(self._particle, first=True)
                    current_num_groups = self._particle.num_levels
                    while current_num_groups != 1:
                        c_step.ahc()
                        c_step.emc()
                        c_step.calc_bic()
                        c_step.group_2_levels()
                        # print([lvl.int_p_s for lvl in c_step.group_levels])
                        steps.append(c_step)
                        current_num_groups = c_step.num_groups
                        if current_num_groups != 1:
                            # print(current_num_groups)
                            c_step = c_step.setup_next_step()

                    self.steps = steps
                    self.num_steps = len(steps)
                    self.bics = [step.bic for step in steps]
                    self.best_step_ind = np.argmax(self.bics)
                    self.selected_step_ind = self.best_step_ind
                    self.has_groups = True
                    logger.info(f"{self._particle.name} levels grouped")
            else:
                logger.info(f"{self._particle.name} has no levels to group")
        except Exception as e:
            logger.error(e)
            pass

    def set_selected_step(self, step_ind: int):
        assert 0 <= step_ind < self.num_steps, "AHCA: Provided step index out of range."
        self.selected_step_ind = step_ind
        if not all([lvl.histogram is not None for lvl in self.steps[step_ind].group_levels]):
            self._particle.makelevelhists(force_group_levels=True)
        if not all([group.histogram is not None for group in self.steps[step_ind].groups]):
            self._particle.makegrouphists()


    def reset_selected_step(self):
        self.selected_step_ind = self.best_step_ind

