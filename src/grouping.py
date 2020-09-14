"""Module for handling performing Agglomerative Hierarchical Clustering Algorithm.

Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
from Watkins nad Yang, J. Phys. Chem. B 2005, 109, 617-628 (http://pubs.acs.org/doi/abs/10.1021/jp0467548)

Joshua Botha
University of Pretoria
2018
"""

from __future__ import annotations
from math import lgamma

from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from scipy.stats import poisson
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from smsh5 import Particle

import dbg


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

    def __init__(self, lvls_inds: List[int] = None, particle: Particle = None):
        self.lvls_inds = lvls_inds
        self.lvls = None

        if self.lvls_inds is not None and particle is not None:
            self.lvls = [particle.levels[i] for i in self.lvls_inds]

    @property
    def num_photons(self) -> int:
        return int(np.sum([level.num_photons for level in self.lvls]))

    @property
    def dwell_time(self) -> float:
        return float(np.sum([level.dwell_time_s for level in self.lvls]))

    @property
    def int(self) -> float:
        return self.num_photons / self.dwell_time


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
                 seed_p_mj: np.ndarray = None):

        self._particle = particle
        self._num_levels = particle.num_levels

        if first:
            self._seed_groups = [Group([i], particle) for i in range(self._num_levels)]
            self._seed_p_mj = np.identity(n=self._num_levels)
        else:
            assert seed_groups is None and seed_p_mj is not None, "ClusteringStep: parameters not provided"
            self._seed_groups = seed_groups
            self._seed_p_mj = seed_p_mj

        self._num_prev_groups = len(self._seed_groups)
        self._log_l_em = None
        self._em_p_mj = None
        self._em_log_l = None
        self._ahc_p_mj = None

        self.groups = None
        self.bic = None
        self.num_groups = None

    @property
    def group_ints(self) -> List[float]:
        if self.groups is not None:
            return [group.int for group in self.groups]

    def calc_int_bounds(self, order: str = 'descending') -> List[Tuple[float, float]]:
        """ Calculates the bounds between the groups.

        Parameters
        ----------
        order : str
            Option are 'descending' and 'ascending'
        """

        if self.groups is not None:
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
                    t_m = group_m.dwell_time
                    t_j = group_j.dwell_time

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
        # return ClusteringStep(particle=self.particle, first=False, groups=new_groups, seed_p_mj=new_p_mj)

    def emc(self):
        """ Expectation Maximisation clustering """

        p_mj = self._ahc_p_mj.copy()
        prev_p_mj = p_mj.copy()
        levels = self._particle.levels

        i = 0
        diff_p_mj = 1
        while diff_p_mj > 1E-5 and i < 50:

            i += 1
            cap_t = self._particle.dwell_time

            t_hat = np.zeros(shape=(self._num_prev_groups,))
            n_hat = np.zeros_like(t_hat)
            p_hat = np.zeros_like(t_hat)
            i_hat = np.zeros_like(t_hat)

            p_hat_g = np.zeros(shape=(self._num_prev_groups, self._num_levels))
            denom = np.zeros(shape=(self._num_levels,))

            for m, group in enumerate(self._seed_groups):

                t_hat[m] = np.sum([p_mj[m, j] * l.dwell_time_s for j, l in enumerate(levels)])
                n_hat[m] = np.sum([p_mj[m, j] * l.num_photons for j, l in enumerate(levels)])
                p_hat[m] = t_hat[m] / cap_t
                i_hat[m] = np.sum([p_mj[m, j] * l.num_photons / t_hat[m] for j, l in enumerate(levels)])

                denom_sum = 0
                for j, l in enumerate(levels):
                    p_hat_g[m, j] = p_hat[m] * poisson.pmf(l.num_photons, i_hat[m] * l.dwell_time_s)

            for j in range(self._num_levels):
                denom[j] = np.sum(p_hat_g[:, j])

                for m in range(self._num_prev_groups):
                    try:
                        p_mj[m, j] = p_hat_g[m, j] / denom[j]
                    except:
                        print('here')
                        pass

            diff_p_mj = np.sum(np.abs(prev_p_mj - p_mj))
            prev_p_mj = p_mj.copy()

        level_p_max = np.argmax(p_mj, 0)
        eff_p_mj = np.zeros_like(p_mj)
        for j in range(self._num_levels):
            eff_p_mj[level_p_max[j], j] = 1
        self._em_p_mj = eff_p_mj

        log_l = 0
        for m in range(self._num_prev_groups):
            for j in range(self._num_levels):
                if p_hat_g[m, j] > 1E-300 and eff_p_mj[m, j] != 0:  # Close to smallest value that doesn't result in -inf
                    log_l += eff_p_mj[m, j] * np.log(p_hat_g[m, j])
        self._em_log_l = log_l

        new_groups = []
        for m in range(self._num_prev_groups):
            g_m_levels = np.nonzero(self._em_p_mj)[0]
            if len(g_m_levels):
                new_groups.append(Group(lvls_inds=g_m_levels, particle=self._particle))
        self.groups = new_groups
        self.num_groups = len(new_groups)

    def calc_bic(self):
        num_cp = self._particle.cpts.num_cpts
        num_g = self.num_groups

        self.bic = 2 * self._em_log_l - (2 * num_g - 1) * np.log(num_cp) - num_cp * np.log(self._particle.num_photons)

    def setup_next_step(self) -> ClusteringStep:
        return ClusteringStep(self._particle, first=Fasle, seed_groups=self.groups, seed_p_mj=self._em_p_mj)


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
        self.particle = particle
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

    def run_grouping(self):
        """
        Run grouping

        Returns
        -------

        """

        if self.particle.has_levels:

            steps = []
            c_step = ClusteringStep(self.particle, first=True)
            current_num_groups = self.particle.num_levels
            while current_num_groups != 1:
                c_step.ahc()
                c_step.emc()
                c_step.calc_bic()
                steps.append(c_step)
                current_num_groups = c_step.num_groups
                if current_num_groups != 1:
                    c_step = c_step.setup_next_step()

            self.steps = steps
            self.num_steps = len(steps)
            self.bics = [step.bic for step in ahca_steps]
            self.best_step_ind = np.argmax(self.bics)
            self.selected_step_ind = self.best_step_ind
            self.has_groups = True
        else:
            dbg.p(f"{self.particle.name} has no levels to group", "ACHA")

    def set_selected_step(self, step_ind: int):
        assert 0 < step_ind < self.num_steps, "AHCA: Provided step index out of range."
        self.selected_step_ind = step_ind

    def reset_selected_step(self):
        self.selected_step_ind = self.best_step_ind
