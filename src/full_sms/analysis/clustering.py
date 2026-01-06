"""Agglomerative Hierarchical Clustering Algorithm (AHCA) for level grouping.

Based on 'Detection of Intensity Change Points in Time-Resolved Single-Molecule Measurements'
from Watkins and Yang, J. Phys. Chem. B 2005, 109, 617-628.

The algorithm groups intensity levels detected by change point analysis into
distinct intensity states using BIC-optimized hierarchical clustering.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numba import jit
from scipy.stats import poisson

from full_sms.models.group import ClusteringResult, ClusteringStep, GroupData
from full_sms.models.level import LevelData


@dataclass
class _LevelInfo:
    """Internal representation of a level for clustering calculations."""

    index: int
    num_photons: int
    dwell_time_s: float


@jit(nopython=True, cache=True)
def _compute_merge_merit_matrix(
    photon_counts: np.ndarray,
    dwell_times: np.ndarray,
    group_assignments: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    """Compute the merge merit matrix for all pairs of groups.

    Uses the log-likelihood merge merit function from Watkins & Yang eq. 11.

    Args:
        photon_counts: Array of photon counts for each level.
        dwell_times: Array of dwell times (in seconds) for each level.
        group_assignments: Current group assignment for each level.
        num_groups: Current number of groups.

    Returns:
        Upper triangular matrix of merge merit values (j, m) where j < m.
        Values are -inf for invalid pairs or the diagonal.
    """
    # First, compute aggregate stats for each group
    group_photons = np.zeros(num_groups, dtype=np.float64)
    group_times = np.zeros(num_groups, dtype=np.float64)

    for i in range(len(photon_counts)):
        g = group_assignments[i]
        group_photons[g] += photon_counts[i]
        group_times[g] += dwell_times[i]

    # Compute merge merit for each pair
    merge_merit = np.full((num_groups, num_groups), -np.inf, dtype=np.float64)

    for j in range(num_groups):
        for m in range(j + 1, num_groups):
            n_j = group_photons[j]
            n_m = group_photons[m]
            t_j = group_times[j]
            t_m = group_times[m]

            if n_j > 0 and n_m > 0 and t_j > 0 and t_m > 0:
                # Merge merit (eq. 11 from Watkins & Yang)
                # merit = (n_m + n_j) * ln[(n_m + n_j)/(t_m + t_j)]
                #         - n_m * ln[n_m/t_m]
                #         - n_j * ln[n_j/t_j]
                merge_merit[j, m] = (
                    (n_m + n_j) * np.log((n_m + n_j) / (t_m + t_j))
                    - n_m * np.log(n_m / t_m)
                    - n_j * np.log(n_j / t_j)
                )

    return merge_merit


def _find_best_merge(merge_merit: np.ndarray) -> Tuple[int, int]:
    """Find the pair of groups with the highest merge merit.

    Args:
        merge_merit: Upper triangular merge merit matrix.

    Returns:
        Tuple (j, m) where j < m of the groups to merge.
    """
    max_idx = np.argmax(merge_merit)
    j, m = np.unravel_index(max_idx, merge_merit.shape)
    return int(j), int(m)


def _merge_groups(
    group_assignments: np.ndarray,
    num_groups: int,
    merge_j: int,
    merge_m: int,
) -> np.ndarray:
    """Merge two groups by reassigning levels.

    Args:
        group_assignments: Current group assignment for each level.
        num_groups: Current number of groups.
        merge_j: Index of first group to merge (will absorb group m).
        merge_m: Index of second group to merge.

    Returns:
        New group assignments array with groups renumbered.
    """
    new_assignments = group_assignments.copy()

    # Reassign group m to group j
    new_assignments[new_assignments == merge_m] = merge_j

    # Renumber groups to fill the gap
    for g in range(merge_m + 1, num_groups):
        new_assignments[new_assignments == g] = g - 1

    return new_assignments


def _run_em_refinement(
    photon_counts: np.ndarray,
    dwell_times: np.ndarray,
    group_assignments: np.ndarray,
    num_groups: int,
    total_dwell_time: float,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """Run Expectation-Maximization to refine group assignments.

    The EM algorithm refines the hard assignments from AHC by computing
    soft assignment probabilities and then converting back to hard assignments.

    Args:
        photon_counts: Array of photon counts for each level.
        dwell_times: Array of dwell times (in seconds) for each level.
        group_assignments: Initial group assignments from AHC.
        num_groups: Number of groups.
        total_dwell_time: Total measurement dwell time.
        max_iterations: Maximum EM iterations.
        tolerance: Convergence tolerance for assignment probabilities.

    Returns:
        Tuple of (refined_assignments, log_likelihood).
    """
    num_levels = len(photon_counts)

    # Initialize soft assignment matrix p_mj from hard assignments
    # p_mj[m, j] = probability level j belongs to group m
    p_mj = np.zeros((num_groups, num_levels), dtype=np.float64)
    for j, g in enumerate(group_assignments):
        p_mj[g, j] = 1.0

    prev_p_mj = p_mj.copy()

    for iteration in range(max_iterations):
        # M-Step: Estimate group parameters from soft assignments
        t_hat = np.zeros(num_groups, dtype=np.float64)
        n_hat = np.zeros(num_groups, dtype=np.float64)

        for m in range(num_groups):
            for j in range(num_levels):
                t_hat[m] += p_mj[m, j] * dwell_times[j]
                n_hat[m] += p_mj[m, j] * photon_counts[j]

        # Group prior probability and intensity
        p_hat = t_hat / total_dwell_time
        i_hat = np.zeros(num_groups, dtype=np.float64)
        for m in range(num_groups):
            if t_hat[m] > 0:
                i_hat[m] = n_hat[m] / t_hat[m]

        # E-Step: Compute alpha_mj = p_hat[m] * Poisson(n_j; i_hat[m] * t_j)
        # Then p_mj = alpha_mj / sum_m(alpha_mj)
        alpha_mj = np.zeros((num_groups, num_levels), dtype=np.float64)
        for m in range(num_groups):
            for j in range(num_levels):
                mu = i_hat[m] * dwell_times[j]
                if mu > 0:
                    alpha_mj[m, j] = p_hat[m] * poisson.pmf(int(photon_counts[j]), mu)

        # Normalize to get probabilities
        for j in range(num_levels):
            denom = np.sum(alpha_mj[:, j])
            if denom > 0:
                p_mj[:, j] = alpha_mj[:, j] / denom
            # else: keep previous probabilities

        # Check convergence
        diff = np.sum(np.abs(prev_p_mj - p_mj))
        if diff < tolerance:
            break
        prev_p_mj = p_mj.copy()

    # Convert soft assignments to hard assignments
    refined_assignments = np.argmax(p_mj, axis=0)

    # Compute log-likelihood for BIC calculation
    # Using the final alpha_mj values
    log_l = 0.0
    for m in range(num_groups):
        for j in range(num_levels):
            if alpha_mj[m, j] > 1e-200 and p_mj[m, j] > 0:
                # Only count levels assigned to this group
                if refined_assignments[j] == m:
                    log_l += np.log(alpha_mj[m, j])

    return refined_assignments, log_l


def _compute_bic(
    log_likelihood: float,
    num_groups: int,
    num_changepoints: int,
    total_photons: int,
) -> float:
    """Compute the Bayesian Information Criterion.

    BIC = 2 * log_l - (2*G - 1) * log(J) - J * log(N)

    where:
    - log_l = log-likelihood from EM
    - G = number of groups
    - J = number of change points (levels - 1)
    - N = total number of photons

    Args:
        log_likelihood: Log-likelihood from EM.
        num_groups: Number of groups.
        num_changepoints: Number of change points (num_levels - 1).
        total_photons: Total number of photons.

    Returns:
        BIC value.
    """
    if num_changepoints <= 0 or total_photons <= 0:
        return -np.inf

    return (
        2 * log_likelihood
        - (2 * num_groups - 1) * np.log(num_changepoints)
        - num_changepoints * np.log(total_photons)
    )


def _build_groups_from_assignments(
    levels: List[LevelData],
    assignments: np.ndarray,
    num_groups: int,
) -> List[GroupData]:
    """Build GroupData objects from level assignments.

    Args:
        levels: Original level data.
        assignments: Group assignment for each level.
        num_groups: Number of groups.

    Returns:
        List of GroupData objects sorted by intensity (ascending).
    """
    groups = []

    for g in range(num_groups):
        level_indices = [i for i, a in enumerate(assignments) if a == g]
        if not level_indices:
            continue

        level_photons = [levels[i].num_photons for i in level_indices]
        level_dwell_times = [levels[i].dwell_time_s for i in level_indices]

        group = GroupData.from_level_data(
            group_id=g,
            level_indices=level_indices,
            level_photons=level_photons,
            level_dwell_times_s=level_dwell_times,
        )
        groups.append(group)

    # Sort by intensity and reassign IDs
    groups.sort(key=lambda g: g.intensity_cps)
    sorted_groups = []
    for new_id, group in enumerate(groups):
        sorted_groups.append(
            GroupData(
                group_id=new_id,
                level_indices=group.level_indices,
                total_photons=group.total_photons,
                total_dwell_time_s=group.total_dwell_time_s,
                intensity_cps=group.intensity_cps,
            )
        )

    return sorted_groups


def _remap_assignments_to_sorted_groups(
    assignments: np.ndarray,
    sorted_groups: List[GroupData],
) -> np.ndarray:
    """Remap level assignments to match the sorted group IDs.

    Args:
        assignments: Original group assignments.
        sorted_groups: Groups sorted by intensity with new IDs.

    Returns:
        Remapped assignments array.
    """
    # Build mapping from old group ID to new sorted group ID
    # The sorted_groups contain the old level_indices which tell us the old group
    old_to_new = {}
    for new_group in sorted_groups:
        # All levels in this group had the same old assignment
        old_group_id = assignments[new_group.level_indices[0]]
        old_to_new[old_group_id] = new_group.group_id

    # Remap
    remapped = np.zeros_like(assignments)
    for i, old_id in enumerate(assignments):
        remapped[i] = old_to_new[old_id]

    return remapped


def cluster_levels(
    levels: List[LevelData],
    use_lifetime: bool = False,  # Not implemented yet
) -> Optional[ClusteringResult]:
    """Perform agglomerative hierarchical clustering on intensity levels.

    Uses the Watkins & Yang (2005) AHCA algorithm to group similar intensity
    levels into distinct intensity states. The optimal number of groups is
    determined by maximizing the Bayesian Information Criterion (BIC).

    The algorithm:
    1. Start with each level as its own group
    2. At each step:
       a. Calculate merge merit for all group pairs
       b. Merge the pair with highest merit
       c. Run EM refinement to optimize assignments
       d. Calculate BIC for this grouping
    3. Continue until only 1 group remains
    4. Return the grouping with maximum BIC as optimal

    Args:
        levels: List of LevelData from change point analysis.
        use_lifetime: If True, incorporate lifetime data in clustering.
            (Not yet implemented - currently ignored)

    Returns:
        ClusteringResult containing all steps and BIC values, or None if
        clustering is not possible (e.g., fewer than 1 level).
    """
    num_levels = len(levels)

    if num_levels == 0:
        return None

    if num_levels == 1:
        # Single level: trivial case
        group = GroupData.from_level_data(
            group_id=0,
            level_indices=[0],
            level_photons=[levels[0].num_photons],
            level_dwell_times_s=[levels[0].dwell_time_s],
        )
        step = ClusteringStep(
            groups=(group,),
            level_group_assignments=(0,),
            bic=0.0,  # BIC undefined for single group
            num_groups=1,
        )
        return ClusteringResult(
            steps=(step,),
            optimal_step_index=0,
            selected_step_index=0,
            num_original_levels=1,
        )

    # Extract level data into arrays for efficient computation
    photon_counts = np.array([level.num_photons for level in levels], dtype=np.float64)
    dwell_times = np.array([level.dwell_time_s for level in levels], dtype=np.float64)

    total_photons = int(np.sum(photon_counts))
    total_dwell_time = float(np.sum(dwell_times))
    num_changepoints = num_levels - 1

    # Initialize: each level is its own group
    group_assignments = np.arange(num_levels, dtype=np.int64)
    num_groups = num_levels

    steps: List[ClusteringStep] = []
    bic_values: List[float] = []

    # Clustering loop: merge until 1 group remains
    while num_groups > 1:
        # Compute merge merit matrix
        merge_merit = _compute_merge_merit_matrix(
            photon_counts, dwell_times, group_assignments, num_groups
        )

        # Find best pair to merge
        merge_j, merge_m = _find_best_merge(merge_merit)

        # Merge the groups
        group_assignments = _merge_groups(
            group_assignments, num_groups, merge_j, merge_m
        )
        num_groups -= 1

        # Run EM refinement
        group_assignments, log_likelihood = _run_em_refinement(
            photon_counts,
            dwell_times,
            group_assignments,
            num_groups,
            total_dwell_time,
        )

        # Count actual groups (EM may have emptied some)
        actual_groups = len(set(group_assignments))

        # Renumber if groups were eliminated
        if actual_groups < num_groups:
            # Compact the group numbers
            unique_groups = sorted(set(group_assignments))
            remap = {old: new for new, old in enumerate(unique_groups)}
            group_assignments = np.array([remap[g] for g in group_assignments])
            num_groups = actual_groups

        # Compute BIC
        bic = _compute_bic(log_likelihood, num_groups, num_changepoints, total_photons)
        bic_values.append(bic)

        # Build GroupData objects
        groups = _build_groups_from_assignments(levels, group_assignments, num_groups)

        # Remap assignments to match sorted group IDs
        remapped_assignments = _remap_assignments_to_sorted_groups(
            group_assignments, groups
        )

        # Create step
        step = ClusteringStep(
            groups=tuple(groups),
            level_group_assignments=tuple(remapped_assignments.tolist()),
            bic=bic,
            num_groups=len(groups),
        )
        steps.append(step)

    if not steps:
        return None

    # Find optimal step (maximum BIC)
    optimal_step_index = int(np.argmax(bic_values))

    return ClusteringResult(
        steps=tuple(steps),
        optimal_step_index=optimal_step_index,
        selected_step_index=optimal_step_index,
        num_original_levels=num_levels,
    )
