"""Pytest configuration and fixtures."""

import pytest

from full_sms.models.group import ClusteringResult, ClusteringStep, GroupData


def make_clustering_result(
    num_steps: int = 2,
    num_original_levels: int = 3,
    optimal_step_index: int = 0,
    selected_step_index: int = 0,
) -> ClusteringResult:
    """Helper to create a ClusteringResult for testing.

    Creates a simple clustering result with the specified number of steps.
    Each step has one fewer group than the previous, starting with
    num_original_levels groups at step 0.

    BIC values are 100.0 + step_index * 5 for predictable testing.
    """
    steps = []
    for i in range(num_steps):
        num_groups = num_original_levels - i
        if num_groups < 1:
            num_groups = 1
        groups = []
        assignments = [0] * num_original_levels

        # Create groups
        for g in range(num_groups):
            groups.append(
                GroupData(
                    group_id=g,
                    level_indices=(g,) if g < num_original_levels else (),
                    total_photons=100,
                    total_dwell_time_s=0.1,
                    intensity_cps=1000.0,
                )
            )

        # Assign levels to groups
        for lvl in range(num_original_levels):
            if lvl < num_groups:
                assignments[lvl] = lvl
            else:
                # Assign extra levels to first group
                assignments[lvl] = 0
                # Update first group's level_indices
                current_indices = list(groups[0].level_indices)
                if lvl not in current_indices:
                    current_indices.append(lvl)
                    groups[0] = GroupData(
                        group_id=0,
                        level_indices=tuple(sorted(current_indices)),
                        total_photons=100 * len(current_indices),
                        total_dwell_time_s=0.1 * len(current_indices),
                        intensity_cps=1000.0,
                    )

        step = ClusteringStep(
            groups=tuple(groups),
            level_group_assignments=tuple(assignments),
            bic=100.0 + i * 5,
            num_groups=num_groups,
        )
        steps.append(step)

    return ClusteringResult(
        steps=tuple(steps),
        optimal_step_index=optimal_step_index,
        selected_step_index=selected_step_index,
        num_original_levels=num_original_levels,
    )
