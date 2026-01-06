"""Data models for groups from hierarchical clustering."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GroupData:
    """A group of levels with similar intensity from hierarchical clustering.

    Groups are the output of Agglomerative Hierarchical Clustering Analysis (AHCA).
    Each group contains levels that have been determined to belong to the same
    intensity state based on the BIC-optimized clustering.

    Attributes:
        group_id: Unique identifier for this group within the clustering result.
        level_indices: Indices of levels (in the particle's level list) that belong to this group.
        total_photons: Sum of photons across all levels in this group.
        total_dwell_time_s: Sum of dwell times (in seconds) across all levels.
        intensity_cps: Average intensity in counts per second (total_photons / total_dwell_time_s).
    """

    group_id: int
    level_indices: Tuple[int, ...]
    total_photons: int
    total_dwell_time_s: float
    intensity_cps: float

    @property
    def num_levels(self) -> int:
        """Number of levels in this group."""
        return len(self.level_indices)

    def __post_init__(self) -> None:
        """Validate group data."""
        if not self.level_indices:
            raise ValueError("level_indices cannot be empty")
        if self.total_photons < 0:
            raise ValueError(f"total_photons must be non-negative, got {self.total_photons}")
        if self.total_dwell_time_s < 0:
            raise ValueError(
                f"total_dwell_time_s must be non-negative, got {self.total_dwell_time_s}"
            )
        if self.intensity_cps < 0:
            raise ValueError(f"intensity_cps must be non-negative, got {self.intensity_cps}")

    @classmethod
    def from_level_data(
        cls,
        group_id: int,
        level_indices: List[int],
        level_photons: List[int],
        level_dwell_times_s: List[float],
    ) -> "GroupData":
        """Create a GroupData from level information.

        Args:
            group_id: Unique identifier for this group.
            level_indices: Indices of levels in this group.
            level_photons: Number of photons in each level.
            level_dwell_times_s: Dwell time in seconds for each level.

        Returns:
            A new GroupData instance with computed aggregate statistics.
        """
        if len(level_indices) != len(level_photons) or len(level_indices) != len(
            level_dwell_times_s
        ):
            raise ValueError("All input lists must have the same length")

        total_photons = sum(level_photons)
        total_dwell_time_s = sum(level_dwell_times_s)

        if total_dwell_time_s > 0:
            intensity_cps = total_photons / total_dwell_time_s
        else:
            intensity_cps = 0.0

        return cls(
            group_id=group_id,
            level_indices=tuple(level_indices),
            total_photons=total_photons,
            total_dwell_time_s=total_dwell_time_s,
            intensity_cps=intensity_cps,
        )


@dataclass(frozen=True)
class ClusteringStep:
    """A single step in the hierarchical clustering process.

    Each step represents a grouping with a specific number of groups.

    Attributes:
        groups: Groups at this step.
        level_group_assignments: For each original level index, the group index it belongs to.
        bic: BIC value at this step.
        num_groups: Number of groups at this step.
    """

    groups: Tuple["GroupData", ...]
    level_group_assignments: Tuple[int, ...]
    bic: float
    num_groups: int

    def __post_init__(self) -> None:
        """Validate step data."""
        if not self.groups:
            raise ValueError("groups cannot be empty")
        if self.num_groups != len(self.groups):
            raise ValueError(
                f"num_groups ({self.num_groups}) doesn't match len(groups) ({len(self.groups)})"
            )


@dataclass(frozen=True)
class ClusteringResult:
    """Complete result of hierarchical clustering analysis.

    Contains all clustering steps from N groups down to 1, along with
    BIC values for selecting the optimal number of groups.

    Attributes:
        steps: All clustering steps, from N-1 groups down to 1 group.
        optimal_step_index: Index of the step with maximum BIC (optimal grouping).
        selected_step_index: Currently selected step (may differ from optimal).
        num_original_levels: Number of levels before clustering.
    """

    steps: Tuple["ClusteringStep", ...]
    optimal_step_index: int
    selected_step_index: int
    num_original_levels: int

    @property
    def groups(self) -> Tuple["GroupData", ...]:
        """Groups at the selected step."""
        return self.steps[self.selected_step_index].groups

    @property
    def all_bic_values(self) -> Tuple[float, ...]:
        """BIC values for all steps."""
        return tuple(step.bic for step in self.steps)

    @property
    def num_groups(self) -> int:
        """Number of groups at the selected step."""
        return self.steps[self.selected_step_index].num_groups

    @property
    def num_steps(self) -> int:
        """Total number of clustering steps."""
        return len(self.steps)

    @property
    def optimal_bic(self) -> float:
        """BIC value at the optimal step."""
        return self.steps[self.optimal_step_index].bic

    @property
    def selected_bic(self) -> float:
        """BIC value at the selected step."""
        return self.steps[self.selected_step_index].bic

    @property
    def is_optimal_selected(self) -> bool:
        """Whether the currently selected step is the optimal one."""
        return self.selected_step_index == self.optimal_step_index

    @property
    def level_group_assignments(self) -> Tuple[int, ...]:
        """For each level, the group it's assigned to at the selected step."""
        return self.steps[self.selected_step_index].level_group_assignments

    def get_groups_at_step(self, step_index: int) -> Tuple["GroupData", ...]:
        """Get groups at a specific step.

        Args:
            step_index: Index of the step (0 = N-1 groups, last = 1 group).

        Returns:
            Groups at the specified step.
        """
        if not 0 <= step_index < len(self.steps):
            raise IndexError(f"step_index {step_index} out of range [0, {len(self.steps)})")
        return self.steps[step_index].groups

    def with_selected_step(self, step_index: int) -> "ClusteringResult":
        """Create a new result with a different selected step.

        Args:
            step_index: The new selected step index.

        Returns:
            A new ClusteringResult with the updated selection.
        """
        if not 0 <= step_index < len(self.steps):
            raise IndexError(f"step_index {step_index} out of range [0, {len(self.steps)})")
        return ClusteringResult(
            steps=self.steps,
            optimal_step_index=self.optimal_step_index,
            selected_step_index=step_index,
            num_original_levels=self.num_original_levels,
        )

    def __post_init__(self) -> None:
        """Validate clustering result."""
        if not self.steps:
            raise ValueError("steps cannot be empty")
        if not 0 <= self.optimal_step_index < len(self.steps):
            raise ValueError(
                f"optimal_step_index ({self.optimal_step_index}) out of range "
                f"[0, {len(self.steps)})"
            )
        if not 0 <= self.selected_step_index < len(self.steps):
            raise ValueError(
                f"selected_step_index ({self.selected_step_index}) out of range "
                f"[0, {len(self.steps)})"
            )
        if self.num_original_levels < 1:
            raise ValueError(
                f"num_original_levels must be positive, got {self.num_original_levels}"
            )
