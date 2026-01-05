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
class ClusteringResult:
    """Complete result of hierarchical clustering analysis.

    Contains all clustering steps from N groups down to 1, along with
    BIC values for selecting the optimal number of groups.

    Attributes:
        groups: List of groups at the selected (optimal) step.
        all_bic_values: BIC values for each clustering step.
        optimal_step_index: Index of the step with maximum BIC (optimal grouping).
        selected_step_index: Currently selected step (may differ from optimal).
        num_original_levels: Number of levels before clustering.
    """

    groups: Tuple["GroupData", ...]
    all_bic_values: Tuple[float, ...]
    optimal_step_index: int
    selected_step_index: int
    num_original_levels: int

    @property
    def num_groups(self) -> int:
        """Number of groups at the selected step."""
        return len(self.groups)

    @property
    def num_steps(self) -> int:
        """Total number of clustering steps."""
        return len(self.all_bic_values)

    @property
    def optimal_bic(self) -> float:
        """BIC value at the optimal step."""
        return self.all_bic_values[self.optimal_step_index]

    @property
    def selected_bic(self) -> float:
        """BIC value at the selected step."""
        return self.all_bic_values[self.selected_step_index]

    @property
    def is_optimal_selected(self) -> bool:
        """Whether the currently selected step is the optimal one."""
        return self.selected_step_index == self.optimal_step_index

    def __post_init__(self) -> None:
        """Validate clustering result."""
        if not self.groups:
            raise ValueError("groups cannot be empty")
        if not self.all_bic_values:
            raise ValueError("all_bic_values cannot be empty")
        if not 0 <= self.optimal_step_index < len(self.all_bic_values):
            raise ValueError(
                f"optimal_step_index ({self.optimal_step_index}) out of range "
                f"[0, {len(self.all_bic_values)})"
            )
        if not 0 <= self.selected_step_index < len(self.all_bic_values):
            raise ValueError(
                f"selected_step_index ({self.selected_step_index}) out of range "
                f"[0, {len(self.all_bic_values)})"
            )
        if self.num_original_levels < 1:
            raise ValueError(
                f"num_original_levels must be positive, got {self.num_original_levels}"
            )
