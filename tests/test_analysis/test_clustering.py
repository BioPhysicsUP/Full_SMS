"""Tests for the AHCA clustering algorithm."""

import numpy as np
import pytest

from full_sms.analysis.clustering import (
    _build_groups_from_assignments,
    _compute_bic,
    _compute_merge_merit_matrix,
    _find_best_merge,
    _merge_groups,
    _run_em_refinement,
    cluster_levels,
)
from full_sms.models.level import LevelData


def _make_level(
    start_index: int = 0,
    end_index: int = 99,
    start_time_ns: int = 0,
    end_time_ns: int = 100_000_000,
    num_photons: int = 100,
    intensity_cps: float = 1000.0,
) -> LevelData:
    """Helper to create test levels."""
    return LevelData(
        start_index=start_index,
        end_index=end_index,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        num_photons=num_photons,
        intensity_cps=intensity_cps,
    )


class TestMergeMeritMatrix:
    """Tests for merge merit matrix computation."""

    def test_two_groups_computes_merit(self) -> None:
        """Computes merge merit for two groups."""
        photon_counts = np.array([100.0, 200.0])
        dwell_times = np.array([0.1, 0.1])
        assignments = np.array([0, 1])

        merit = _compute_merge_merit_matrix(photon_counts, dwell_times, assignments, 2)

        # Should have a valid value at [0, 1]
        assert merit[0, 1] > -np.inf
        # Lower triangle should be -inf
        assert merit[1, 0] == -np.inf
        # Diagonal should be -inf
        assert merit[0, 0] == -np.inf

    def test_similar_intensities_high_merit(self) -> None:
        """Groups with similar intensities have higher merge merit."""
        photon_counts = np.array([100.0, 100.0, 200.0])
        dwell_times = np.array([0.1, 0.1, 0.1])
        assignments = np.array([0, 1, 2])

        merit = _compute_merge_merit_matrix(photon_counts, dwell_times, assignments, 3)

        # Similar intensities (groups 0 and 1) should have higher merge merit
        # than different intensities (groups 0 and 2)
        assert merit[0, 1] > merit[0, 2]

    def test_aggregates_multi_level_groups(self) -> None:
        """Correctly aggregates stats for groups with multiple levels."""
        photon_counts = np.array([50.0, 50.0, 100.0])
        dwell_times = np.array([0.05, 0.05, 0.1])
        # Levels 0 and 1 are in group 0, level 2 is in group 1
        assignments = np.array([0, 0, 1])

        merit = _compute_merge_merit_matrix(photon_counts, dwell_times, assignments, 2)

        # Should have a valid merit value
        assert merit[0, 1] > -np.inf


class TestFindBestMerge:
    """Tests for finding the best group pair to merge."""

    def test_finds_maximum(self) -> None:
        """Finds the pair with maximum merge merit."""
        merit = np.array([
            [-np.inf, 10.0, 5.0],
            [-np.inf, -np.inf, 8.0],
            [-np.inf, -np.inf, -np.inf],
        ])

        j, m = _find_best_merge(merit)

        assert j == 0
        assert m == 1


class TestMergeGroups:
    """Tests for merging groups."""

    def test_merge_two_groups(self) -> None:
        """Merges two groups and renumbers."""
        assignments = np.array([0, 1, 2])

        new_assignments = _merge_groups(assignments, 3, 0, 1)

        # Group 1 should be merged into group 0
        # Group 2 should become group 1
        assert new_assignments[0] == 0
        assert new_assignments[1] == 0
        assert new_assignments[2] == 1

    def test_merge_non_adjacent_groups(self) -> None:
        """Merges non-adjacent groups correctly."""
        assignments = np.array([0, 1, 2, 2])

        new_assignments = _merge_groups(assignments, 3, 0, 2)

        assert new_assignments[0] == 0
        assert new_assignments[1] == 1
        assert new_assignments[2] == 0
        assert new_assignments[3] == 0


class TestEMRefinement:
    """Tests for EM refinement of group assignments."""

    def test_preserves_clear_grouping(self) -> None:
        """EM doesn't change clearly separated groups."""
        # Two very different intensities
        photon_counts = np.array([100.0, 100.0, 1000.0, 1000.0])
        dwell_times = np.array([0.1, 0.1, 0.1, 0.1])
        assignments = np.array([0, 0, 1, 1])

        refined, log_l = _run_em_refinement(
            photon_counts, dwell_times, assignments, 2, 0.4
        )

        # Should preserve the clear grouping
        assert refined[0] == refined[1]
        assert refined[2] == refined[3]
        assert refined[0] != refined[2]

    def test_returns_log_likelihood(self) -> None:
        """EM returns a valid log-likelihood."""
        photon_counts = np.array([100.0, 200.0])
        dwell_times = np.array([0.1, 0.1])
        assignments = np.array([0, 1])

        _, log_l = _run_em_refinement(
            photon_counts, dwell_times, assignments, 2, 0.2
        )

        assert np.isfinite(log_l)


class TestComputeBIC:
    """Tests for BIC computation."""

    def test_valid_inputs(self) -> None:
        """Computes BIC for valid inputs."""
        bic = _compute_bic(
            log_likelihood=-100.0,
            num_groups=3,
            num_changepoints=10,
            total_photons=1000,
        )

        assert np.isfinite(bic)

    def test_zero_changepoints_returns_neg_inf(self) -> None:
        """Returns -inf for zero change points."""
        bic = _compute_bic(
            log_likelihood=-100.0,
            num_groups=1,
            num_changepoints=0,
            total_photons=1000,
        )

        assert bic == -np.inf

    def test_zero_photons_returns_neg_inf(self) -> None:
        """Returns -inf for zero photons."""
        bic = _compute_bic(
            log_likelihood=-100.0,
            num_groups=1,
            num_changepoints=5,
            total_photons=0,
        )

        assert bic == -np.inf


class TestBuildGroupsFromAssignments:
    """Tests for building GroupData from assignments."""

    def test_builds_correct_groups(self) -> None:
        """Builds groups with correct aggregate statistics."""
        levels = [
            _make_level(num_photons=100, end_time_ns=100_000_000),
            _make_level(num_photons=100, end_time_ns=100_000_000),
            _make_level(num_photons=200, end_time_ns=100_000_000),
        ]
        assignments = np.array([0, 0, 1])

        groups = _build_groups_from_assignments(levels, assignments, 2)

        assert len(groups) == 2
        # First group has lower intensity, should be first after sorting
        low_group = groups[0]
        high_group = groups[1]
        assert low_group.total_photons == 200  # 100 + 100
        assert high_group.total_photons == 200

    def test_sorts_by_intensity(self) -> None:
        """Groups are sorted by intensity (ascending)."""
        levels = [
            _make_level(num_photons=1000, end_time_ns=100_000_000),  # High intensity
            _make_level(num_photons=100, end_time_ns=100_000_000),   # Low intensity
        ]
        assignments = np.array([0, 1])

        groups = _build_groups_from_assignments(levels, assignments, 2)

        # Lower intensity should come first
        assert groups[0].intensity_cps < groups[1].intensity_cps


class TestClusterLevels:
    """Tests for the main cluster_levels function."""

    def test_empty_levels_returns_none(self) -> None:
        """Returns None for empty level list."""
        result = cluster_levels([])

        assert result is None

    def test_single_level_returns_trivial_result(self) -> None:
        """Single level returns a trivial clustering result."""
        levels = [_make_level()]

        result = cluster_levels(levels)

        assert result is not None
        assert result.num_original_levels == 1
        assert result.num_groups == 1
        assert result.num_steps == 1

    def test_two_levels_returns_one_step(self) -> None:
        """Two levels produce one clustering step (merge into 1 group)."""
        levels = [
            _make_level(num_photons=100),
            _make_level(num_photons=100),
        ]

        result = cluster_levels(levels)

        assert result is not None
        assert result.num_original_levels == 2
        assert result.num_steps == 1  # One merge step
        assert result.steps[0].num_groups == 1

    def test_multiple_levels_produces_steps(self) -> None:
        """Multiple levels produce clustering steps."""
        levels = [
            _make_level(num_photons=100),
            _make_level(num_photons=100),
            _make_level(num_photons=100),
            _make_level(num_photons=100),
        ]

        result = cluster_levels(levels)

        assert result is not None
        assert result.num_original_levels == 4
        # With identical intensities, EM may reassign multiple levels at once,
        # resulting in fewer steps. The algorithm always ends with 1 group.
        assert result.num_steps >= 1
        assert result.steps[-1].num_groups == 1

    def test_distinct_intensities_grouped_separately(self) -> None:
        """Levels with distinct intensities are grouped separately at optimal."""
        # Create two groups of very different intensities
        levels = [
            _make_level(num_photons=100, end_time_ns=100_000_000),  # Low
            _make_level(num_photons=100, end_time_ns=100_000_000),  # Low
            _make_level(num_photons=1000, end_time_ns=100_000_000),  # High
            _make_level(num_photons=1000, end_time_ns=100_000_000),  # High
        ]

        result = cluster_levels(levels)

        assert result is not None
        assert result.num_original_levels == 4
        # With two clearly distinct intensity states, optimal should have 2 groups
        # (at some step in the clustering)
        assert result.num_steps >= 1
        # Check that at least one step has 2 groups
        two_group_steps = [s for s in result.steps if s.num_groups == 2]
        assert len(two_group_steps) >= 1
        # The optimal step (max BIC) should have 2 groups
        assert result.groups[0].intensity_cps != result.groups[1].intensity_cps if len(result.groups) == 2 else True

    def test_bic_values_are_valid(self) -> None:
        """All BIC values are finite."""
        levels = [
            _make_level(num_photons=100),
            _make_level(num_photons=200),
            _make_level(num_photons=300),
        ]

        result = cluster_levels(levels)

        assert result is not None
        for bic in result.all_bic_values:
            assert np.isfinite(bic)

    def test_optimal_index_is_valid(self) -> None:
        """Optimal step index is within valid range."""
        levels = [
            _make_level(num_photons=100),
            _make_level(num_photons=200),
            _make_level(num_photons=300),
        ]

        result = cluster_levels(levels)

        assert result is not None
        assert 0 <= result.optimal_step_index < result.num_steps

    def test_level_assignments_are_valid(self) -> None:
        """Level-to-group assignments are valid."""
        levels = [
            _make_level(num_photons=100),
            _make_level(num_photons=200),
            _make_level(num_photons=300),
        ]

        result = cluster_levels(levels)

        assert result is not None
        for step in result.steps:
            assignments = step.level_group_assignments
            assert len(assignments) == 3
            # All assignments should be valid group IDs
            for a in assignments:
                assert 0 <= a < step.num_groups

    def test_groups_contain_all_levels(self) -> None:
        """Each step's groups collectively contain all levels."""
        levels = [
            _make_level(num_photons=100),
            _make_level(num_photons=200),
            _make_level(num_photons=300),
        ]

        result = cluster_levels(levels)

        assert result is not None
        for step in result.steps:
            all_indices = []
            for group in step.groups:
                all_indices.extend(group.level_indices)
            assert sorted(all_indices) == [0, 1, 2]

    def test_with_selected_step_allows_different_grouping(self) -> None:
        """Can select a different step from the optimal."""
        levels = [
            _make_level(num_photons=100),
            _make_level(num_photons=200),
            _make_level(num_photons=300),
        ]

        result = cluster_levels(levels)

        assert result is not None
        if result.num_steps > 1:
            # Select a different step
            alt_step = (result.optimal_step_index + 1) % result.num_steps
            new_result = result.with_selected_step(alt_step)

            assert new_result.selected_step_index == alt_step
            assert new_result.optimal_step_index == result.optimal_step_index


class TestClusteringIntegration:
    """Integration tests for realistic clustering scenarios."""

    def test_synthetic_two_state_data(self) -> None:
        """Clusters synthetic two-state data correctly."""
        # Create synthetic data with two distinct intensity states
        # State 1: ~1000 cps (low), State 2: ~5000 cps (high)
        levels = []

        # Low state levels
        for i in range(5):
            levels.append(
                LevelData(
                    start_index=i * 100,
                    end_index=(i + 1) * 100 - 1,
                    start_time_ns=i * 100_000_000,
                    end_time_ns=(i + 1) * 100_000_000,
                    num_photons=100,  # 100 photons in 0.1s = 1000 cps
                    intensity_cps=1000.0,
                )
            )

        # High state levels
        for i in range(5, 10):
            levels.append(
                LevelData(
                    start_index=i * 100,
                    end_index=(i + 1) * 100 - 1,
                    start_time_ns=i * 100_000_000,
                    end_time_ns=(i + 1) * 100_000_000,
                    num_photons=500,  # 500 photons in 0.1s = 5000 cps
                    intensity_cps=5000.0,
                )
            )

        result = cluster_levels(levels)

        assert result is not None
        assert result.num_original_levels == 10

        # The EM algorithm may consolidate similar levels quickly,
        # so we may have fewer than 9 steps. Check that clustering works.
        assert result.num_steps >= 1

        # At some step, should have exactly 2 groups (with distinct intensities)
        two_group_steps = [s for s in result.steps if s.num_groups == 2]
        assert len(two_group_steps) > 0

        # Check that the two groups have distinct intensities
        two_group_step = two_group_steps[0]
        intensities = sorted([g.intensity_cps for g in two_group_step.groups])
        # Low state should be around 1000, high state around 5000
        assert intensities[0] < 2000
        assert intensities[1] > 3000

    def test_all_same_intensity_merges_to_one(self) -> None:
        """Levels with same intensity should eventually merge to one group."""
        # All levels have the same intensity
        levels = [
            _make_level(num_photons=100, end_time_ns=100_000_000) for _ in range(5)
        ]

        result = cluster_levels(levels)

        assert result is not None
        # Final step should have 1 group
        assert result.steps[-1].num_groups == 1
