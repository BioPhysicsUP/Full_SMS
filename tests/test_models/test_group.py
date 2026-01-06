"""Tests for group and clustering data models."""

import pytest

from full_sms.models.group import ClusteringResult, ClusteringStep, GroupData


class TestGroupData:
    """Tests for GroupData."""

    def test_create_group(self) -> None:
        """Can create a GroupData with valid parameters."""
        group = GroupData(
            group_id=0,
            level_indices=(0, 3, 5),
            total_photons=1000,
            total_dwell_time_s=0.5,
            intensity_cps=2000.0,
        )

        assert group.group_id == 0
        assert group.level_indices == (0, 3, 5)
        assert group.total_photons == 1000
        assert group.total_dwell_time_s == 0.5
        assert group.intensity_cps == 2000.0

    def test_num_levels(self) -> None:
        """num_levels returns count of levels in group."""
        group = GroupData(
            group_id=0,
            level_indices=(0, 1, 2, 3, 4),
            total_photons=500,
            total_dwell_time_s=1.0,
            intensity_cps=500.0,
        )

        assert group.num_levels == 5

    def test_single_level_group(self) -> None:
        """Can create a group with a single level."""
        group = GroupData(
            group_id=0,
            level_indices=(2,),
            total_photons=100,
            total_dwell_time_s=0.1,
            intensity_cps=1000.0,
        )

        assert group.num_levels == 1
        assert group.level_indices == (2,)

    def test_empty_level_indices_raises(self) -> None:
        """Raises ValueError for empty level_indices."""
        with pytest.raises(ValueError, match="level_indices cannot be empty"):
            GroupData(
                group_id=0,
                level_indices=(),
                total_photons=100,
                total_dwell_time_s=0.1,
                intensity_cps=1000.0,
            )

    def test_negative_photons_raises(self) -> None:
        """Raises ValueError for negative total_photons."""
        with pytest.raises(ValueError, match="total_photons must be non-negative"):
            GroupData(
                group_id=0,
                level_indices=(0,),
                total_photons=-100,
                total_dwell_time_s=0.1,
                intensity_cps=1000.0,
            )

    def test_negative_dwell_time_raises(self) -> None:
        """Raises ValueError for negative total_dwell_time_s."""
        with pytest.raises(ValueError, match="total_dwell_time_s must be non-negative"):
            GroupData(
                group_id=0,
                level_indices=(0,),
                total_photons=100,
                total_dwell_time_s=-0.1,
                intensity_cps=1000.0,
            )

    def test_negative_intensity_raises(self) -> None:
        """Raises ValueError for negative intensity_cps."""
        with pytest.raises(ValueError, match="intensity_cps must be non-negative"):
            GroupData(
                group_id=0,
                level_indices=(0,),
                total_photons=100,
                total_dwell_time_s=0.1,
                intensity_cps=-1000.0,
            )

    def test_immutable(self) -> None:
        """GroupData is immutable (frozen dataclass)."""
        group = GroupData(
            group_id=0,
            level_indices=(0, 1),
            total_photons=100,
            total_dwell_time_s=0.1,
            intensity_cps=1000.0,
        )

        with pytest.raises(AttributeError):
            group.total_photons = 200


class TestGroupDataFromLevelData:
    """Tests for GroupData.from_level_data factory method."""

    def test_from_level_data(self) -> None:
        """Can create GroupData from level information."""
        group = GroupData.from_level_data(
            group_id=0,
            level_indices=[0, 1, 2],
            level_photons=[100, 200, 300],
            level_dwell_times_s=[0.1, 0.2, 0.3],
        )

        assert group.group_id == 0
        assert group.level_indices == (0, 1, 2)
        assert group.total_photons == 600
        assert group.total_dwell_time_s == pytest.approx(0.6)
        assert group.intensity_cps == pytest.approx(1000.0)

    def test_from_level_data_single_level(self) -> None:
        """Can create GroupData from single level."""
        group = GroupData.from_level_data(
            group_id=5,
            level_indices=[10],
            level_photons=[500],
            level_dwell_times_s=[0.25],
        )

        assert group.group_id == 5
        assert group.level_indices == (10,)
        assert group.total_photons == 500
        assert group.total_dwell_time_s == 0.25
        assert group.intensity_cps == pytest.approx(2000.0)

    def test_from_level_data_zero_dwell(self) -> None:
        """Handles zero total dwell time."""
        group = GroupData.from_level_data(
            group_id=0,
            level_indices=[0],
            level_photons=[100],
            level_dwell_times_s=[0.0],
        )

        assert group.intensity_cps == 0.0

    def test_from_level_data_mismatched_lengths_raises(self) -> None:
        """Raises ValueError if input lists have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            GroupData.from_level_data(
                group_id=0,
                level_indices=[0, 1, 2],
                level_photons=[100, 200],  # Too short
                level_dwell_times_s=[0.1, 0.2, 0.3],
            )


class TestClusteringStep:
    """Tests for ClusteringStep."""

    def test_create_step(self) -> None:
        """Can create a ClusteringStep with valid parameters."""
        group = GroupData(
            group_id=0,
            level_indices=(0, 1),
            total_photons=200,
            total_dwell_time_s=0.2,
            intensity_cps=1000.0,
        )
        step = ClusteringStep(
            groups=(group,),
            level_group_assignments=(0, 0),
            bic=100.5,
            num_groups=1,
        )

        assert step.groups == (group,)
        assert step.level_group_assignments == (0, 0)
        assert step.bic == 100.5
        assert step.num_groups == 1

    def test_num_groups_mismatch_raises(self) -> None:
        """Raises ValueError if num_groups doesn't match len(groups)."""
        group = GroupData(
            group_id=0,
            level_indices=(0,),
            total_photons=100,
            total_dwell_time_s=0.1,
            intensity_cps=1000.0,
        )
        with pytest.raises(ValueError, match="num_groups.*doesn't match"):
            ClusteringStep(
                groups=(group,),
                level_group_assignments=(0,),
                bic=100.0,
                num_groups=2,  # Wrong
            )


def _make_clustering_result(
    num_steps: int = 2,
    num_original_levels: int = 3,
    optimal_step_index: int = 0,
    selected_step_index: int = 0,
) -> ClusteringResult:
    """Helper to create a ClusteringResult for testing."""
    steps = []
    for i in range(num_steps):
        num_groups = max(1, num_original_levels - i)
        groups = []
        assignments = [0] * num_original_levels

        # Create groups and assign levels
        for g in range(num_groups):
            groups.append(
                GroupData(
                    group_id=g,
                    level_indices=(g,) if g < num_original_levels else (0,),
                    total_photons=100,
                    total_dwell_time_s=0.1,
                    intensity_cps=1000.0,
                )
            )
            if g < num_original_levels:
                assignments[g] = g

        # If we have fewer groups than levels, merge extra levels into group 0
        if num_groups < num_original_levels:
            merged_indices = list(range(num_groups))
            for lvl in range(num_groups, num_original_levels):
                assignments[lvl] = 0
                merged_indices.append(lvl)
            groups[0] = GroupData(
                group_id=0,
                level_indices=tuple(sorted(merged_indices)),
                total_photons=100 * len(merged_indices),
                total_dwell_time_s=0.1 * len(merged_indices),
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


class TestClusteringResult:
    """Tests for ClusteringResult."""

    def test_create_clustering_result(self) -> None:
        """Can create a ClusteringResult with valid parameters."""
        result = _make_clustering_result(num_steps=3, optimal_step_index=1, selected_step_index=1)

        assert result.num_steps == 3
        assert result.optimal_step_index == 1
        assert result.selected_step_index == 1

    def test_num_groups(self) -> None:
        """num_groups returns count of groups at selected step."""
        result = _make_clustering_result(num_steps=2, num_original_levels=5, selected_step_index=0)

        # First step has 5 groups (one per level)
        assert result.num_groups == 5

    def test_num_steps(self) -> None:
        """num_steps returns total number of clustering steps."""
        result = _make_clustering_result(num_steps=4)

        assert result.num_steps == 4

    def test_optimal_bic(self) -> None:
        """optimal_bic returns BIC at optimal step."""
        result = _make_clustering_result(num_steps=3, optimal_step_index=1)

        # BIC is 100.0 + step_index * 5
        assert result.optimal_bic == 105.0

    def test_selected_bic(self) -> None:
        """selected_bic returns BIC at selected step."""
        result = _make_clustering_result(num_steps=3, selected_step_index=2)

        assert result.selected_bic == 110.0

    def test_is_optimal_selected_true(self) -> None:
        """is_optimal_selected returns True when indices match."""
        result = _make_clustering_result(optimal_step_index=1, selected_step_index=1)

        assert result.is_optimal_selected is True

    def test_is_optimal_selected_false(self) -> None:
        """is_optimal_selected returns False when indices differ."""
        result = _make_clustering_result(num_steps=3, optimal_step_index=1, selected_step_index=2)

        assert result.is_optimal_selected is False

    def test_all_bic_values(self) -> None:
        """all_bic_values returns BICs from all steps."""
        result = _make_clustering_result(num_steps=3)

        assert result.all_bic_values == (100.0, 105.0, 110.0)

    def test_groups_returns_selected_step_groups(self) -> None:
        """groups property returns groups at selected step."""
        result = _make_clustering_result(num_steps=2, num_original_levels=3, selected_step_index=0)

        # At step 0, we have 3 groups (one per level)
        assert len(result.groups) == 3

    def test_level_group_assignments(self) -> None:
        """level_group_assignments returns assignments at selected step."""
        result = _make_clustering_result(num_steps=2, num_original_levels=3, selected_step_index=0)

        assert len(result.level_group_assignments) == 3

    def test_get_groups_at_step(self) -> None:
        """get_groups_at_step returns groups at specified step."""
        result = _make_clustering_result(num_steps=3, num_original_levels=5)

        groups_0 = result.get_groups_at_step(0)
        groups_1 = result.get_groups_at_step(1)

        assert len(groups_0) == 5  # First step: 5 groups
        assert len(groups_1) == 4  # Second step: 4 groups

    def test_get_groups_at_step_out_of_range(self) -> None:
        """get_groups_at_step raises IndexError for invalid step."""
        result = _make_clustering_result(num_steps=2)

        with pytest.raises(IndexError, match="out of range"):
            result.get_groups_at_step(5)

    def test_with_selected_step(self) -> None:
        """with_selected_step returns new result with different selection."""
        result = _make_clustering_result(num_steps=3, selected_step_index=0)

        new_result = result.with_selected_step(2)

        assert result.selected_step_index == 0  # Original unchanged
        assert new_result.selected_step_index == 2  # New has different selection
        assert new_result.optimal_step_index == result.optimal_step_index  # Same optimal

    def test_with_selected_step_out_of_range(self) -> None:
        """with_selected_step raises IndexError for invalid step."""
        result = _make_clustering_result(num_steps=2)

        with pytest.raises(IndexError, match="out of range"):
            result.with_selected_step(10)

    def test_empty_steps_raises(self) -> None:
        """Raises ValueError for empty steps."""
        with pytest.raises(ValueError, match="steps cannot be empty"):
            ClusteringResult(
                steps=(),
                optimal_step_index=0,
                selected_step_index=0,
                num_original_levels=5,
            )

    def test_optimal_index_out_of_range_raises(self) -> None:
        """Raises ValueError for out-of-range optimal_step_index."""
        result = _make_clustering_result(num_steps=2)
        with pytest.raises(ValueError, match="optimal_step_index.*out of range"):
            ClusteringResult(
                steps=result.steps,
                optimal_step_index=5,
                selected_step_index=0,
                num_original_levels=3,
            )

    def test_selected_index_out_of_range_raises(self) -> None:
        """Raises ValueError for out-of-range selected_step_index."""
        result = _make_clustering_result(num_steps=2)
        with pytest.raises(ValueError, match="selected_step_index.*out of range"):
            ClusteringResult(
                steps=result.steps,
                optimal_step_index=0,
                selected_step_index=10,
                num_original_levels=3,
            )

    def test_zero_original_levels_raises(self) -> None:
        """Raises ValueError for non-positive num_original_levels."""
        result = _make_clustering_result(num_steps=1)
        with pytest.raises(ValueError, match="num_original_levels must be positive"):
            ClusteringResult(
                steps=result.steps,
                optimal_step_index=0,
                selected_step_index=0,
                num_original_levels=0,
            )
