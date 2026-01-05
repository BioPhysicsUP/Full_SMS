"""Tests for group and clustering data models."""

import pytest

from full_sms.models.group import ClusteringResult, GroupData


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


class TestClusteringResult:
    """Tests for ClusteringResult."""

    @pytest.fixture
    def sample_groups(self) -> tuple:
        """Create sample groups for testing."""
        return (
            GroupData(
                group_id=0,
                level_indices=(0, 2, 4),
                total_photons=300,
                total_dwell_time_s=0.3,
                intensity_cps=1000.0,
            ),
            GroupData(
                group_id=1,
                level_indices=(1, 3),
                total_photons=400,
                total_dwell_time_s=0.2,
                intensity_cps=2000.0,
            ),
        )

    def test_create_clustering_result(self, sample_groups: tuple) -> None:
        """Can create a ClusteringResult with valid parameters."""
        result = ClusteringResult(
            groups=sample_groups,
            all_bic_values=(100.5, 105.2, 102.3),
            optimal_step_index=1,
            selected_step_index=1,
            num_original_levels=5,
        )

        assert result.groups == sample_groups
        assert result.all_bic_values == (100.5, 105.2, 102.3)
        assert result.optimal_step_index == 1
        assert result.selected_step_index == 1
        assert result.num_original_levels == 5

    def test_num_groups(self, sample_groups: tuple) -> None:
        """num_groups returns count of groups at selected step."""
        result = ClusteringResult(
            groups=sample_groups,
            all_bic_values=(100.5,),
            optimal_step_index=0,
            selected_step_index=0,
            num_original_levels=5,
        )

        assert result.num_groups == 2

    def test_num_steps(self, sample_groups: tuple) -> None:
        """num_steps returns total number of clustering steps."""
        result = ClusteringResult(
            groups=sample_groups,
            all_bic_values=(100.5, 105.2, 102.3, 98.1),
            optimal_step_index=1,
            selected_step_index=1,
            num_original_levels=5,
        )

        assert result.num_steps == 4

    def test_optimal_bic(self, sample_groups: tuple) -> None:
        """optimal_bic returns BIC at optimal step."""
        result = ClusteringResult(
            groups=sample_groups,
            all_bic_values=(100.5, 105.2, 102.3),
            optimal_step_index=1,
            selected_step_index=0,
            num_original_levels=5,
        )

        assert result.optimal_bic == 105.2

    def test_selected_bic(self, sample_groups: tuple) -> None:
        """selected_bic returns BIC at selected step."""
        result = ClusteringResult(
            groups=sample_groups,
            all_bic_values=(100.5, 105.2, 102.3),
            optimal_step_index=1,
            selected_step_index=2,
            num_original_levels=5,
        )

        assert result.selected_bic == 102.3

    def test_is_optimal_selected_true(self, sample_groups: tuple) -> None:
        """is_optimal_selected returns True when indices match."""
        result = ClusteringResult(
            groups=sample_groups,
            all_bic_values=(100.5, 105.2),
            optimal_step_index=1,
            selected_step_index=1,
            num_original_levels=5,
        )

        assert result.is_optimal_selected is True

    def test_is_optimal_selected_false(self, sample_groups: tuple) -> None:
        """is_optimal_selected returns False when indices differ."""
        result = ClusteringResult(
            groups=sample_groups,
            all_bic_values=(100.5, 105.2),
            optimal_step_index=1,
            selected_step_index=0,
            num_original_levels=5,
        )

        assert result.is_optimal_selected is False

    def test_empty_groups_raises(self) -> None:
        """Raises ValueError for empty groups."""
        with pytest.raises(ValueError, match="groups cannot be empty"):
            ClusteringResult(
                groups=(),
                all_bic_values=(100.5,),
                optimal_step_index=0,
                selected_step_index=0,
                num_original_levels=5,
            )

    def test_empty_bic_values_raises(self, sample_groups: tuple) -> None:
        """Raises ValueError for empty all_bic_values."""
        with pytest.raises(ValueError, match="all_bic_values cannot be empty"):
            ClusteringResult(
                groups=sample_groups,
                all_bic_values=(),
                optimal_step_index=0,
                selected_step_index=0,
                num_original_levels=5,
            )

    def test_optimal_index_out_of_range_raises(self, sample_groups: tuple) -> None:
        """Raises ValueError for out-of-range optimal_step_index."""
        with pytest.raises(ValueError, match="optimal_step_index.*out of range"):
            ClusteringResult(
                groups=sample_groups,
                all_bic_values=(100.5, 105.2),
                optimal_step_index=5,
                selected_step_index=0,
                num_original_levels=5,
            )

    def test_selected_index_out_of_range_raises(self, sample_groups: tuple) -> None:
        """Raises ValueError for out-of-range selected_step_index."""
        with pytest.raises(ValueError, match="selected_step_index.*out of range"):
            ClusteringResult(
                groups=sample_groups,
                all_bic_values=(100.5, 105.2),
                optimal_step_index=0,
                selected_step_index=10,
                num_original_levels=5,
            )

    def test_zero_original_levels_raises(self, sample_groups: tuple) -> None:
        """Raises ValueError for non-positive num_original_levels."""
        with pytest.raises(ValueError, match="num_original_levels must be positive"):
            ClusteringResult(
                groups=sample_groups,
                all_bic_values=(100.5,),
                optimal_step_index=0,
                selected_step_index=0,
                num_original_levels=0,
            )
