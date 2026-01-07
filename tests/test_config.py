"""Tests for the config module."""

import json
import tempfile
from pathlib import Path

import pytest

from full_sms.config import (
    ChangePointSettings,
    DisplaySettings,
    LifetimeSettings,
    Settings,
)


class TestChangePointSettings:
    """Tests for ChangePointSettings dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = ChangePointSettings()
        assert settings.min_photons == 20
        assert settings.min_boundary_offset == 7

    def test_custom_values(self) -> None:
        """Test custom values."""
        settings = ChangePointSettings(min_photons=50, min_boundary_offset=10)
        assert settings.min_photons == 50
        assert settings.min_boundary_offset == 10


class TestLifetimeSettings:
    """Tests for LifetimeSettings dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = LifetimeSettings()
        assert settings.use_moving_avg is True
        assert settings.moving_avg_window == 10
        assert settings.start_percent == 80
        assert settings.end_multiple == 20
        assert settings.end_percent == 1
        assert settings.minimum_decay_window == 0.5
        assert settings.bg_percent == 5

    def test_custom_values(self) -> None:
        """Test custom values."""
        settings = LifetimeSettings(
            use_moving_avg=False,
            moving_avg_window=5,
            start_percent=70,
        )
        assert settings.use_moving_avg is False
        assert settings.moving_avg_window == 5
        assert settings.start_percent == 70


class TestDisplaySettings:
    """Tests for DisplaySettings dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        settings = DisplaySettings()
        assert settings.default_bin_size_ms == 10.0
        assert settings.auto_resolve_levels is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        settings = DisplaySettings(
            default_bin_size_ms=5.0,
            auto_resolve_levels=True,
        )
        assert settings.default_bin_size_ms == 5.0
        assert settings.auto_resolve_levels is True


class TestSettings:
    """Tests for Settings class."""

    def test_defaults(self) -> None:
        """Test default settings values."""
        settings = Settings()
        assert settings.change_point.min_photons == 20
        assert settings.lifetime.use_moving_avg is True
        assert settings.display.default_bin_size_ms == 10.0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        settings = Settings()
        d = settings.to_dict()

        assert "change_point" in d
        assert "lifetime" in d
        assert "display" in d
        assert d["change_point"]["min_photons"] == 20
        assert d["lifetime"]["use_moving_avg"] is True
        assert d["display"]["default_bin_size_ms"] == 10.0

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "change_point": {"min_photons": 30, "min_boundary_offset": 10},
            "lifetime": {"use_moving_avg": False, "moving_avg_window": 5},
            "display": {"default_bin_size_ms": 5.0, "auto_resolve_levels": True},
        }
        settings = Settings.from_dict(data)

        assert settings.change_point.min_photons == 30
        assert settings.change_point.min_boundary_offset == 10
        assert settings.lifetime.use_moving_avg is False
        assert settings.lifetime.moving_avg_window == 5
        assert settings.display.default_bin_size_ms == 5.0
        assert settings.display.auto_resolve_levels is True

    def test_from_dict_with_missing_keys(self) -> None:
        """Test deserialization handles missing keys with defaults."""
        data = {"change_point": {"min_photons": 30}}
        settings = Settings.from_dict(data)

        # Specified value used
        assert settings.change_point.min_photons == 30
        # Default used for missing key
        assert settings.change_point.min_boundary_offset == 7
        # Defaults used for missing sections
        assert settings.lifetime.use_moving_avg is True
        assert settings.display.default_bin_size_ms == 10.0

    def test_save_and_load(self) -> None:
        """Test round-trip save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"

            # Create settings with custom values
            original = Settings()
            original.change_point.min_photons = 50
            original.lifetime.use_moving_avg = False
            original.display.default_bin_size_ms = 20.0

            # Save
            original.save(path)
            assert path.exists()

            # Load
            loaded = Settings.load(path)
            assert loaded.change_point.min_photons == 50
            assert loaded.lifetime.use_moving_avg is False
            assert loaded.display.default_bin_size_ms == 20.0

    def test_load_nonexistent_file_returns_defaults(self) -> None:
        """Test loading from nonexistent file returns defaults."""
        path = Path("/nonexistent/path/settings.json")
        settings = Settings.load(path)

        assert settings.change_point.min_photons == 20
        assert settings.lifetime.use_moving_avg is True
        assert settings.display.default_bin_size_ms == 10.0

    def test_load_invalid_json_returns_defaults(self) -> None:
        """Test loading invalid JSON returns defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("invalid json {{{")

            settings = Settings.load(path)
            assert settings.change_point.min_photons == 20

    def test_reset_to_defaults(self) -> None:
        """Test resetting settings to defaults."""
        settings = Settings()
        settings.change_point.min_photons = 100
        settings.lifetime.use_moving_avg = False
        settings.display.default_bin_size_ms = 50.0

        settings.reset_to_defaults()

        assert settings.change_point.min_photons == 20
        assert settings.lifetime.use_moving_avg is True
        assert settings.display.default_bin_size_ms == 10.0

    def test_json_format(self) -> None:
        """Test that saved JSON is readable and properly formatted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"

            settings = Settings()
            settings.save(path)

            # Read and parse manually
            with open(path) as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert "change_point" in data
            assert "lifetime" in data
            assert "display" in data
