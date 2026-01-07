"""Tests for configuration module."""

import sys
from pathlib import Path

import pytest

from full_sms.config import DEFAULT_CONFIG_PATH, Settings


class TestConfigPath:
    """Tests for platform-specific config paths."""

    def test_config_path_is_absolute(self):
        """Config path should be absolute."""
        assert DEFAULT_CONFIG_PATH.is_absolute()

    def test_config_path_ends_with_settings_json(self):
        """Config path should end with settings.json."""
        assert DEFAULT_CONFIG_PATH.name == "settings.json"

    def test_config_path_platform_appropriate(self):
        """Config path should be in platform-appropriate location."""
        path_str = str(DEFAULT_CONFIG_PATH)

        if sys.platform == "darwin":
            # macOS: ~/Library/Application Support/Full SMS/
            assert "Library" in path_str
            assert "Application Support" in path_str
            assert "Full SMS" in path_str
        elif sys.platform == "win32":
            # Windows: %APPDATA%\Full SMS\ or ~\AppData\Roaming\Full SMS\
            assert "AppData" in path_str or "Full SMS" in path_str
        else:
            # Linux: ~/.config/full_sms/ or XDG_CONFIG_HOME/full_sms/
            assert ".config" in path_str or "full_sms" in path_str

    def test_config_path_in_user_home(self):
        """Config path should be in user's home directory."""
        home = Path.home()
        # The config path should be under the user's home directory
        try:
            DEFAULT_CONFIG_PATH.relative_to(home)
        except ValueError:
            # On some systems, APPDATA might not be under home
            if sys.platform == "win32":
                # Allow Windows APPDATA to be elsewhere
                pass
            else:
                pytest.fail("Config path should be under user home directory")


class TestSettingsRoundTrip:
    """Tests for settings serialization."""

    def test_to_dict_and_from_dict(self):
        """Settings should round-trip through dict conversion."""
        original = Settings()
        original.change_point.min_photons = 50
        original.lifetime.use_moving_avg = False
        original.display.default_bin_size_ms = 25.0

        data = original.to_dict()
        restored = Settings.from_dict(data)

        assert restored.change_point.min_photons == 50
        assert restored.lifetime.use_moving_avg is False
        assert restored.display.default_bin_size_ms == 25.0

    def test_from_dict_with_missing_keys(self):
        """Settings should handle missing keys gracefully."""
        data = {"change_point": {"min_photons": 100}}
        settings = Settings.from_dict(data)

        assert settings.change_point.min_photons == 100
        # Other values should be defaults
        assert settings.lifetime.use_moving_avg is True
        assert settings.display.default_bin_size_ms == 10.0

    def test_reset_to_defaults(self):
        """reset_to_defaults should restore default values."""
        settings = Settings()
        settings.change_point.min_photons = 999
        settings.reset_to_defaults()

        assert settings.change_point.min_photons == 20
