"""Application configuration and settings.

Provides persistent settings storage with JSON serialization.
Settings are automatically saved when modified and loaded on startup.

Platform-specific config locations:
- macOS: ~/Library/Application Support/Full SMS/settings.json
- Windows: %APPDATA%\\Full SMS\\settings.json
- Linux: ~/.config/full_sms/settings.json (XDG Base Directory spec)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_config_dir() -> Path:
    """Get the platform-specific configuration directory.

    Returns:
        Path to the configuration directory for the current platform.
    """
    if sys.platform == "darwin":
        # macOS: ~/Library/Application Support/Full SMS
        return Path.home() / "Library" / "Application Support" / "Full SMS"
    elif sys.platform == "win32":
        # Windows: %APPDATA%\Full SMS
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Full SMS"
        # Fallback if APPDATA not set
        return Path.home() / "AppData" / "Roaming" / "Full SMS"
    else:
        # Linux and other Unix: ~/.config/full_sms (XDG spec)
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / "full_sms"
        return Path.home() / ".config" / "full_sms"


# Default config file location
DEFAULT_CONFIG_PATH = _get_config_dir() / "settings.json"


@dataclass
class ChangePointSettings:
    """Settings for change point analysis.

    Attributes:
        min_photons: Minimum photons in a segment to analyze.
        min_boundary_offset: Minimum distance from segment edges.
    """

    min_photons: int = 20
    min_boundary_offset: int = 7


@dataclass
class LifetimeSettings:
    """Settings for lifetime fitting.

    Attributes:
        use_moving_avg: Whether to use moving average smoothing.
        moving_avg_window: Window size for moving average.
        start_percent: Percentage of max for auto start detection.
        end_multiple: Multiple of decay time for auto end.
        end_percent: Percentage of max for auto end detection.
        minimum_decay_window: Minimum decay window in ns.
        bg_percent: Percentage of data to use for background estimation.
    """

    use_moving_avg: bool = True
    moving_avg_window: int = 10
    start_percent: int = 80
    end_multiple: int = 20
    end_percent: int = 1
    minimum_decay_window: float = 0.5
    bg_percent: int = 5


@dataclass
class DisplaySettings:
    """Settings for display and visualization.

    Attributes:
        default_bin_size_ms: Default bin size in milliseconds.
        auto_resolve_levels: Automatically resolve levels on file load.
    """

    default_bin_size_ms: float = 10.0
    auto_resolve_levels: bool = False


@dataclass
class FileDialogSettings:
    """Settings for file dialogs.

    Attributes:
        last_open_directory: Last directory used for opening files.
        last_session_directory: Last directory used for session files.
    """

    last_open_directory: str = ""
    last_session_directory: str = ""


@dataclass
class Settings:
    """Application settings container.

    All settings are grouped into logical categories:
    - change_point: Settings for change point analysis
    - lifetime: Settings for lifetime fitting
    - display: Settings for display and visualization
    - file_dialogs: Settings for file dialog paths

    Settings are persisted to JSON and loaded on startup.
    """

    change_point: ChangePointSettings = field(default_factory=ChangePointSettings)
    lifetime: LifetimeSettings = field(default_factory=LifetimeSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    file_dialogs: FileDialogSettings = field(default_factory=FileDialogSettings)

    def to_dict(self) -> dict:
        """Convert settings to a dictionary for serialization.

        Returns:
            Dictionary representation of all settings.
        """
        return {
            "change_point": asdict(self.change_point),
            "lifetime": asdict(self.lifetime),
            "display": asdict(self.display),
            "file_dialogs": asdict(self.file_dialogs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        """Create Settings from a dictionary.

        Args:
            data: Dictionary with settings data.

        Returns:
            Settings instance populated from the dictionary.
        """
        settings = cls()

        if "change_point" in data:
            cp = data["change_point"]
            settings.change_point = ChangePointSettings(
                min_photons=cp.get("min_photons", 20),
                min_boundary_offset=cp.get("min_boundary_offset", 7),
            )

        if "lifetime" in data:
            lt = data["lifetime"]
            settings.lifetime = LifetimeSettings(
                use_moving_avg=lt.get("use_moving_avg", True),
                moving_avg_window=lt.get("moving_avg_window", 10),
                start_percent=lt.get("start_percent", 80),
                end_multiple=lt.get("end_multiple", 20),
                end_percent=lt.get("end_percent", 1),
                minimum_decay_window=lt.get("minimum_decay_window", 0.5),
                bg_percent=lt.get("bg_percent", 5),
            )

        if "display" in data:
            disp = data["display"]
            settings.display = DisplaySettings(
                default_bin_size_ms=disp.get("default_bin_size_ms", 10.0),
                auto_resolve_levels=disp.get("auto_resolve_levels", False),
            )

        if "file_dialogs" in data:
            fd = data["file_dialogs"]
            settings.file_dialogs = FileDialogSettings(
                last_open_directory=fd.get("last_open_directory", ""),
                last_session_directory=fd.get("last_session_directory", ""),
            )

        return settings

    def save(self, path: Optional[Path] = None) -> None:
        """Save settings to a JSON file.

        Args:
            path: Path to save settings. Defaults to DEFAULT_CONFIG_PATH.
        """
        if path is None:
            path = DEFAULT_CONFIG_PATH

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Settings saved to {path}")
        except OSError as e:
            logger.error(f"Failed to save settings to {path}: {e}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Settings:
        """Load settings from a JSON file.

        Args:
            path: Path to load settings from. Defaults to DEFAULT_CONFIG_PATH.

        Returns:
            Settings instance. Returns defaults if file doesn't exist or is invalid.
        """
        if path is None:
            path = DEFAULT_CONFIG_PATH

        if not path.exists():
            logger.info(f"No settings file at {path}, using defaults")
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)
            settings = cls.from_dict(data)
            logger.info(f"Settings loaded from {path}")
            return settings
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load settings from {path}: {e}, using defaults")
            return cls()

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        self.change_point = ChangePointSettings()
        self.lifetime = LifetimeSettings()
        self.display = DisplaySettings()
        self.file_dialogs = FileDialogSettings()


# Global settings instance - loaded on module import
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance.

    Loads from disk on first access.

    Returns:
        The global Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def save_settings() -> None:
    """Save the global settings to disk."""
    if _settings is not None:
        _settings.save()
