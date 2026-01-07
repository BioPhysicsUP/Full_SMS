"""Tests for platform-specific utilities."""

import sys

import pytest

from full_sms.utils.platform import (
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    get_dpi_scale,
    get_gpu_backend_name,
    get_modifier_key_name,
    get_platform_name,
    get_shortcut_string,
)


class TestPlatformDetection:
    """Tests for platform detection constants."""

    def test_exactly_one_platform_detected(self):
        """Only one platform flag should be True."""
        flags = [IS_MACOS, IS_WINDOWS, IS_LINUX]
        # At least one should be true (or all false for unknown platforms)
        assert sum(flags) <= 1

    def test_platform_detection_matches_sys_platform(self):
        """Platform detection should match sys.platform."""
        if sys.platform == "darwin":
            assert IS_MACOS is True
            assert IS_WINDOWS is False
            assert IS_LINUX is False
        elif sys.platform == "win32":
            assert IS_MACOS is False
            assert IS_WINDOWS is True
            assert IS_LINUX is False
        elif sys.platform.startswith("linux"):
            assert IS_MACOS is False
            assert IS_WINDOWS is False
            assert IS_LINUX is True


class TestGetPlatformName:
    """Tests for get_platform_name()."""

    def test_returns_string(self):
        """Should return a string."""
        name = get_platform_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_known_platform_names(self):
        """Should return known platform names for common platforms."""
        name = get_platform_name()
        if sys.platform == "darwin":
            assert name == "macOS"
        elif sys.platform == "win32":
            assert name == "Windows"
        elif sys.platform.startswith("linux"):
            assert name == "Linux"


class TestGetGpuBackendName:
    """Tests for get_gpu_backend_name()."""

    def test_returns_string(self):
        """Should return a string."""
        backend = get_gpu_backend_name()
        assert isinstance(backend, str)
        assert len(backend) > 0

    def test_known_backends(self):
        """Should return appropriate backend for each platform."""
        backend = get_gpu_backend_name()
        if sys.platform == "darwin":
            assert backend == "Metal"
        elif sys.platform == "win32":
            assert backend == "DirectX 11"
        elif sys.platform.startswith("linux"):
            assert backend == "OpenGL"


class TestGetDpiScale:
    """Tests for get_dpi_scale()."""

    def test_returns_float(self):
        """Should return a float."""
        scale = get_dpi_scale()
        assert isinstance(scale, float)

    def test_positive_value(self):
        """Should return a positive value."""
        scale = get_dpi_scale()
        assert scale > 0

    def test_reasonable_range(self):
        """Should return a reasonable DPI scale (0.5 to 4.0)."""
        scale = get_dpi_scale()
        assert 0.5 <= scale <= 4.0


class TestGetModifierKeyName:
    """Tests for get_modifier_key_name()."""

    def test_returns_string(self):
        """Should return a string."""
        name = get_modifier_key_name()
        assert isinstance(name, str)

    def test_platform_appropriate_name(self):
        """Should return Cmd on macOS, Ctrl elsewhere."""
        name = get_modifier_key_name()
        if sys.platform == "darwin":
            assert name == "Cmd"
        else:
            assert name == "Ctrl"


class TestGetShortcutString:
    """Tests for get_shortcut_string()."""

    def test_includes_modifier(self):
        """Should include the modifier key."""
        shortcut = get_shortcut_string("O")
        if sys.platform == "darwin":
            assert shortcut == "Cmd+O"
        else:
            assert shortcut == "Ctrl+O"

    def test_various_keys(self):
        """Should work with various keys."""
        for key in ["O", "S", "Q", "A", "R", "E"]:
            shortcut = get_shortcut_string(key)
            assert key in shortcut
            assert "+" in shortcut
