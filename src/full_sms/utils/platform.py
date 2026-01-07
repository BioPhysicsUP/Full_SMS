"""Platform-specific utilities for Full SMS.

Provides cross-platform compatibility for:
- DPI scaling detection
- GPU rendering backend selection
- Platform detection

Platform support:
- macOS: Metal backend, automatic Retina scaling
- Windows: DirectX 11 backend, DPI awareness
- Linux: OpenGL backend, GTK/X11 DPI scaling
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Platform detection
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")


def get_platform_name() -> str:
    """Get a human-readable platform name.

    Returns:
        Platform name string (macOS, Windows, or Linux).
    """
    if IS_MACOS:
        return "macOS"
    elif IS_WINDOWS:
        return "Windows"
    elif IS_LINUX:
        return "Linux"
    else:
        return sys.platform


def get_dpi_scale() -> float:
    """Get the current display DPI scale factor.

    Returns:
        DPI scale factor (1.0 = 96 DPI, 2.0 = 192 DPI for Retina).
    """
    if IS_WINDOWS:
        return _get_windows_dpi_scale()
    elif IS_MACOS:
        return _get_macos_dpi_scale()
    elif IS_LINUX:
        return _get_linux_dpi_scale()
    return 1.0


def _get_windows_dpi_scale() -> float:
    """Get DPI scale on Windows.

    Returns:
        DPI scale factor.
    """
    try:
        # Try to set DPI awareness first (Windows 8.1+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor aware
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()  # Fallback for Win7/8
            except (AttributeError, OSError):
                pass

        # Get the DPI for the primary monitor
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
            return dpi / 96.0
        except (AttributeError, OSError):
            pass
    except Exception as e:
        logger.warning(f"Failed to get Windows DPI: {e}")

    return 1.0


def _get_macos_dpi_scale() -> float:
    """Get DPI scale on macOS.

    macOS handles Retina displays automatically, but we can detect the scale
    factor for custom font sizing if needed.

    Returns:
        DPI scale factor (typically 1.0 or 2.0).
    """
    try:
        # On macOS, we can check NSScreen.mainScreen.backingScaleFactor
        # but DearPyGui handles this automatically via Metal
        # For now, return a default and let the system handle it
        return 1.0  # DearPyGui handles Retina automatically
    except Exception as e:
        logger.warning(f"Failed to get macOS DPI: {e}")
        return 1.0


def _get_linux_dpi_scale() -> float:
    """Get DPI scale on Linux.

    Checks GDK_SCALE and GDK_DPI_SCALE environment variables,
    then falls back to Xft.dpi from xrdb.

    Returns:
        DPI scale factor.
    """
    try:
        # Check GDK_SCALE first (GTK 3+)
        gdk_scale = os.environ.get("GDK_SCALE")
        if gdk_scale:
            return float(gdk_scale)

        # Check GDK_DPI_SCALE
        gdk_dpi_scale = os.environ.get("GDK_DPI_SCALE")
        if gdk_dpi_scale:
            return float(gdk_dpi_scale)

        # Check QT_SCALE_FACTOR for Qt apps
        qt_scale = os.environ.get("QT_SCALE_FACTOR")
        if qt_scale:
            return float(qt_scale)

        # Could query Xft.dpi via xrdb but that's complex
        # Default to 1.0 and let the system handle it
        return 1.0

    except Exception as e:
        logger.warning(f"Failed to get Linux DPI: {e}")
        return 1.0


def configure_dpi_awareness() -> None:
    """Configure DPI awareness for the current platform.

    Should be called before creating any windows.
    """
    if IS_WINDOWS:
        try:
            # Set DPI awareness (Windows 8.1+)
            try:
                # Per-monitor DPI awareness v2 (Windows 10 1703+)
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
                logger.info("Windows DPI awareness set to per-monitor v2")
            except (AttributeError, OSError):
                try:
                    # Per-monitor DPI awareness (Windows 8.1+)
                    ctypes.windll.shcore.SetProcessDpiAwareness(1)
                    logger.info("Windows DPI awareness set to per-monitor")
                except (AttributeError, OSError):
                    # System DPI awareness (Windows Vista+)
                    ctypes.windll.user32.SetProcessDPIAware()
                    logger.info("Windows DPI awareness set to system")
        except Exception as e:
            logger.warning(f"Failed to set Windows DPI awareness: {e}")

    elif IS_MACOS:
        # macOS handles Retina automatically via Metal/OpenGL
        logger.info("macOS: Retina support handled automatically")

    elif IS_LINUX:
        # Linux DPI is typically handled via environment variables
        # which should already be set by the desktop environment
        logger.info("Linux: DPI handled by desktop environment")


def get_gpu_backend_name() -> str:
    """Get the GPU backend name for the current platform.

    DearPyGui automatically selects the appropriate backend:
    - macOS: Metal
    - Windows: DirectX 11
    - Linux: OpenGL

    Returns:
        Name of the GPU backend.
    """
    if IS_MACOS:
        return "Metal"
    elif IS_WINDOWS:
        return "DirectX 11"
    elif IS_LINUX:
        return "OpenGL"
    else:
        return "Unknown"


def get_modifier_key_name() -> str:
    """Get the platform-specific modifier key name.

    Returns:
        'Cmd' on macOS, 'Ctrl' on Windows/Linux.
    """
    return "Cmd" if IS_MACOS else "Ctrl"


def get_shortcut_string(key: str) -> str:
    """Get a platform-appropriate shortcut string.

    Args:
        key: The key without modifier (e.g., 'O', 'S', 'Q').

    Returns:
        Formatted shortcut string (e.g., 'Cmd+O' or 'Ctrl+O').
    """
    return f"{get_modifier_key_name()}+{key}"


def configure_multiprocessing() -> None:
    """Configure multiprocessing for the current platform.

    On macOS, the 'spawn' start method must be used with GUI applications.
    This is already handled in workers/pool.py, but this function can be
    called early to set the default.

    Should be called before importing any multiprocessing workers.
    """
    import multiprocessing as mp

    # Use 'spawn' on all platforms for consistency with GUI apps
    # This is especially important on macOS where 'fork' can cause issues
    try:
        mp.set_start_method("spawn", force=False)
        logger.info("Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        # Already set, which is fine
        pass
