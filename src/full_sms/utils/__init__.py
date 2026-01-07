"""Utility functions and helpers."""

from full_sms.utils.platform import (
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    configure_dpi_awareness,
    configure_multiprocessing,
    get_dpi_scale,
    get_gpu_backend_name,
    get_modifier_key_name,
    get_platform_name,
    get_shortcut_string,
)

__all__ = [
    "IS_LINUX",
    "IS_MACOS",
    "IS_WINDOWS",
    "configure_dpi_awareness",
    "configure_multiprocessing",
    "get_dpi_scale",
    "get_gpu_backend_name",
    "get_modifier_key_name",
    "get_platform_name",
    "get_shortcut_string",
]
