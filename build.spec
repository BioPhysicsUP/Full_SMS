# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for Full SMS.

Build command:
    uv run pyinstaller build.spec

The application is bundled as a one-folder distribution for easier debugging
and faster startup on most platforms.

Platform-specific notes:
- macOS: Creates a .app bundle in dist/Full_SMS.app
- Windows: Creates dist/Full_SMS/Full_SMS.exe
- Linux: Creates dist/Full_SMS/Full_SMS
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Determine platform-specific settings
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

# Project paths
PROJECT_ROOT = Path(SPECPATH)
SRC_DIR = PROJECT_ROOT / "src" / "full_sms"
RESOURCES_DIR = SRC_DIR / "resources"
DATA_DIR = SRC_DIR / "data"

# Icon paths
if IS_WINDOWS:
    ICON_PATH = str(RESOURCES_DIR / "icons" / "Full-SMS.ico")
elif IS_MACOS:
    # macOS prefers .icns but can use .png
    ICON_PATH = str(RESOURCES_DIR / "icons" / "Full-SMS.png")
else:
    ICON_PATH = str(RESOURCES_DIR / "icons" / "Full-SMS.png")

# Collect data files
datas = [
    # Tau data files for change point analysis precomputed sums
    (str(DATA_DIR / "tau_data"), "full_sms/data/tau_data"),
    # Application icon
    (str(RESOURCES_DIR / "icons"), "full_sms/resources/icons"),
]

# Collect DearPyGui data files (fonts, etc.)
datas += collect_data_files("dearpygui")

# Hidden imports - scientific computing packages with submodules
hiddenimports = [
    # NumPy and SciPy
    "numpy",
    "numpy.core._methods",
    "numpy.lib.format",
    "scipy",
    "scipy.special",
    "scipy.special._cdflib",
    "scipy.optimize",
    "scipy.stats",
    "scipy.signal",
    "scipy.ndimage",
    # HDF5
    "h5py",
    "h5py.defs",
    "h5py.utils",
    "h5py.h5ac",
    "h5py._proxy",
    # Numba (JIT compiler)
    "numba",
    "numba.core",
    # Matplotlib (for plot export)
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    # Logging
    "logging.handlers",
    # JSON for session files
    "json",
    # Standard library for multiprocessing
    "multiprocessing",
    "concurrent.futures",
]

# Add all numba submodules (it has many)
hiddenimports += collect_submodules("numba")

# Excludes - packages we don't need
excludes = [
    # Testing frameworks
    "pytest",
    "pytest_cov",
    # Jupyter/IPython
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "ipython",
    "ipykernel",
    "notebook",
    # Development tools
    "sphinx",
    "numpydoc",
    # Qt (we're using DearPyGui, not Qt)
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    # Tkinter (not used)
    "tkinter",
    "_tkinter",
]


a = Analysis(
    [str(SRC_DIR / "app.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

if IS_MACOS:
    # macOS: Create an app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="Full_SMS",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,  # No console on macOS
        disable_windowed_traceback=False,
        argv_emulation=True,  # Allow drag-and-drop files on macOS
        target_arch=None,  # Universal binary if possible
        codesign_identity=None,
        entitlements_file=None,
        icon=ICON_PATH,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="Full_SMS",
    )
    app = BUNDLE(
        coll,
        name="Full_SMS.app",
        icon=ICON_PATH,
        bundle_identifier="com.up-biophysics.full-sms",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "0.8.0",
            "CFBundleVersion": "0.8.0",
            "NSPrincipalClass": "NSApplication",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "HDF5 File",
                    "CFBundleTypeExtensions": ["h5", "hdf5"],
                    "CFBundleTypeRole": "Viewer",
                },
                {
                    "CFBundleTypeName": "SMS Analysis",
                    "CFBundleTypeExtensions": ["smsa"],
                    "CFBundleTypeRole": "Editor",
                },
            ],
        },
    )
else:
    # Windows/Linux: Create a folder distribution
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="Full_SMS",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,  # Keep console for debugging on Windows/Linux
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=ICON_PATH if IS_WINDOWS else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="Full_SMS",
    )
