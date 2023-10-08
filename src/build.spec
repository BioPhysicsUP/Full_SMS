# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['./src/main.py'],
    pathex=[],
    binaries=[],
    datas=[('./src/resources/', './resources/')],
    hiddenimports=['icons_rc', 'numexpr'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['numba', 'progressbar2', 'jupyter', 'dfply', 'auto-py-to-exe', 'Pympler', 'pyinstaller-versionfile'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Full_SMS',
    debug=False,
    # bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='./versionfile.txt',
    icon=['./src/resources/icons/Full-SMS.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Full_SMS',
)
