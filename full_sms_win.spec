# -*- mode: python ; coding: utf-8 -*-
#
#  NOTE:
#  The developer version of PyInstaller is needed to allow PyQT5 to use the more modern
#  Widows Vista style. To install the developer version, run the following command:
#
#  pip install https://github.com/pyinstaller/pyinstaller/archive/develop.zip
#
#  It ran successfully with PyInstaller version 4.0.dev0+b3dd91c8a8
#
#  To compile run the command:
#  pyinstaller --noconfirm --clean --windowed full_sms_win.spec
#
# NOTE
# ----
# If using UPX, find the vcruntime.dll used in the enviroment, and copy it over the VCRUNTIME.dll after build

block_cipher = None


a = Analysis(['.\\src\\main.py'],
             pathex=['.\\src'],
             binaries=[('.\\src\\qwindows.dll', 'platforms')],
             datas=[('.\\src\\all_sums.pickle', '.'), ('.\\src\\ui\\mainwindow.ui', '.\\ui'), ('.\\src\\tau data\\*', 'tau data'), ('.\\src\\Full-SMS.ico', '.')],
             hiddenimports=['scipy.stats'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Full SMS',
          icon='.\\src\\Full-SMS.ico',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Full SMS')
