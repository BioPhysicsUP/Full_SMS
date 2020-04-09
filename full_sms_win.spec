# -*- mode: python ; coding: utf-8 -*-
#  To compile run the command:
#  pyinstaller --noconfirm --windowed full_sms_win.spec
block_cipher = None


a = Analysis(['.\\src\\main.py'],
             pathex=['.\\src'],
             binaries=[('.\\src\\qwindows.dll', 'platforms')],
             datas=[('.\\src\\ui\\mainwindow.ui', '.\\ui'), ('.\\src\\tau data\\*', 'tau data'), ('.\\src\\Full-SMS.ico', '.')],
             hiddenimports=[],
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
          name='Full_SMS',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Full_SMS')
