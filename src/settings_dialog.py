from __future__ import annotations
import file_manager as fm
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog
from my_logger import setup_logger

settings_dialog_file = fm.path(name="settings_dialog.ui", file_type=fm.Type.UI)
UI_Settings_Dialog, _ = uic.loadUiType(settings_dialog_file)

class SettingsDialog(QDialog, UI_Settings_Dialog):

    def __init__(self, mainwindow):
        QDialog.__init__(self)
        UI_Settings_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow