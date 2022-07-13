from __future__ import annotations

from matplotlib.font_manager import json_dump
import file_manager as fm
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
from my_logger import setup_logger
import sys
import os
import copy
import json
from typing import Union, Any
import file_manager as fm

trim_traces_dialog_file = fm.path(name="trim_traces_dialog.ui", file_type=fm.Type.UI)
UI_Trim_Traces_Dialog, _ = uic.loadUiType(trim_traces_dialog_file)


class TrimTracesDialog(QDialog, UI_Trim_Traces_Dialog):

    def __init__(self, mainwindow):
        QDialog.__init__(self)
        UI_Trim_Traces_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.parent = mainwindow

        print('here')
        self.rdbManual.toggled.connect(self.mode_changed)

        self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(self.accepted_callback)
        self.buttonBox.rejected.connect(self.rejected_callback)

        self.should_trim_traces = False

    def mode_changed(self):
        is_manual = self.rdbManual.isChecked()
        self.gpbManual.setEnabled(is_manual)
        self.gpbAuto.setEnabled(not is_manual)

    def accepted_callback(self):
        self.should_trim_traces = True
        self.close()

    def rejected_callback(self):
        self.close()
