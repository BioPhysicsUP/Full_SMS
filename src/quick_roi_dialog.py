from __future__ import annotations

# from matplotlib.font_manager import json_dump
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
# from my_logger import setup_logger
# import sys
# import os
# import copy
# import json
# from typing import Union, Any
import file_manager as fm

quick_roi_dialog_file = fm.path(name="quick_roi_dialog.ui", file_type=fm.Type.UI)
UI_Quick_ROI_Dialog, _ = uic.loadUiType(quick_roi_dialog_file)


class QuickROIDialog(QDialog, UI_Quick_ROI_Dialog):

    def __init__(self, mainwindow):
        QDialog.__init__(self)
        UI_Quick_ROI_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.parent = mainwindow

        print('here')
        self.rdbManual.toggled.connect(self.mode_changed)

        self.btnResetCurrent.clicked.connect(self.gui_reset_roi_current)
        self.btnResetSelected.clicked.connect(self.gui_reset_roi_selected)
        self.btnResetAll.clicked.connect(self.gui_reset_roi_all)

        self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(self.accepted_callback)
        self.buttonBox.rejected.connect(self.rejected_callback)

        self.should_trim_traces = False

    def mode_changed(self):
        is_manual = self.rdbManual.isChecked()
        self.gpbManual.setEnabled(is_manual)
        self.gpbAuto.setEnabled(not is_manual)

    def gui_reset_roi_current(self):
        self.reset_roi(mode='current')

    def gui_reset_roi_selected(self):
        self.reset_roi(mode='selected')

    def gui_reset_roi_all(self):
        self.reset_roi(mode='all')

    def reset_roi(self, mode: str = 'all'):
        if mode == 'current':
            particles = [self.mainwindow.current_particle]
        elif mode == 'selected':
            particles = self.mainwindow.get_checked_particles()
        elif mode == 'all':
            particles = self.mainwindow.current_dataset.particles
        else:
            return

        for particle in particles:
            particle.roi_region = (0, particle.abstimes[-1], None)

        self.mainwindow.display_data()

    def accepted_callback(self):
        particles = None
        if self.rdbCurrent.isChecked():
            particles = [self.mainwindow.current_particle]
        elif self.rdbSelected.isChecked():
            particles = self.mainwindow.get_checked_particles()
        elif self.rdbAll.isChecked():
            particles = self.mainwindow.current_dataset.particles

        for particle in particles:
            if self.rdbManual.isChecked():
                trimmed = particle.trim_trace(min_level_int=self.spbManual_Min_Int.value(),
                                              min_level_dwell_time=self.dsbManual_Min_Time.value(),
                                              reset_roi=self.chbReset_ROI.isChecked())
                if trimmed is False and self.chbUncheck_If_Not_Valid.isChecked():
                    self.mainwindow.set_particle_check_state(particle.dataset_ind, False)
        self.mainwindow.lifetime_controller.test_need_roi_apply()
        self.mainwindow.intensity_controller.plot_all()

        self.close()

    def rejected_callback(self):
        self.close()
