from __future__ import annotations
import file_manager as fm
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QDialogButtonBox
from my_logger import setup_logger
import sys
import copy

if "--debug" in sys.argv:
    import ptvsd

settings_dialog_file = fm.path(name="settings_dialog.ui", file_type=fm.Type.UI)
UI_Settings_Dialog, _ = uic.loadUiType(settings_dialog_file)

class SettingsDialog(QDialog, UI_Settings_Dialog):

    def __init__(self, mainwindow):
        QDialog.__init__(self)
        UI_Settings_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.parent = mainwindow

        if "--debug" in sys.argv:
            ptvsd.debug_this_thread()

        self.settings = self.get_dialog_settings()

        self.default_settings = copy.deepcopy(self.settings)

        self.buttonBox.button(QDialogButtonBox.Reset).clicked.connect(
            self.reset_to_default)
        self.buttonBox.accepted.connect(self.accepted_callback)
        self.buttonBox.rejected.connect(self.rejected_callback)

    def get_dialog_settings(self) -> dict:
        cpa_min_num_photons = self.spbCPA_min_num_photons.value()
        cpa_min_boundary_offset = self.spbCPA_min_boundary_off.value()
        pb_min_dwell_time = self.dsbPB_min_dwell_time.value()
        pb_use_sigma_thresh = self.rdbPB_use_sigma.isChecked()
        pb_sigma_int_thresh = self.dsbPB_sigma_int_thresh.value()
        pb_defined_int_thresh = self.spbPB_defined_int_thresh.value()

        settings = {
                    "change_point_analysis": {
                        "min_num_photons": cpa_min_num_photons,
                        "min_boundary_offset": cpa_min_boundary_offset,
                    },
                    "photon_bursts": {
                        "min_level_dwell_time": pb_min_dwell_time,
                        "use_sigma_int_thresh": pb_use_sigma_thresh,
                        "sigma_int_thresh": pb_sigma_int_thresh,
                        "defined_int_thresh": pb_defined_int_thresh
                    }
                }
        return settings

    def reset_to_default(self) -> None:
        self.set_settings(settings = self.default_settings)

    def set_settings(self, settings: dict) -> None:
        cpa_min_num_photons = settings["change_point_analysis"]["min_num_photons"]
        cpa_min_boundary_offset = settings["change_point_analysis"]["min_boundary_offset"]
        pb_min_dwell_time = settings["photon_bursts"]["min_level_dwell_time"]
        pb_use_sigma_thresh = settings["photon_bursts"]["use_sigma_int_thresh"]
        pb_sigma_int_thresh = settings["photon_bursts"]["sigma_int_thresh"]
        pb_defined_int_thresh = settings["photon_bursts"]["defined_int_thresh"]

        self.spbCPA_min_num_photons.setValue(cpa_min_num_photons)
        self.spbCPA_min_boundary_off.setValue(cpa_min_boundary_offset)
        self.dsbPB_min_dwell_time.setValue(pb_min_dwell_time)
        self.rdbPB_use_sigma.setChecked(pb_use_sigma_thresh)
        self.dsbPB_sigma_int_thresh.setValue(pb_sigma_int_thresh)
        self.spbPB_defined_int_thresh.setValue(pb_defined_int_thresh)
    
    def accepted_callback(self):
        print('here')
        self.close()

    def rejected_callback(self):
        print('here')
        self.close()
