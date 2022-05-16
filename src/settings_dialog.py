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

if "--vscode" in sys.argv:
    import ptvsd

settings_dialog_file = fm.path(name="settings_dialog.ui", file_type=fm.Type.UI)
UI_Settings_Dialog, _ = uic.loadUiType(settings_dialog_file)


class Settings:

    def __init__(self, 
                 save_file_or_path: str = None,
                 settings_dict: dict = None,
                 load_file_or_path: str = None):
        self.cpa_min_num_photons = None
        self.cpa_min_boundary_offset = None
        self.pb_min_dwell_time = None
        self.pb_use_sigma_thresh = None
        self.pb_sigma_int_thresh = None
        self.pb_defined_int_thresh = None

        if load_file_or_path:
            self.load_settings_from_file(file_or_path=load_file_or_path)
        elif settings_dict:
            self.set_all_dict(settings_dict)
        
        if save_file_or_path:
            self.save_settings_to_file(file_or_path=save_file_or_path)
    
    def set_all_dict(self, settings_dict: dict):
        self.cpa_min_num_photons = settings_dict["change_point_analysis"]["min_num_photons"]
        self.cpa_min_boundary_offset = settings_dict["change_point_analysis"]["min_boundary_offset"]
        self.pb_min_dwell_time = settings_dict["photon_bursts"]["min_level_dwell_time"]
        self.pb_use_sigma_thresh = settings_dict["photon_bursts"]["use_sigma_int_thresh"]
        self.pb_sigma_int_thresh = settings_dict["photon_bursts"]["sigma_int_thresh"]
        self.pb_defined_int_thresh = settings_dict["photon_bursts"]["defined_int_thresh"]
    
    def get_all_dict(self) -> dict:
        settings_dict = {
                        "change_point_analysis": {
                            "min_num_photons": self.cpa_min_num_photons,
                            "min_boundary_offset": self.cpa_min_boundary_offset,
                        },
                        "photon_bursts": {
                            "min_level_dwell_time": self.pb_min_dwell_time,
                            "use_sigma_int_thresh": self.pb_use_sigma_thresh,
                            "sigma_int_thresh": self.pb_sigma_int_thresh,
                            "defined_int_thresh": self.pb_defined_int_thresh
                        }
                    }
        return settings_dict

    def get_all_json_string(self):
        settings_dict = self.set_to_dict()
        return json.dumps(settings_dict)

    def get(self, setting_name: str) -> Any:
        return self.get_all_dict()[setting_name]
    
    def save_settings_to_file(self, file_or_path: str): 
        created_file = False
        if type(file_or_path) is str:
            assert os.path.exists(file_or_path), "Path provided does not exist."
            assert os.path.isdir(file_or_path), "Path provided is not valid directory."
            file = open(file_or_path, mode="w")
            created_file = True
        else:
            file = file_or_path
            assert file.closed != True, "File provided is not open."
        
        settings_dict = self.get_all_dict()
        file.write(json.dumps(settings_dict, indent=4))

        if created_file:
            file.close()

    def load_settings_from_file(self, file_or_path: str):
        opened_file = False
        if type(file_or_path) is str:
            assert os.path.exists(file_or_path), "Path provided does not exist."
            assert os.path.isfile(file_or_path), "Path provided is not valid file."
            file = open(file_or_path, mode="r")
            opened_file = True
        else:
            file = file_or_path
            assert file.closed != True, "File provided is not open."
        
        loaded_settings_dict = json.load(file)
        self.set_all_dict(settings_dict=loaded_settings_dict)

        if opened_file:
            file.close()


class SettingsDialog(QDialog, UI_Settings_Dialog):

    def __init__(self, mainwindow, current_settings: Settings = None):
        QDialog.__init__(self)
        UI_Settings_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.parent = mainwindow

        if "--vscode" in sys.argv:
            ptvsd.debug_this_thread()
        
        self.rdbPB_use_sigma.toggled.connect(self.pb_use_changed)
        self.rdbPB_use_defined_int.toggled.connect(self.pb_use_changed)

        if current_settings == None:
            self.settings = self.get_dialog_settings()
        else:
            self.settings = current_settings
        
        save_settings_file_path = fm.path('settings.json', fm.Type.ProjectRoot)
        with open(save_settings_file_path, 'w') as save_settings_file:
            self.settings.save_settings_to_file(file_or_path=save_settings_file)

        self.default_settings = copy.deepcopy(self.settings)

        self.buttonBox.button(QDialogButtonBox.Reset).clicked.connect(
            self.reset_to_default)
        self.buttonBox.accepted.connect(self.accepted_callback)
        self.buttonBox.rejected.connect(self.rejected_callback)
    
    def pb_use_changed(self):
        pb_use_sigma = self.rdbPB_use_sigma.isChecked()
        self.dsbPB_sigma_int_thresh.setEnabled(pb_use_sigma)
        self.spbPB_defined_int_thresh.setEnabled(not pb_use_sigma)

    def get_dialog_settings(self) -> Settings:
        cpa_min_num_photons = self.spbCPA_min_num_photons.value()
        cpa_min_boundary_offset = self.spbCPA_min_boundary_off.value()
        pb_min_dwell_time = self.dsbPB_min_dwell_time.value()
        pb_use_sigma_thresh = self.rdbPB_use_sigma.isChecked()
        pb_sigma_int_thresh = self.dsbPB_sigma_int_thresh.value()
        pb_defined_int_thresh = self.spbPB_defined_int_thresh.value()

        new_settings_dict = {
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
        new_settings = Settings(settings_dict=new_settings_dict)

        return new_settings

    def set_dialog_settings(self, settings: Settings = None) -> None:
        if settings is None:
            settings = self.settings
        self.spbCPA_min_num_photons.setValue(settings.cpa_min_num_photons)
        self.spbCPA_min_boundary_off.setValue(settings.cpa_min_boundary_offset)
        self.dsbPB_min_dwell_time.setValue(settings.pb_min_dwell_time)
        self.rdbPB_use_sigma.setChecked(settings.pb_use_sigma_thresh)
        self.rdbPB_use_sigma.setChecked(not settings.pb_use_sigma_thresh)
        self.dsbPB_sigma_int_thresh.setValue(settings.pb_sigma_int_thresh)
        self.spbPB_defined_int_thresh.setValue(settings.pb_defined_int_thresh)

    def update_settings_from_dialog(self) -> None:
        self.settings = self.get_dialog_settings()

    def load_settings(self, settings: Settings):
        self.settings = settings
        self.set_dialog_settings()

    def save_settings_to_file(self):
        settings_file_path = fm.path('settings.json', fm.Type.ProjectRoot)
        with open(settings_file_path, 'w') as settings_file:
            self.settings.save_settings_to_file(file_or_path=settings_file)

    def reset_to_default(self) -> None:
        self.settings = copy.deepcopy(self.default_settings)
        self.set_dialog_settings()
        self.save_settings_to_file()
    
    def accepted_callback(self):
        self.update_settings_from_dialog()
        self.save_settings_to_file()
        self.close()

    def rejected_callback(self):
        self.set_dialog_settings()
        self.close()
