from __future__ import annotations

__docformat__ = 'NumPy'

from matplotlib.font_manager import json_dump
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog, QDialogButtonBox
from my_logger import setup_logger
import sys
import os
import copy
import json
from typing import Union, Any, TextIO
import file_manager as fm

logger = setup_logger(__name__)

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
        self.lt_use_moving_avg = None
        self.lt_moving_avg_window = None
        self.lt_start_percent = None
        self.lt_end_multiple = None
        self.lt_end_percent = None
        self.lt_minimum_decay_window = None
        self.lt_bg_percent = None

        self.default_settings_dict = {
                "change_point_analysis": {
                    "min_num_photons": 20,
                    "min_boundary_offset": 7
                },
                "photon_bursts": {
                    "min_level_dwell_time": 0.001,
                    "use_sigma_int_thresh": True,
                    "sigma_int_thresh": 3.0,
                    "defined_int_thresh": 5000
                },
                "lifetimes": {
                    "use_moving_avg": True,
                    "moving_avg_window": 10,
                    "start_percent": 80,
                    "end_multiple": 20,
                    "end_percent": 1,
                    "minimum_decay_window": 0.5,
                    "bg_percent": 5
                }
            }

        if load_file_or_path:
            try:
                self.load_settings_from_file(file_or_path=load_file_or_path)
            except json.JSONDecodeError and ValueError as err:
                print(err)
                self.set_all_dict(self.default_settings_dict)
        elif settings_dict:
            self.set_all_dict(settings_dict)
        else:
            self.set_all_dict(self.default_settings_dict)

        if save_file_or_path:
            self.save_settings_to_file(file_or_path=save_file_or_path)

    def get_default_settings(self):
        return Settings(settings_dict=self.default_settings_dict)

    def set_all_dict(self, settings_dict: dict):
        self.cpa_min_num_photons = settings_dict["change_point_analysis"]["min_num_photons"]
        self.cpa_min_boundary_offset = settings_dict["change_point_analysis"]["min_boundary_offset"]
        self.pb_min_dwell_time = settings_dict["photon_bursts"]["min_level_dwell_time"]
        self.pb_use_sigma_thresh = settings_dict["photon_bursts"]["use_sigma_int_thresh"]
        self.pb_sigma_int_thresh = settings_dict["photon_bursts"]["sigma_int_thresh"]
        self.pb_defined_int_thresh = settings_dict["photon_bursts"]["defined_int_thresh"]
        self.lt_use_moving_avg = settings_dict["lifetimes"]["use_moving_avg"]
        self.lt_moving_avg_window = settings_dict["lifetimes"]["moving_avg_window"]
        self.lt_start_percent = settings_dict["lifetimes"]["start_percent"]
        self.lt_end_multiple = settings_dict["lifetimes"]["end_multiple"]
        self.lt_end_percent = settings_dict["lifetimes"]["end_percent"]
        self.lt_minimum_decay_window = settings_dict["lifetimes"]["minimum_decay_window"]
        self.lt_bg_percent = settings_dict["lifetimes"]["bg_percent"]

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
                        },
                        "lifetimes": {
                            "start_percent": self.lt_start_percent,
                            "use_moving_avg": self.lt_use_moving_avg,
                            "moving_avg_window": self.lt_moving_avg_window,
                            "end_multiple": self.lt_end_multiple,
                            "end_percent": self.lt_end_percent,
                            "minimum_decay_window": self.lt_minimum_decay_window,
                            "bg_percent": self.lt_bg_percent,
                        }
                    }
        return settings_dict

    # def get_all_json_string(self):
    #     settings_dict = self.set_to_dict()
    #     return json.dumps(settings_dict)

    def get(self, setting_name: str) -> Any:
        return self.get_all_dict()[setting_name]

    def save_settings_to_file(self, file_or_path: Union[str, TextIO]):
        created_file = False
        if type(file_or_path) is str:
            assert os.path.exists(file_or_path), "Path provided does not exist."
            assert not os.path.isdir(file_or_path), "Path provided is not valid file."
            file = open(file_or_path, mode="w")
            created_file = True
        else:
            file = file_or_path
            assert file.closed is not True, "File provided is not open."

        settings_dict = self.get_all_dict()
        file.write(json.dumps(settings_dict, indent=4))

        if created_file:
            file.close()

    def load_settings_from_file(self, file_or_path: str):
        opened_file = False
        if type(file_or_path) is str:
            if not os.path.exists(file_or_path):
                raise FileNotFoundError(f"Path provided does not exist. {file_or_path}")
            if not os.path.isfile(file_or_path):
                raise FileNotFoundError(f"Path provided is not valid file. {file_or_path}")
            file = open(file_or_path, mode="r")
            opened_file = True
        else:
            file = file_or_path
            # assert file.closed is True, "File provided is not open."

        loaded_settings_dict = json.load(file)
        try:
            self.set_all_dict(settings_dict=loaded_settings_dict)
        except KeyError as err:
            self.set_all_dict(self.default_settings_dict)
            settings_file_path = fm.path('settings.json', fm.Type.ProjectRoot)
            self.save_settings_to_file(settings_file_path)

        if opened_file:
            file.close()


class SettingsDialog(QDialog, UI_Settings_Dialog):

    def __init__(self, mainwindow, current_settings: Settings = None,
                 get_saved_settings: bool = None):
        QDialog.__init__(self)
        UI_Settings_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.parent = mainwindow

        self.rdbPB_use_sigma.toggled.connect(self.pb_use_changed)
        self.rdbPB_use_defined_int.toggled.connect(self.pb_use_changed)

        settings_file_path = fm.path('settings.json', fm.Type.ProjectRoot)
        if get_saved_settings:
            self.settings = Settings(load_file_or_path=settings_file_path)
            self.set_dialog_settings(self.settings)
        elif current_settings is not None:
            self.settings = current_settings
        else:
            self.settings = Settings()

        with open(settings_file_path, 'w') as save_settings_file:
            self.settings.save_settings_to_file(file_or_path=save_settings_file)

        self.default_settings = self.settings.get_default_settings()

        self.buttonBox.button(QDialogButtonBox.Reset).clicked.connect(
            self.reset_to_default)
        self.buttonBox.accepted.connect(self.accepted_callback)
        self.buttonBox.rejected.connect(self.rejected_callback)
    
    def pb_use_changed(self):
        pb_use_sigma = self.rdbPB_use_sigma.isChecked()
        self.dsbPB_sigma_int_thresh.setEnabled(pb_use_sigma)
        self.spbPB_defined_int_thresh.setEnabled(not pb_use_sigma)

    def update_settings_from_dialog(self):
        cpa_min_num_photons = self.spbCPA_min_num_photons.value()
        cpa_min_boundary_offset = self.spbCPA_min_boundary_off.value()
        pb_min_dwell_time = self.dsbPB_min_dwell_time.value()
        pb_use_sigma_thresh = self.rdbPB_use_sigma.isChecked()
        pb_sigma_int_thresh = self.dsbPB_sigma_int_thresh.value()
        pb_defined_int_thresh = self.spbPB_defined_int_thresh.value()
        lt_use_moving_avg = self.chb_use_moving_avg.isChecked()
        lt_moving_avg_window = self.spb_moving_avg_window.value()
        lt_start_percent = self.spb_start_percent.value()
        lt_end_multiple = self.spb_end_multiple.value()
        lt_end_percent = self.spb_end_percent.value()
        lt_minimum_decay_window = self.dsb_minimum_decay_window.value()
        lt_bg_percent = self.spb_bg_percent.value()

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
                    },
                    "lifetimes": {
                        "use_moving_avg": lt_use_moving_avg,
                        "moving_avg_window": lt_moving_avg_window,
                        "start_percent": lt_start_percent,
                        "end_multiple": lt_end_multiple,
                        "end_percent": lt_end_percent,
                        "minimum_decay_window": lt_minimum_decay_window,
                        "bg_percent": lt_bg_percent,
                    }
        }
        self.settings.set_all_dict(new_settings_dict)

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
        self.chb_use_moving_avg.setChecked(settings.lt_use_moving_avg)
        self.spb_moving_avg_window.setValue(settings.lt_moving_avg_window)
        self.spb_start_percent.setValue(settings.lt_start_percent)
        self.spb_end_multiple.setValue(settings.lt_end_multiple)
        self.spb_end_percent.setValue(settings.lt_end_percent)
        self.dsb_minimum_decay_window.setValue(settings.lt_minimum_decay_window)
        self.spb_bg_percent.setValue(settings.lt_bg_percent)

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
