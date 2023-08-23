from __future__ import annotations

__docformat__ = "NumPy"

from typing import TYPE_CHECKING

import pickle
import lzma

import h5pickle
from PyQt5.QtCore import QRunnable, pyqtSlot

import smsh5
from tree_model import DatasetTreeNode
from threads import WorkerSignals
from antibunching import AntibunchingAnalysis

if TYPE_CHECKING:
    from smsh5 import H5dataset
    from main import MainWindow

SAVING_VERSION = "1.07"
SAVE_FORMAT = "pickle"  # 'pickle' or 'lzma'


class SaveAnalysisWorker(QRunnable):
    def __init__(self, main_window: MainWindow, dataset: H5dataset):
        super().__init__()
        self.main_window = main_window
        self.dataset = dataset
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            save_analysis(main_window=self.main_window, dataset=self.dataset, signals=self.signals)
        except Exception as err:
            self.signals.error.emit(err)


def save_analysis(main_window: MainWindow, dataset: H5dataset, signals: WorkerSignals = None):
    if signals:
        signals.status_message.emit("Saving analysis...")
        signals.start_progress.emit(0)

    dataset_path = dataset.name
    dataset.unload_file()

    dataset.save_version = SAVING_VERSION

    dataset.save_selected = [node.checked() for node in main_window.part_nodes]
    dataset.settings = main_window.settings_dialog.settings
    dataset.global_settings = {
        "int_global_groups_checked": main_window.chbInt_Show_Global_Groups.isChecked(),
        "global_grouping_mode_selected": main_window.tabGroupingMode.currentWidget().objectName(),
    }

    save_file_name = dataset.name[:-2] + "smsa"

    if SAVE_FORMAT == "pickle":
        with open(save_file_name, "wb") as f:
            pickle.dump(dataset, f)
    elif SAVE_FORMAT == "lzma":
        with lzma.open(save_file_name, "wb") as f:
            pickle.dump(dataset, f)
    else:  # Default to pickle
        with open(save_file_name, "wb") as f:
            pickle.dump(dataset, f)

    reopened_file = h5pickle.File(dataset_path)
    if reopened_file.__bool__() is False:
        h5pickle.cache.clear()
        reopened_file = h5pickle.File(dataset_path)

    dataset.load_file(new_file=reopened_file)

    if signals:
        signals.status_message.emit("Done")
        signals.end_progress.emit()


class LoadAnalysisWorker(QRunnable):
    def __init__(self, main_window: MainWindow, file_path: str):
        super().__init__()
        self.main_window = main_window
        self.signals = WorkerSignals()
        self.file_path = file_path

    @pyqtSlot()
    def run(self) -> None:
        try:
            load_analysis(
                main_window=self.main_window,
                analysis_file=self.file_path,
                signals=self.signals,
            )
            self.signals.openfile_finished.emit(False)
        except Exception as err:
            self.signals.error.emit(err)


def load_analysis(main_window: MainWindow, analysis_file: str, signals: WorkerSignals = None):
    if signals:
        signals.status_message.emit("Loading analysis file...")
        signals.start_progress.emit(0)

    h5_file = h5pickle.File(analysis_file[:-4] + "h5")
    file_format = None
    with open(analysis_file, "rb") as f:
        peek_bytes = f.read(7)
    if len(peek_bytes) >= 7 and peek_bytes[0:7] == b"\xfd7zXZ\x00\x00":
        file_format = "lzma"
    elif len(peek_bytes) >= 1 and peek_bytes[0:1] == b"\x80":
        file_format = "pickle"

    if file_format == "pickle":
        with open(analysis_file, "rb") as f:
            loaded_dataset = pickle.load(f)
    elif file_format == "pickle":
        with lzma.open(analysis_file, "rb") as f:
            loaded_dataset = pickle.load(f)
    else:
        try:
            with lzma.open(analysis_file, "rb") as f:
                loaded_dataset = pickle.load(f)
        except lzma.LZMAError:
            try:
                with open(analysis_file, "rb") as f:
                    loaded_dataset = pickle.load(f)
            except pickle.UnpicklingError:
                raise TypeError("File type note known")

    loaded_dataset.load_file(h5_file)

    if not hasattr(loaded_dataset, "save_version") or loaded_dataset.save_version != SAVING_VERSION:
        if float(loaded_dataset.save_version) >= 1.05:
            for particle in loaded_dataset.particles:
                if float(loaded_dataset.save_version) <= 1.05:  # Added in version 1.06
                    particle.is_secondary_part = False
                    particle.tcspc_card = "TCSPC Card"
                    particle.sec_part = None
                if float(loaded_dataset.save_version) <= 1.06:  # Added in version 1.07
                    particle.ab_analysis = AntibunchingAnalysis(particle=particle)
        else:
            signals.save_file_version_outdated.emit()
            signals.status_message.emit("Done")
            signals.end_progress.emit()
            return

    for particle in loaded_dataset.particles:
        if not hasattr(particle, "roi_region"):
            particle.roi_region = (0, particle.abstimes[-1] / 1e9)
        if not hasattr(particle, "use_roi_for_grouping"):
            particle.ahca.use_roi_for_grouping = False
        if not hasattr(particle, "grouped_with_roi"):
            particle.ahca.grouped_with_roi = False
        if not hasattr(particle.ahca, "backup"):
            particle.ahca.backup = None
        if not hasattr(particle.ahca, "plots_need_to_be_updated"):
            particle.ahca.plots_need_to_be_updated = None

    dataset_node = DatasetTreeNode(analysis_file[analysis_file.rfind("/") + 1 : -3], loaded_dataset, "dataset")

    all_nodes = [(dataset_node, -1)]
    all_has_lifetimes = list()
    for i, particle in enumerate(loaded_dataset.particles):
        particlenode = DatasetTreeNode(particle.name, particle, "particle")
        all_nodes.append((particlenode, i))
        if hasattr(particlenode.dataobj.histogram, "fitted"):
            all_has_lifetimes.append(particlenode.dataobj.histogram.fitted)

    if loaded_dataset.has_irf:
        main_window.lifetime_controller.fitparam.irf = loaded_dataset.irf
        main_window.lifetime_controller.fitparam.irft = loaded_dataset.irf_t
        main_window.lifetime_controller.irf_loaded = True
        main_window.chbHasIRF.setChecked(True)
    main_window.add_all_nodes(all_nodes=all_nodes)
    for i, node in enumerate(main_window.part_nodes):
        node.setChecked(loaded_dataset.save_selected[i])
    dataset_node.setChecked(all([node.checked() for node in main_window.part_nodes]))
    num_checked = sum([node.checked() for node in main_window.part_nodes])
    main_window.lblNum_Selected.setText(str(num_checked))

    show_global_checked = loaded_dataset.global_settings["int_global_groups_checked"]
    main_window.chbInt_Show_Global_Groups.blockSignals(True)
    main_window.chbInt_Show_Global_Groups.setChecked(show_global_checked)
    main_window.chbInt_Show_Global_Groups.blockSignals(False)
    main_window.chbInt_Show_Groups.blockSignals(True)
    main_window.chbInt_Show_Groups.setChecked(not show_global_checked)
    main_window.chbInt_Show_Groups.blockSignals(False)

    grouping_mode_index = 1 if loaded_dataset.global_settings["global_grouping_mode_selected"] == "tabGlobal" else 0
    main_window.tabGroupingMode.blockSignals(True)
    main_window.tabGroupingMode.setCurrentIndex(grouping_mode_index)
    main_window.tabGroupingMode.blockSignals(False)

    if hasattr(loaded_dataset, "settings"):
        main_window.settings_dialog.load_settings(loaded_dataset.settings)

    main_window.data_loaded = True
    if signals:
        if any(all_has_lifetimes):
            signals.show_residual_widget.emit(True)
        signals.status_message.emit("Done")
        signals.end_progress.emit()


def convert_v1_05_to_v1_06(dataset_1_05: smsh5.H5dataset) -> smsh5.H5dataset:
    for particle in dataset_1_05.particles:
        particle.is_secondary_part = False
        particle.tcspc_card = "TCSPC Card"
        particle.sec_part = None
    return dataset_1_05
