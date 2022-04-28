from __future__ import annotations
from typing import TYPE_CHECKING

import pickle
import copy
import os
import lzma

import h5pickle
from PyQt5.QtCore import QRunnable, pyqtSlot


from tree_model import DatasetTreeNode
from threads import WorkerSignals
import smsh5_file_reader as h5_fr

if TYPE_CHECKING:
    from smsh5 import H5dataset
    from main import MainWindow


SAVING_VERSION = '1.04'


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

    copy_dataset = copy.deepcopy(dataset)
    copy_dataset.save_version = SAVING_VERSION
    copy_dataset.file = None
    for particle in copy_dataset.particles:
        particle.file = None
        particle.datadict = None
        particle.abstimes = None
        particle.microtimes = None
        particle.cpts._cpa._abstimes = None
        particle.cpts._cpa._microtimes = None
        if particle.has_levels:
            levels = particle.levels
            if particle.has_groups:
                particle.all_group_levels = [step.group_levels for step in particle.ahca.steps]
                # for group_levels in all_group_levels:
                #     levels.extend(group_levels)
            for level in levels:
                level.microtimes._dataset = None
        if particle.has_spectra:
            particle.spectra.data = None

    if copy_dataset.has_raster_scans:
        for raster_scan in copy_dataset.all_raster_scans:
            raster_scan.dataset = None
    copy_dataset.save_selected = [node.checked() for node in main_window.part_nodes]
    copy_dataset.settings = main_window.settings_dialog.settings

    save_file_name = copy_dataset.name[:-2] + 'smsa'
    # with open(save_file_name, 'wb') as f:
    with lzma.open(save_file_name, 'wb') as f:
        pickle.dump(copy_dataset, f)

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
            load_analysis(main_window=self.main_window, analysis_file=self.file_path,
                          signals = self.signals)
            self.signals.openfile_finished.emit(False)
        except Exception as err:
            self.signals.error.emit(err)


def load_analysis(main_window: MainWindow, analysis_file: str, signals: WorkerSignals = None):
    if signals:
        signals.status_message.emit("Loading analysis file...")
        signals.start_progress.emit(0)

    h5_file = h5pickle.File(analysis_file[:-4] + 'h5')
    with lzma.open(analysis_file, 'rb') as f:
        loaded_dataset = pickle.load(f)

    if not hasattr(loaded_dataset, 'save_version') or loaded_dataset.save_version != SAVING_VERSION:
        signals.save_file_version_outdated.emit()
        signals.status_message.emit("Done")
        signals.end_progress.emit()
        return

    for particle in loaded_dataset.particles:
        particle.file = h5_file
        particle.datadict = h5_file[particle.name]
        particle.abstimes = h5_fr.abstimes(particle=particle)
        particle.microtimes = h5_fr.microtimes(particle=particle)
        particle.cpts._cpa._abstimes = particle.abstimes
        particle.cpts._cpa._microtimes = particle.microtimes
        if particle.has_levels:
            levels = particle.levels
            if particle.has_groups:
                for i, step in enumerate(particle.ahca.steps):
                    step.group_levels = particle.all_group_levels[i]
                particle.all_group_levels = None
                # all_group_levels = [step.group_levels for step in particle.ahca.steps]
                # for group_levels in all_group_levels:
                #     levels.extend(group_levels)
            for level in levels:
                level.microtimes._dataset = particle.microtimes
        if particle.has_spectra:
            particle.spectra.data = h5_fr.spectra(particle=particle)
        if not hasattr(particle, 'roi_region'):
            particle.roi_region = (0, particle.abstimes[-1])

    if loaded_dataset.has_raster_scans:
        for raster_scan in loaded_dataset.all_raster_scans:
            first_particle = loaded_dataset.particles[raster_scan.particle_indexes[0]]
            raster_scan.dataset = h5_fr.raster_scan(particle=first_particle)

    loaded_dataset.name = analysis_file[:-4] + 'h5'

    datasetnode = DatasetTreeNode(analysis_file[analysis_file.rfind('/') + 1:-3],
                                  loaded_dataset, 'dataset')

    all_nodes = [(datasetnode, -1)]
    all_has_lifetimes = list()
    for i, particle in enumerate(loaded_dataset.particles):
        particlenode = DatasetTreeNode(particle.name, particle, 'particle')
        all_nodes.append((particlenode, i))
        if hasattr(particlenode.dataobj.histogram, 'fitted'):
            all_has_lifetimes.append(particlenode.dataobj.histogram.fitted)

    if loaded_dataset.has_irf:
        main_window.lifetime_controller.fitparam.irf = loaded_dataset.irf
        main_window.lifetime_controller.fitparam.irft = loaded_dataset.irf_t
        main_window.lifetime_controller.irf_loaded = True
        main_window.chbHasIRF.setChecked(True)
    main_window.add_all_nodes(all_nodes=all_nodes)
    for i, node in enumerate(main_window.part_nodes):
        node.setChecked(loaded_dataset.save_selected[i])
    num_checked = sum([node.checked() for node in main_window.part_nodes])
    main_window.lblNum_Selected.setText(str(num_checked))

    if hasattr(loaded_dataset, 'settings'):
        main_window.settings_dialog.load_settings(loaded_dataset.settings)

    main_window.data_loaded = True
    if signals:
        if any(all_has_lifetimes):
            signals.show_residual_widget.emit(True)
        signals.status_message.emit("Done")
        signals.end_progress.emit()
