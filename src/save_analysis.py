from __future__ import annotations
from typing import TYPE_CHECKING

import pickle
import copy
import os

import h5pickle
from tree_model import DatasetTreeNode

if TYPE_CHECKING:
    from smsh5 import H5dataset
    from main import MainWindow


def save_analysis(main_window: MainWindow, dataset: H5dataset):
    copy_dataset = copy.copy(dataset)
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
                all_group_levels = [step.group_levels for step in particle.ahca.steps]
                for group_levels in all_group_levels:
                    levels.extend(group_levels)
            for level in levels:
                level.microtimes._dataset = None
        if particle.has_spectra:
            particle.spectra.data = None

    if copy_dataset.has_raster_scans:
        for raster_scan in copy_dataset.all_raster_scans:
            raster_scan.dataset = None
    copy_dataset.save_selected = [node.checked() for node in main_window.part_nodes]

    save_file_name = copy_dataset.name[:-2] + 'smsa'
    with open(save_file_name, 'wb') as f:
        pickle.dump(copy_dataset, f)


def load_analysis(main_window: MainWindow, analysis_file: str):
    h5_file = h5pickle.File(analysis_file[:-4] + 'h5')
    with open(analysis_file, 'rb') as f:
        loaded_dataset = pickle.load(f)

    for particle in loaded_dataset.particles:
        particle.file = h5_file
        particle.datadict = h5_file[particle.name]
        particle.abstimes = particle.datadict['Absolute Times (ns)']
        particle.microtimes = particle.datadict['Micro Times (s)']
        particle.cpts._cpa._abstimes = particle.abstimes
        particle.cpts._cpa._microtimes = particle.microtimes
        if particle.has_levels:
            levels = particle.levels
            if particle.has_groups:
                all_group_levels = [step.group_levels for step in particle.ahca.steps]
                for group_levels in all_group_levels:
                    levels.extend(group_levels)
            for level in levels:
                level.microtimes._dataset = particle.microtimes
        if particle.has_spectra:
            particle.spectra.data = particle.datadict['Spectra (counts\\s)']

    if loaded_dataset.has_raster_scans:
        for raster_scan in loaded_dataset.all_raster_scans:
            first_particle_name = loaded_dataset.particles[raster_scan.particle_indexes[0]].name
            raster_scan.dataset = h5_file[first_particle_name + '/Raster Scan']

    loaded_dataset.name = analysis_file[:-4] + 'h5'

    datasetnode = DatasetTreeNode(analysis_file[analysis_file.rfind('/') + 1:-3],
                                  loaded_dataset, 'dataset')

    all_nodes = [(datasetnode, -1)]
    for i, particle in enumerate(loaded_dataset.particles):
        particlenode = DatasetTreeNode(particle.name, particle, 'particle')
        all_nodes.append((particlenode, i))

    main_window.add_all_nodes(all_nodes=all_nodes)
    for i, node in enumerate(main_window.part_nodes):
        node.setChecked(loaded_dataset.save_selected[i])
