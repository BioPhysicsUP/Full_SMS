from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from smsh5 import H5dataset, Particle, RasterScan, Spectra
    import h5pickle
    import numpy as np


# File Attributes
def num_parts(dataset: H5dataset) -> np.int32:
    return dataset.file.attrs['# Particles']


def file_version(dataset: H5dataset) -> str:
    if 'Version' in dataset.file.attrs.keys():
        version = dataset.file.attrs['Version']
    else:
        version = '1.0'
    return version


# File Groups
def particle(particle_num: int, dataset: H5dataset) -> h5pickle.Dataset:
    return dataset.file[f"Particle {particle_num}"]


# Particle Group Attributes
def date(particle: Particle) -> str:
    return particle.datadict['Date']


def description(particle: Particle) -> str:
    if particle.file_version in ['1.0', '1.01', '1.02']:
        descrip_text = particle.datadict.attrs['Discription']
    else:
        descrip_text = particle.datadict.attrs['Description']
    return descrip_text


def has_power_measurement(particle: Particle) -> bool:
    if particle.file_version in ['1.0', '1.01', '1.02']:
        has_power = False
    else:
        has_power = bool(particle.datadict.attrs["Has Power Measurement?"])
    return has_power


def power_measurement(particle: Particle) -> np.ndarray:
    return particle.datadict.attrs["Power Measurement"]


def has_intensity(particle: Particle) -> bool:
    return bool(particle.datadict.attrs["Intensity?"])


def raster_scan_coord(particle: Particle) -> np.ndarray:
    return particle.datadict.attrs["RS Coord. (um)"]


def has_spectra(particle: Particle) -> bool:
    return bool(particle.datadict.attrs["Spectra?"])


def user(particle: Particle) -> str:
    return particle.datadict.attrs["User"]


#Particle Groups
def abstimes(particle: Particle) -> h5pickle.Dataset:
    return particle.datadict["Absolute Times (ns)"]


def microtimes(particle: Particle) -> h5pickle.Dataset:
    if particle.file_version in ['1.0', '1.01', '1.02']:
        microtimes_dataset = particle.datadict['Micro Times (s)']
    else:
        microtimes_dataset = particle.datadict['Micro Times (ns)']
    return microtimes_dataset


def has_raster_scan(particle: Union[Particle, h5pickle.Dataset]) -> bool:
    if TYPE_CHECKING:
        if type(particle) is Particle:
            has_rs = "Raster Scan" in particle.datadict.keys()
        elif type(particle) is h5pickle.Dataset:
            has_rs = "Raster Scan" in particle.keys()
    return


def raster_scan(particle: Particle) -> h5pickle.Dataset:
    return particle.datadict["Raster Scan"]


def has_spectra(particle: Particle) -> bool:
    return "Spectra (counts\s)" in particle.datadict.keys()


def spectra(particle: Particle) -> h5pickle.Dataset:
    return particle.datadict["Spectra (counts\s)"]


# Raster Scan Attributes
def __get_rs_dataset(part_or_rs: Union[Particle, RasterScan]) -> h5pickle.Dataset:
    if type(part_or_rs) is Particle:
        raster_scan_dataset = raster_scan(particle=part_or_rs)
    elif type(part_or_rs) is RasterScan:
        raster_scan_dataset = part_or_rs.dataset
    else:
        raise TypeError('Type provided must be smsh5.Particle or smsh5.RasterScan')

def rs_integration_time(part_or_rs: Union[Particle, RasterScan]) -> np.float64:
    raster_scan_dataset = __get_rs_dataset(part_or_rs=part_or_rs)
    return raster_scan_dataset.attrs["Int. Time (ms/um)"]


def rs_pixels_per_line(part_or_rs: Union[Particle, RasterScan]) -> np.int32:
    raster_scan_dataset = __get_rs_dataset(part_or_rs=part_or_rs)
    return raster_scan_dataset.attrs["Pixels per Line"]


def rs_range(part_or_rs: Union[Particle, RasterScan]) -> np.float64:
    raster_scan_dataset = __get_rs_dataset(part_or_rs=part_or_rs)
    return raster_scan_dataset.attrs["Range (um)"]


def rs_x_start(part_or_rs: Union[Particle, RasterScan]) -> np.float64:
    raster_scan_dataset = __get_rs_dataset(part_or_rs=part_or_rs)
    return raster_scan_dataset.attrs["XStart (um)"]


def rs_y_start(part_or_rs: Union[Particle, RasterScan]) -> np.float64:
    raster_scan_dataset = __get_rs_dataset(part_or_rs=part_or_rs)
    return raster_scan_dataset.attrs["YStart (um)"]


# Spectra Attributes
def spectra_exposure_time(particle: Particle) -> np.float32:
    spectra_dataset = spectra(particle=particle)
    return spectra_dataset.attrs["Exposure Times (s)"]


def spectra_abstimes(particle: Particle) -> np.ndarray:
    spectra_dataset = spectra(particle=particle)
    return spectra_dataset.attrs["Spectra Abs. Times (s)"]


def spectra_wavelengths(particle: Particle) -> np.ndarray:
    spectra_dataset = spectra(particle=particle)
    return spectra_dataset.attrs["Wavelengths"]
