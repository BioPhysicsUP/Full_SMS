from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from smsh5 import H5dataset, Particle, RasterScan, Spectra
    import h5pickle
    import numpy as np


# File Attributes
def num_parts(dataset: H5dataset) -> np.int32:
    return dataset.file.attrs["# Particles"]


def file_version(dataset: H5dataset) -> str:
    if "Version" in dataset.file.attrs.keys():
        version = dataset.file.attrs["Version"]
    else:
        version = "1.0"
    return version


# File Groups
def particle(particle_num: int, dataset: H5dataset) -> h5pickle.Dataset:
    return dataset.file[f"Particle {particle_num+1}"]


# Particle Group Attributes
def date(particle: Particle) -> str:
    return particle.file_group["Date"]


def description(particle: Particle) -> str:
    if particle.file_version in ["1.0", "1.01", "1.02"]:
        descrip_text = particle.file_group.attrs["Discription"]
    else:
        descrip_text = particle.file_group.attrs["Description"]
    return descrip_text


def has_power_measurement(particle: Particle) -> bool:
    if particle.file_version in ["1.0", "1.01", "1.02"]:
        has_power = False
    else:
        has_power = bool(particle.file_group.attrs["Has Power Measurement?"])
    return has_power


def power_measurement(particle: Particle) -> np.ndarray:
    return particle.file_group.attrs["Power Measurement"]


def has_intensity(particle: Particle) -> bool:
    return bool(particle.file_group.attrs["Intensity?"])


def raster_scan_coord(particle: Particle) -> np.ndarray:
    return particle.file_group.attrs["RS Coord. (um)"]


# def has_spectra(particle: Particle) -> bool:
#     return bool(particle.datadict.attrs["Spectra?"])


def user(particle: Particle) -> str:
    return particle.file_group.attrs["User"]


# Particle Groups
def abstimes(particle: Particle) -> h5pickle.Dataset:
    return particle.file_group["Absolute Times (ns)"]


def abstimes2(particle: Particle) -> h5pickle.Dataset:
    if particle.file_version not in [
        "1.0",
        "1.01",
        "1.02",
        "1.03",
        "1.04",
        "1.05",
        "1.06",
    ]:
        abstimes_dataset = particle.file_group["Absolute Times 2 (ns)"]
    else:
        abstimes_dataset = None
    return abstimes_dataset


def microtimes(particle: Particle) -> h5pickle.Dataset:
    if particle.file_version in ["1.0", "1.01", "1.02"]:
        microtimes_dataset = particle.file_group["Micro Times (s)"]
    else:
        microtimes_dataset = particle.file_group["Micro Times (ns)"]
    return microtimes_dataset


def microtimes2(particle: Particle) -> h5pickle.Dataset:
    if particle.file_version not in [
        "1.0",
        "1.01",
        "1.02",
        "1.03",
        "1.04",
        "1.05",
        "1.06",
    ]:
        microtimes_dataset = particle.file_group["Micro Times 2 (ns)"]
    else:
        microtimes_dataset = None
    return microtimes_dataset


def tcspc_card(particle: Particle) -> str:
    if particle.file_version not in [
        "1.0",
        "1.01",
        "1.02",
        "1.03",
        "1.04",
        "1.05",
        "1.06",
    ]:
        if particle.is_secondary_part:
            tcspc_card = particle.file_group["Absolute Times 2 (ns)"].attrs["bh Card"]
        else:
            tcspc_card = particle.file_group["Absolute Times (ns)"].attrs["bh Card"]
    else:
        tcspc_card = None
    return tcspc_card


def has_raster_scan(particle: Union[Particle, h5pickle.Dataset]) -> bool:
    if str(type(particle)) == "<class 'smsh5.Particle'>":
        has_rs = "Raster Scan" in particle.file_group.keys()
    elif str(type(particle)) == "<class 'h5pickle.Group'>":
        has_rs = "Raster Scan" in particle.keys()
    else:
        raise TypeError("Type provided must be smsh5.Particle or smsh5.RasterScan")
    return has_rs


def raster_scan(particle: Union[Particle, h5pickle.Dataset]) -> h5pickle.Dataset:
    if str(type(particle)) == "<class 'smsh5.Particle'>":
        rs = particle.file_group["Raster Scan"]
    elif str(type(particle)) == "<class 'h5pickle.Group'>":
        rs = particle["Raster Scan"]
    else:
        raise TypeError("Type provided must be smsh5.Particle or smsh5.RasterScan")
    return rs


def has_spectra(particle: Particle) -> bool:
    return "Spectra (counts\s)" in particle.file_group.keys()


def spectra(particle: Particle) -> h5pickle.Dataset:
    return particle.file_group["Spectra (counts\s)"]


def int_trace(particle: Particle) -> h5pickle.Dataset:
    if particle.file_version not in ["1.0", "1.01", "1.02", "1.03"]:
        return particle.file_group["Intensity Trace (cps)"][0]


# Raster Scan Attributes
def __get_rs_dataset(part_or_rs: Union[Particle, RasterScan]) -> h5pickle.Dataset:
    if str(type(part_or_rs)) in [
        "<class 'smsh5.Particle'>",
        "<class 'h5pickle.Dataset'>",
    ]:
        raster_scan_dataset = raster_scan(particle=part_or_rs)
    elif str(type(part_or_rs)) == "<class 'smsh5.RasterScan'>":
        raster_scan_dataset = part_or_rs.dataset
    else:
        raise TypeError("Type provided must be smsh5.Particle or smsh5.RasterScan")
    return raster_scan_dataset


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
