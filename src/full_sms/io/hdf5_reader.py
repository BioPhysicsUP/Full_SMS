"""HDF5 file reader for SMS data files.

This module reads single-molecule spectroscopy data from HDF5 files
in the format produced by the UP Biophysics SMS acquisition software.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from numpy.typing import NDArray

from full_sms.models.particle import ChannelData, ParticleData
from full_sms.models.session import FileMetadata


# File versions that predate certain features
_OLD_VERSIONS_NO_DUAL_CHANNEL = frozenset(
    ["1.0", "1.01", "1.02", "1.03", "1.04", "1.05", "1.06"]
)
_OLD_VERSIONS_DESCRIPTION_TYPO = frozenset(["1.0", "1.01", "1.02"])
_OLD_VERSIONS_MICROTIMES_SECONDS = frozenset(["1.0", "1.01", "1.02"])


def _get_file_version(h5file: h5py.File) -> str:
    """Get the file format version."""
    if "Version" in h5file.attrs:
        return str(h5file.attrs["Version"])
    return "1.0"


def _get_particle_names(h5file: h5py.File) -> List[str]:
    """Get sorted particle group names from the file.

    Particles are named "Particle 1", "Particle 2", etc.
    This function returns them in natural numeric order.
    """
    all_keys = list(h5file.keys())
    part_keys = [key for key in all_keys if key.startswith("Particle ")]

    # Natural sort by extracting the number
    def extract_number(name: str) -> int:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    return sorted(part_keys, key=extract_number)


def _get_description(particle_group: h5py.Group, version: str) -> str:
    """Get particle description, handling version differences."""
    if version in _OLD_VERSIONS_DESCRIPTION_TYPO:
        # Old versions had a typo in the attribute name
        return str(particle_group.attrs.get("Discription", ""))
    return str(particle_group.attrs.get("Description", ""))


def _get_tcspc_card(
    particle_group: h5py.Group, version: str, is_secondary: bool = False
) -> str:
    """Get TCSPC card identifier."""
    if version in _OLD_VERSIONS_NO_DUAL_CHANNEL:
        return "unknown"

    dataset_name = "Absolute Times 2 (ns)" if is_secondary else "Absolute Times (ns)"
    if dataset_name in particle_group:
        dataset = particle_group[dataset_name]
        if "bh Card" in dataset.attrs:
            return str(dataset.attrs["bh Card"])
    return "unknown"


def _determine_channelwidth(microtimes: NDArray[np.float64]) -> float:
    """Determine TCSPC channel width from microtime data.

    The channel width is determined by finding the smallest common
    difference between sorted, unique microtime values.

    Args:
        microtimes: Array of microtime values in nanoseconds.

    Returns:
        Channel width in nanoseconds. Returns default if cannot be determined.
    """
    default_channelwidth = 0.01220703125  # Common default for bh cards

    if len(microtimes) == 0:
        return default_channelwidth

    # Find unique differences in sorted microtimes
    sorted_times = np.sort(microtimes)
    differences = np.diff(sorted_times)

    # Get unique differences
    unique_diffs = np.unique(differences)

    # Filter out very small values (rounding errors)
    unique_diffs = unique_diffs[unique_diffs > 1e-4]

    if len(unique_diffs) == 0:
        return default_channelwidth

    # Find the GCD-like smallest difference
    possible_widths = np.unique(np.diff(np.unique(differences)))
    possible_widths = possible_widths[possible_widths > 1e-4]

    if len(possible_widths) == 0:
        return unique_diffs[0] if len(unique_diffs) > 0 else default_channelwidth

    return float(possible_widths[0])


def _read_channel_data(
    particle_group: h5py.Group, version: str, is_secondary: bool = False
) -> Optional[ChannelData]:
    """Read channel data (abstimes and microtimes) from a particle group.

    Args:
        particle_group: The HDF5 group for the particle.
        version: File format version string.
        is_secondary: Whether to read the secondary channel (channel 2).

    Returns:
        ChannelData if the channel exists, None otherwise.
    """
    if is_secondary:
        abstimes_name = "Absolute Times 2 (ns)"
        if version in _OLD_VERSIONS_MICROTIMES_SECONDS:
            microtimes_name = "Micro Times 2 (s)"
        else:
            microtimes_name = "Micro Times 2 (ns)"
    else:
        abstimes_name = "Absolute Times (ns)"
        if version in _OLD_VERSIONS_MICROTIMES_SECONDS:
            microtimes_name = "Micro Times (s)"
        else:
            microtimes_name = "Micro Times (ns)"

    # Check if datasets exist
    if abstimes_name not in particle_group:
        return None
    if microtimes_name not in particle_group:
        return None

    # Read the data
    abstimes = np.array(particle_group[abstimes_name], dtype=np.uint64)
    microtimes = np.array(particle_group[microtimes_name], dtype=np.float64)

    # Convert old format (seconds) to nanoseconds
    if version in _OLD_VERSIONS_MICROTIMES_SECONDS:
        microtimes = microtimes * 1e9

    return ChannelData(abstimes=abstimes, microtimes=microtimes)


def _has_spectra(particle_group: h5py.Group) -> bool:
    """Check if particle has spectra data."""
    return "Spectra (counts\\s)" in particle_group


def _has_raster_scan(particle_group: h5py.Group) -> bool:
    """Check if particle has raster scan data."""
    return "Raster Scan" in particle_group


def load_h5_file(path: Path | str) -> Tuple[FileMetadata, List[ParticleData]]:
    """Load an HDF5 file and return metadata and particle data.

    This function reads single-molecule spectroscopy data from an HDF5 file
    in the format produced by the UP Biophysics SMS acquisition software.

    Args:
        path: Path to the HDF5 file.

    Returns:
        A tuple of (FileMetadata, list of ParticleData).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a valid SMS HDF5 file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    particles: List[ParticleData] = []
    has_spectra = False
    has_raster = False

    with h5py.File(path, "r") as h5file:
        # Check for required attributes
        if "# Particles" not in h5file.attrs:
            raise ValueError(
                f"Invalid SMS HDF5 file: missing '# Particles' attribute: {path}"
            )

        version = _get_file_version(h5file)
        num_particles = int(h5file.attrs["# Particles"])
        particle_names = _get_particle_names(h5file)

        # Validate particle count
        if len(particle_names) != num_particles:
            # Some files have mismatched counts; use actual number of groups
            num_particles = len(particle_names)

        for idx, part_name in enumerate(particle_names):
            if part_name not in h5file:
                continue

            particle_group = h5file[part_name]

            # Read primary channel
            channel1 = _read_channel_data(particle_group, version, is_secondary=False)
            if channel1 is None:
                # Skip particles without valid channel data
                continue

            # Read secondary channel if available
            channel2 = None
            if version not in _OLD_VERSIONS_NO_DUAL_CHANNEL:
                channel2 = _read_channel_data(
                    particle_group, version, is_secondary=True
                )

            # Get metadata
            description = _get_description(particle_group, version)
            tcspc_card = _get_tcspc_card(particle_group, version)
            channelwidth = _determine_channelwidth(channel1.microtimes)

            # Track file-level features
            if _has_spectra(particle_group):
                has_spectra = True
            if _has_raster_scan(particle_group):
                has_raster = True

            particle = ParticleData(
                id=idx + 1,  # 1-based particle IDs
                name=part_name,
                tcspc_card=tcspc_card,
                channelwidth=channelwidth,
                channel1=channel1,
                channel2=channel2,
                description=description,
            )
            particles.append(particle)

    metadata = FileMetadata(
        path=path,
        filename=path.name,
        num_particles=len(particles),
        has_irf=False,  # IRF is loaded separately
        has_spectra=has_spectra,
        has_raster=has_raster,
    )

    return metadata, particles


def load_irf(path: Path | str) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Load an IRF (Instrument Response Function) from a file.

    IRF files can be:
    - HDF5 files with the same format as measurement files
    - Text files with two columns (time, counts)

    Args:
        path: Path to the IRF file.

    Returns:
        Tuple of (time array in ns, counts array), or None if loading fails.
    """
    path = Path(path)
    if not path.exists():
        return None

    # Try loading as HDF5 first
    try:
        with h5py.File(path, "r") as h5file:
            # IRF files typically have a single particle
            particle_names = _get_particle_names(h5file)
            if not particle_names:
                return None

            version = _get_file_version(h5file)
            particle_group = h5file[particle_names[0]]
            channel = _read_channel_data(particle_group, version, is_secondary=False)

            if channel is None:
                return None

            # Build histogram from microtimes
            channelwidth = _determine_channelwidth(channel.microtimes)
            microtimes = channel.microtimes

            if len(microtimes) == 0:
                return None

            tmin = float(np.min(microtimes))
            tmax = float(np.max(microtimes))
            bins = np.arange(tmin, tmax + channelwidth, channelwidth)
            counts, edges = np.histogram(microtimes, bins=bins)
            t = edges[:-1]  # Use left edges as time values

            return t, counts.astype(np.float64)

    except (OSError, KeyError):
        pass

    # Try loading as text file
    try:
        data = np.loadtxt(path)
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 0], data[:, 1]
    except Exception:
        pass

    return None
