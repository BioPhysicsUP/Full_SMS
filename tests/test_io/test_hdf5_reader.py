"""Tests for HDF5 file reader."""

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest

from full_sms.io.hdf5_reader import (
    ensure_analysis_uuid,
    load_h5_file,
    load_irf,
    _get_file_version,
    _get_particle_names,
    _get_description,
    _determine_channelwidth,
    _read_channel_data,
    _has_spectra,
    _has_raster_scan,
)
from full_sms.models.particle import ChannelData, ParticleData
from full_sms.models.session import FileMetadata


class TestGetFileVersion:
    """Tests for _get_file_version."""

    def test_returns_version_when_present(self, tmp_path: Path) -> None:
        """Should return version string when Version attribute exists."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["Version"] = "1.07"
            f.attrs["# Particles"] = 0

        with h5py.File(filepath, "r") as f:
            assert _get_file_version(f) == "1.07"

    def test_returns_default_when_missing(self, tmp_path: Path) -> None:
        """Should return '1.0' when Version attribute is missing."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 0

        with h5py.File(filepath, "r") as f:
            assert _get_file_version(f) == "1.0"


class TestGetParticleNames:
    """Tests for _get_particle_names."""

    def test_returns_sorted_particle_names(self, tmp_path: Path) -> None:
        """Should return particle names in natural numeric order."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 3
            f.create_group("Particle 2")
            f.create_group("Particle 1")
            f.create_group("Particle 10")
            f.create_group("Other Group")  # Should be ignored

        with h5py.File(filepath, "r") as f:
            names = _get_particle_names(f)
            assert names == ["Particle 1", "Particle 2", "Particle 10"]

    def test_returns_empty_list_when_no_particles(self, tmp_path: Path) -> None:
        """Should return empty list when no particle groups exist."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 0

        with h5py.File(filepath, "r") as f:
            names = _get_particle_names(f)
            assert names == []


class TestGetDescription:
    """Tests for _get_description."""

    def test_reads_description_new_format(self, tmp_path: Path) -> None:
        """Should read Description attribute for newer versions."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("Particle 1")
            grp.attrs["Description"] = "Test particle"

        with h5py.File(filepath, "r") as f:
            desc = _get_description(f["Particle 1"], "1.07")
            assert desc == "Test particle"

    def test_reads_description_old_format_typo(self, tmp_path: Path) -> None:
        """Should read Discription (typo) for old versions."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("Particle 1")
            grp.attrs["Discription"] = "Old format particle"

        with h5py.File(filepath, "r") as f:
            desc = _get_description(f["Particle 1"], "1.0")
            assert desc == "Old format particle"

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        """Should return empty string when attribute is missing."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("Particle 1")

        with h5py.File(filepath, "r") as f:
            desc = _get_description(f["Particle 1"], "1.07")
            assert desc == ""


class TestDetermineChannelwidth:
    """Tests for _determine_channelwidth."""

    def test_determines_channelwidth_from_microtimes(self) -> None:
        """Should determine channel width from microtime differences."""
        # Create microtimes with regular spacing of 0.05 ns
        channelwidth = 0.05
        microtimes = np.arange(0, 25, channelwidth)
        result = _determine_channelwidth(microtimes)
        assert np.isclose(result, channelwidth, rtol=0.1)

    def test_returns_default_for_empty_array(self) -> None:
        """Should return default channel width for empty array."""
        microtimes = np.array([])
        result = _determine_channelwidth(microtimes)
        assert np.isclose(result, 0.01220703125)

    def test_handles_irregular_microtimes(self) -> None:
        """Should handle microtimes with some irregularity."""
        # Simulate realistic data with base channelwidth of 0.012
        base_width = 0.012
        microtimes = np.array([0, 0.012, 0.024, 0.036, 0.048, 0.072, 0.084])
        result = _determine_channelwidth(microtimes)
        assert result > 0


class TestReadChannelData:
    """Tests for _read_channel_data."""

    def test_reads_primary_channel(self, tmp_path: Path) -> None:
        """Should read primary channel data."""
        filepath = tmp_path / "test.h5"
        abstimes = np.array([0, 1000, 2000, 3000], dtype=np.uint64)
        microtimes = np.array([0.5, 1.2, 0.8, 1.5], dtype=np.float64)

        with h5py.File(filepath, "w") as f:
            grp = f.create_group("Particle 1")
            grp.create_dataset("Absolute Times (ns)", data=abstimes)
            grp.create_dataset("Micro Times (ns)", data=microtimes)

        with h5py.File(filepath, "r") as f:
            channel = _read_channel_data(f["Particle 1"], "1.07", is_secondary=False)
            assert channel is not None
            np.testing.assert_array_equal(channel.abstimes, abstimes)
            np.testing.assert_array_equal(channel.microtimes, microtimes)

    def test_reads_secondary_channel(self, tmp_path: Path) -> None:
        """Should read secondary channel data."""
        filepath = tmp_path / "test.h5"
        abstimes = np.array([100, 1100, 2100], dtype=np.uint64)
        microtimes = np.array([0.3, 1.1, 0.7], dtype=np.float64)

        with h5py.File(filepath, "w") as f:
            grp = f.create_group("Particle 1")
            grp.create_dataset("Absolute Times 2 (ns)", data=abstimes)
            grp.create_dataset("Micro Times 2 (ns)", data=microtimes)

        with h5py.File(filepath, "r") as f:
            channel = _read_channel_data(f["Particle 1"], "1.07", is_secondary=True)
            assert channel is not None
            np.testing.assert_array_equal(channel.abstimes, abstimes)
            np.testing.assert_array_equal(channel.microtimes, microtimes)

    def test_returns_none_when_channel_missing(self, tmp_path: Path) -> None:
        """Should return None when channel data is missing."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("Particle 1")

        with h5py.File(filepath, "r") as f:
            channel = _read_channel_data(f["Particle 1"], "1.07", is_secondary=False)
            assert channel is None

    def test_converts_old_format_seconds_to_ns(self, tmp_path: Path) -> None:
        """Should convert microtimes from seconds to nanoseconds for old versions."""
        filepath = tmp_path / "test.h5"
        abstimes = np.array([0, 1000000000], dtype=np.uint64)
        microtimes_seconds = np.array([0.5e-9, 1.2e-9], dtype=np.float64)

        with h5py.File(filepath, "w") as f:
            grp = f.create_group("Particle 1")
            grp.create_dataset("Absolute Times (ns)", data=abstimes)
            grp.create_dataset("Micro Times (s)", data=microtimes_seconds)

        with h5py.File(filepath, "r") as f:
            channel = _read_channel_data(f["Particle 1"], "1.0", is_secondary=False)
            assert channel is not None
            # Should be converted to nanoseconds
            expected_ns = microtimes_seconds * 1e9
            np.testing.assert_array_almost_equal(channel.microtimes, expected_ns)


class TestHasSpectra:
    """Tests for _has_spectra."""

    def test_returns_true_when_spectra_present(self, tmp_path: Path) -> None:
        """Should return True when spectra dataset exists."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("Particle 1")
            grp.create_dataset("Spectra (counts\\s)", data=np.zeros((10, 100)))

        with h5py.File(filepath, "r") as f:
            assert _has_spectra(f["Particle 1"]) is True

    def test_returns_false_when_spectra_missing(self, tmp_path: Path) -> None:
        """Should return False when spectra dataset is missing."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("Particle 1")

        with h5py.File(filepath, "r") as f:
            assert _has_spectra(f["Particle 1"]) is False


class TestHasRasterScan:
    """Tests for _has_raster_scan."""

    def test_returns_true_when_raster_present(self, tmp_path: Path) -> None:
        """Should return True when Raster Scan group exists."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("Particle 1")
            grp.create_group("Raster Scan")

        with h5py.File(filepath, "r") as f:
            assert _has_raster_scan(f["Particle 1"]) is True

    def test_returns_false_when_raster_missing(self, tmp_path: Path) -> None:
        """Should return False when Raster Scan group is missing."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("Particle 1")

        with h5py.File(filepath, "r") as f:
            assert _has_raster_scan(f["Particle 1"]) is False


class TestLoadH5File:
    """Tests for load_h5_file."""

    def test_loads_single_particle_file(self, tmp_path: Path) -> None:
        """Should load a file with a single particle."""
        filepath = tmp_path / "single.h5"
        abstimes = np.array([0, 1000000, 2000000], dtype=np.uint64)
        microtimes = np.array([0.5, 1.2, 0.8], dtype=np.float64)

        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 1
            f.attrs["Version"] = "1.07"
            grp = f.create_group("Particle 1")
            grp.attrs["Description"] = "Test"
            ds = grp.create_dataset("Absolute Times (ns)", data=abstimes)
            ds.attrs["bh Card"] = "SPC-150"
            grp.create_dataset("Micro Times (ns)", data=microtimes)

        metadata, particles = load_h5_file(filepath)

        assert isinstance(metadata, FileMetadata)
        assert metadata.num_particles == 1
        assert metadata.filename == "single.h5"
        assert metadata.has_spectra is False
        assert metadata.has_raster is False

        assert len(particles) == 1
        p = particles[0]
        assert p.id == 1
        assert p.name == "Particle 1"
        assert p.description == "Test"
        assert p.tcspc_card == "SPC-150"
        np.testing.assert_array_equal(p.channel1.abstimes, abstimes)
        np.testing.assert_array_equal(p.channel1.microtimes, microtimes)
        assert p.channel2 is None

    def test_loads_multi_particle_file(self, tmp_path: Path) -> None:
        """Should load a file with multiple particles."""
        filepath = tmp_path / "multi.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 3
            f.attrs["Version"] = "1.07"

            for i in range(1, 4):
                abstimes = np.array([0, 1000 * i, 2000 * i], dtype=np.uint64)
                microtimes = np.array([0.1 * i, 0.2 * i, 0.3 * i], dtype=np.float64)
                grp = f.create_group(f"Particle {i}")
                grp.attrs["Description"] = f"Particle {i} description"
                grp.create_dataset("Absolute Times (ns)", data=abstimes)
                grp.create_dataset("Micro Times (ns)", data=microtimes)

        metadata, particles = load_h5_file(filepath)

        assert metadata.num_particles == 3
        assert len(particles) == 3

        for i, p in enumerate(particles, 1):
            assert p.id == i
            assert p.name == f"Particle {i}"
            assert p.description == f"Particle {i} description"

    def test_loads_dual_channel_particle(self, tmp_path: Path) -> None:
        """Should load particle with dual TCSPC channels."""
        filepath = tmp_path / "dual.h5"
        abstimes1 = np.array([0, 1000, 2000], dtype=np.uint64)
        microtimes1 = np.array([0.5, 1.2, 0.8], dtype=np.float64)
        abstimes2 = np.array([100, 1100, 2100], dtype=np.uint64)
        microtimes2 = np.array([0.6, 1.3, 0.9], dtype=np.float64)

        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 1
            f.attrs["Version"] = "1.07"
            grp = f.create_group("Particle 1")
            grp.create_dataset("Absolute Times (ns)", data=abstimes1)
            grp.create_dataset("Micro Times (ns)", data=microtimes1)
            grp.create_dataset("Absolute Times 2 (ns)", data=abstimes2)
            grp.create_dataset("Micro Times 2 (ns)", data=microtimes2)

        metadata, particles = load_h5_file(filepath)

        assert len(particles) == 1
        p = particles[0]
        assert p.has_dual_channel is True
        np.testing.assert_array_equal(p.channel1.abstimes, abstimes1)
        np.testing.assert_array_equal(p.channel2.abstimes, abstimes2)

    def test_detects_spectra_and_raster(self, tmp_path: Path) -> None:
        """Should detect spectra and raster scan data."""
        filepath = tmp_path / "features.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 1
            f.attrs["Version"] = "1.07"
            grp = f.create_group("Particle 1")
            grp.create_dataset(
                "Absolute Times (ns)", data=np.array([0, 1000], dtype=np.uint64)
            )
            grp.create_dataset(
                "Micro Times (ns)", data=np.array([0.5, 1.2], dtype=np.float64)
            )
            grp.create_dataset("Spectra (counts\\s)", data=np.zeros((10, 100)))
            grp.create_group("Raster Scan")

        metadata, particles = load_h5_file(filepath)

        assert metadata.has_spectra is True
        assert metadata.has_raster is True

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        filepath = tmp_path / "nonexistent.h5"
        with pytest.raises(FileNotFoundError):
            load_h5_file(filepath)

    def test_raises_on_invalid_file(self, tmp_path: Path) -> None:
        """Should raise ValueError for file without # Particles attribute."""
        filepath = tmp_path / "invalid.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("SomeData")

        with pytest.raises(ValueError, match="missing '# Particles' attribute"):
            load_h5_file(filepath)

    def test_handles_old_version_format(self, tmp_path: Path) -> None:
        """Should handle old version files correctly."""
        filepath = tmp_path / "old.h5"
        abstimes = np.array([0, 1000000000], dtype=np.uint64)
        microtimes_seconds = np.array([0.5e-9, 1.2e-9], dtype=np.float64)

        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 1
            # No Version attribute - should default to 1.0
            grp = f.create_group("Particle 1")
            grp.attrs["Discription"] = "Old format"  # Old typo
            grp.create_dataset("Absolute Times (ns)", data=abstimes)
            grp.create_dataset("Micro Times (s)", data=microtimes_seconds)

        metadata, particles = load_h5_file(filepath)

        assert len(particles) == 1
        p = particles[0]
        assert p.description == "Old format"
        # Microtimes should be converted to nanoseconds
        expected_ns = microtimes_seconds * 1e9
        np.testing.assert_array_almost_equal(p.channel1.microtimes, expected_ns)

    def test_skips_particles_without_channel_data(self, tmp_path: Path) -> None:
        """Should skip particles that don't have channel data."""
        filepath = tmp_path / "partial.h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 2
            f.attrs["Version"] = "1.07"

            # Particle 1 has data
            grp1 = f.create_group("Particle 1")
            grp1.create_dataset(
                "Absolute Times (ns)", data=np.array([0, 1000], dtype=np.uint64)
            )
            grp1.create_dataset(
                "Micro Times (ns)", data=np.array([0.5, 1.2], dtype=np.float64)
            )

            # Particle 2 is empty
            f.create_group("Particle 2")

        metadata, particles = load_h5_file(filepath)

        assert metadata.num_particles == 1
        assert len(particles) == 1
        assert particles[0].name == "Particle 1"


class TestLoadIrf:
    """Tests for load_irf."""

    def test_loads_irf_from_h5_file(self, tmp_path: Path) -> None:
        """Should load IRF from HDF5 file."""
        filepath = tmp_path / "irf.h5"
        microtimes = np.array([0.1, 0.2, 0.1, 0.15, 0.12, 0.18, 0.11], dtype=np.float64)
        abstimes = np.arange(len(microtimes), dtype=np.uint64)

        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 1
            f.attrs["Version"] = "1.07"
            grp = f.create_group("Particle 1")
            grp.create_dataset("Absolute Times (ns)", data=abstimes)
            grp.create_dataset("Micro Times (ns)", data=microtimes)

        result = load_irf(filepath)

        assert result is not None
        t, counts = result
        assert len(t) == len(counts)
        assert np.sum(counts) > 0

    def test_loads_irf_from_text_file(self, tmp_path: Path) -> None:
        """Should load IRF from text file with two columns."""
        filepath = tmp_path / "irf.txt"
        t = np.linspace(0, 25, 100)
        counts = np.exp(-((t - 5) ** 2) / 2)  # Gaussian-like IRF

        np.savetxt(filepath, np.column_stack([t, counts]))

        result = load_irf(filepath)

        assert result is not None
        loaded_t, loaded_counts = result
        np.testing.assert_array_almost_equal(loaded_t, t)
        np.testing.assert_array_almost_equal(loaded_counts, counts)

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Should return None for missing file."""
        filepath = tmp_path / "nonexistent.h5"
        result = load_irf(filepath)
        assert result is None

    def test_returns_none_for_invalid_file(self, tmp_path: Path) -> None:
        """Should return None for files that can't be parsed."""
        filepath = tmp_path / "invalid.txt"
        filepath.write_text("not a valid data file")

        result = load_irf(filepath)
        assert result is None


class TestEnsureAnalysisUUID:
    """Tests for ensure_analysis_uuid function."""

    def test_creates_uuid_when_missing(self, tmp_path: Path) -> None:
        """Should create a new UUID when file has none."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 0

        result = ensure_analysis_uuid(filepath)

        assert result is not None
        assert len(result) == 36  # UUID format: 8-4-4-4-12

        # Verify it was written to the file
        with h5py.File(filepath, "r") as f:
            assert f.attrs["analysis_uuid"] == result

    def test_returns_existing_uuid(self, tmp_path: Path) -> None:
        """Should return existing UUID without modifying it."""
        filepath = tmp_path / "test.h5"
        existing_uuid = "existing-uuid-value"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 0
            f.attrs["analysis_uuid"] = existing_uuid

        result = ensure_analysis_uuid(filepath)

        assert result == existing_uuid

    def test_handles_readonly_file(self, tmp_path: Path) -> None:
        """Should return existing UUID from read-only file."""
        filepath = tmp_path / "test.h5"
        existing_uuid = "readonly-uuid"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 0
            f.attrs["analysis_uuid"] = existing_uuid

        # Make file read-only
        import os
        os.chmod(filepath, 0o444)
        try:
            result = ensure_analysis_uuid(filepath)
            assert result == existing_uuid
        finally:
            os.chmod(filepath, 0o644)

    def test_returns_none_for_readonly_without_uuid(self, tmp_path: Path) -> None:
        """Should return None for read-only file with no UUID."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 0

        # Make file read-only
        import os
        os.chmod(filepath, 0o444)
        try:
            result = ensure_analysis_uuid(filepath)
            assert result is None
        finally:
            os.chmod(filepath, 0o644)

    def test_idempotent(self, tmp_path: Path) -> None:
        """Calling twice returns the same UUID."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["# Particles"] = 0

        first = ensure_analysis_uuid(filepath)
        second = ensure_analysis_uuid(filepath)

        assert first == second
