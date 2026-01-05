"""Tests for particle and channel data models."""

import numpy as np
import pytest

from full_sms.models import ChannelData, ParticleData


class TestChannelData:
    """Tests for ChannelData."""

    def test_create_channel(self) -> None:
        """Can create a ChannelData with valid arrays."""
        abstimes = np.array([0, 100, 200, 300], dtype=np.uint64)
        microtimes = np.array([1.5, 2.3, 1.8, 2.1], dtype=np.float64)

        channel = ChannelData(abstimes=abstimes, microtimes=microtimes)

        assert channel.num_photons == 4
        np.testing.assert_array_equal(channel.abstimes, abstimes)
        np.testing.assert_array_equal(channel.microtimes, microtimes)

    def test_num_photons(self) -> None:
        """num_photons returns the correct count."""
        abstimes = np.array([0, 100, 200], dtype=np.uint64)
        microtimes = np.array([1.5, 2.3, 1.8], dtype=np.float64)

        channel = ChannelData(abstimes=abstimes, microtimes=microtimes)

        assert channel.num_photons == 3

    def test_measurement_time_s(self) -> None:
        """measurement_time_s calculates time in seconds."""
        # 1 second = 1e9 nanoseconds
        abstimes = np.array([0, 500_000_000, 1_000_000_000], dtype=np.uint64)
        microtimes = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        channel = ChannelData(abstimes=abstimes, microtimes=microtimes)

        assert channel.measurement_time_s == pytest.approx(1.0)

    def test_measurement_time_empty(self) -> None:
        """measurement_time_s returns 0 for empty channel."""
        channel = ChannelData(
            abstimes=np.array([], dtype=np.uint64),
            microtimes=np.array([], dtype=np.float64),
        )

        assert channel.measurement_time_s == 0.0

    def test_mismatched_lengths_raises(self) -> None:
        """Raises ValueError if abstimes and microtimes have different lengths."""
        abstimes = np.array([0, 100, 200], dtype=np.uint64)
        microtimes = np.array([1.5, 2.3], dtype=np.float64)

        with pytest.raises(ValueError, match="same length"):
            ChannelData(abstimes=abstimes, microtimes=microtimes)

    def test_immutable(self) -> None:
        """ChannelData is immutable (frozen dataclass)."""
        channel = ChannelData(
            abstimes=np.array([0, 100], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3], dtype=np.float64),
        )

        with pytest.raises(AttributeError):
            channel.abstimes = np.array([999], dtype=np.uint64)


class TestParticleData:
    """Tests for ParticleData."""

    @pytest.fixture
    def sample_channel(self) -> ChannelData:
        """Create a sample channel for testing."""
        return ChannelData(
            abstimes=np.array([0, 100_000_000, 200_000_000], dtype=np.uint64),
            microtimes=np.array([1.5, 2.3, 1.8], dtype=np.float64),
        )

    def test_create_particle(self, sample_channel: ChannelData) -> None:
        """Can create a ParticleData with required fields."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )

        assert particle.id == 1
        assert particle.name == "Particle 1"
        assert particle.tcspc_card == "SPC-150"
        assert particle.channelwidth == 0.012
        assert particle.channel1 is sample_channel
        assert particle.channel2 is None
        assert particle.description == ""

    def test_num_photons_single_channel(self, sample_channel: ChannelData) -> None:
        """num_photons returns count from single channel."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )

        assert particle.num_photons == 3

    def test_num_photons_dual_channel(self, sample_channel: ChannelData) -> None:
        """num_photons returns combined count from both channels."""
        channel2 = ChannelData(
            abstimes=np.array([50_000_000, 150_000_000], dtype=np.uint64),
            microtimes=np.array([1.2, 1.9], dtype=np.float64),
        )

        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
            channel2=channel2,
        )

        assert particle.num_photons == 5  # 3 + 2

    def test_has_dual_channel(self, sample_channel: ChannelData) -> None:
        """has_dual_channel indicates presence of second channel."""
        particle_single = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )

        particle_dual = ParticleData(
            id=2,
            name="Particle 2",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
            channel2=sample_channel,
        )

        assert particle_single.has_dual_channel is False
        assert particle_dual.has_dual_channel is True

    def test_measurement_time_s(self, sample_channel: ChannelData) -> None:
        """measurement_time_s uses primary channel time."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )

        # 200_000_000 ns = 0.2 s
        assert particle.measurement_time_s == pytest.approx(0.2)

    def test_convenience_accessors(self, sample_channel: ChannelData) -> None:
        """abstimes and microtimes properties access channel1 data."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
        )

        np.testing.assert_array_equal(particle.abstimes, sample_channel.abstimes)
        np.testing.assert_array_equal(particle.microtimes, sample_channel.microtimes)

    def test_with_description(self, sample_channel: ChannelData) -> None:
        """Can create a particle with a description."""
        particle = ParticleData(
            id=1,
            name="Particle 1",
            tcspc_card="SPC-150",
            channelwidth=0.012,
            channel1=sample_channel,
            description="Sample quantum dot",
        )

        assert particle.description == "Sample quantum dot"
