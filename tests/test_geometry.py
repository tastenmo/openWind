"""Tests for tin_whistle_openwind.geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tin_whistle_openwind.geometry import (
    MouthpieceGeometry,
    PipeGeometry,
    WhistleGeometryFactory,
    WhistleInstrument,
)


@pytest.fixture
def sample_mouthpiece() -> MouthpieceGeometry:
    return MouthpieceGeometry(
        name="TestMouthpiece",
        segments=[(0.0, 0.0065), (0.00561, 0.0065)],
        l_mouth=0.021991,
    )


@pytest.fixture
def sample_pipe() -> PipeGeometry:
    return PipeGeometry(name="TestPipe", length_m=0.26361, radius_m=0.0072)


def test_exit_radius_returns_last_segment_radius(sample_mouthpiece):
    assert sample_mouthpiece.exit_radius == pytest.approx(0.0065)


def test_to_dict_serializes_segments_and_l_mouth(sample_mouthpiece):
    payload = sample_mouthpiece.to_dict()
    assert payload["name"] == "TestMouthpiece"
    assert payload["segments"] == [[0.0, 0.0065], [0.00561, 0.0065]]
    assert payload["l_mouth_m"] == pytest.approx(0.021991)


def test_save_writes_json_file(tmp_path: Path, sample_mouthpiece):
    path = sample_mouthpiece.save(tmp_path)
    assert path == tmp_path / "TestMouthpiece.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "TestMouthpiece"
    assert payload["l_mouth_m"] == pytest.approx(0.021991)


def test_save_creates_missing_data_dir(tmp_path: Path, sample_mouthpiece):
    nested_dir = tmp_path / "nested" / "data"
    sample_mouthpiece.save(nested_dir)
    assert (nested_dir / "TestMouthpiece.json").exists()


def test_load_round_trips_saved_mouthpiece(tmp_path: Path, sample_mouthpiece):
    sample_mouthpiece.save(tmp_path)
    loaded = MouthpieceGeometry.load("TestMouthpiece", tmp_path)
    assert loaded.name == sample_mouthpiece.name
    assert loaded.segments == sample_mouthpiece.segments
    assert loaded.l_mouth == pytest.approx(sample_mouthpiece.l_mouth)


def test_load_missing_file_raises_file_not_found_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MouthpieceGeometry.load("DoesNotExist", tmp_path)


def test_build_bore_appends_junction_step_and_pipe_end(sample_mouthpiece, sample_pipe):
    factory = WhistleGeometryFactory()
    bore = factory.build_bore(sample_mouthpiece, sample_pipe)
    assert bore[: len(sample_mouthpiece.segments)] == sample_mouthpiece.segments
    junction_position = sample_mouthpiece.segments[-1][0]
    assert bore[-2] == pytest.approx((junction_position, sample_pipe.radius_m))
    expected_last_position = sample_mouthpiece.l_mouth + sample_pipe.length_m
    assert bore[-1] == pytest.approx((expected_last_position, sample_pipe.radius_m))


def test_build_bore_loads_mouthpiece_by_name(tmp_path: Path, sample_mouthpiece, sample_pipe):
    sample_mouthpiece.save(tmp_path)
    factory = WhistleGeometryFactory(data_dir=tmp_path)
    bore = factory.build_bore(sample_mouthpiece.name, sample_pipe)
    assert bore[-1][1] == pytest.approx(sample_pipe.radius_m)


def test_build_bore_allows_mouthpiece_radius_smaller_than_pipe(sample_mouthpiece):
    wider_pipe = PipeGeometry(name="Wider", length_m=0.2, radius_m=0.008)
    factory = WhistleGeometryFactory()
    bore = factory.build_bore(sample_mouthpiece, wider_pipe)
    assert bore[-1][1] == pytest.approx(0.008)


def test_create_instrument_returns_whistle_instrument_with_expected_bore(sample_mouthpiece, sample_pipe):
    factory = WhistleGeometryFactory(temperature=18.0, radiation="infinite_flanged")
    instrument = factory.create_instrument(sample_mouthpiece, sample_pipe)
    assert isinstance(instrument, WhistleInstrument)
    assert instrument.radiation == "infinite_flanged"
    assert instrument.temperature == pytest.approx(18.0)
    assert instrument.bore == factory.build_bore(sample_mouthpiece, sample_pipe)


def test_whistle_instrument_get_impedance_returns_finite_complex_array(sample_mouthpiece, sample_pipe):
    factory = WhistleGeometryFactory()
    instrument = factory.create_instrument(sample_mouthpiece, sample_pipe)
    frequencies = np.linspace(400.0, 700.0, 5)
    impedance = instrument.get_impedance(frequencies)
    assert impedance.shape == frequencies.shape
    assert np.iscomplexobj(impedance)
    assert np.all(np.isfinite(impedance))
