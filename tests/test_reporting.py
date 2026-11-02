import numpy as np
import pytest

from tin_whistle_openwind.geometry import MouthpieceGeometry, TaperedBoreGeometry, UndercutTonehole
from tin_whistle_openwind.keys import SUPPORTED_KEYS
from tin_whistle_openwind.optimizer import ACTIVE_HOLE_LABELS, ToneholeLayout
from tin_whistle_openwind.reporting import drilling_chart_rows, fingering_tuning_rows
from tin_whistle_openwind.tuning import IRISH_FOLK_HYBRID, generate_key_frequencies


@pytest.fixture
def layout():
    return ToneholeLayout([
        UndercutTonehole(label, 0.1 + index * 0.02, 0.006, 0.003)
        for index, label in enumerate(ACTIVE_HOLE_LABELS)
    ])


def test_drilling_chart_uses_pipe_coordinates_and_millimeters(layout):
    rows = drilling_chart_rows(layout, 0.25)
    assert len(rows) == 6
    assert rows[0]["Position from pipe start (mm)"] == pytest.approx(100.0)
    assert rows[0]["Spacing (mm)"] is None
    assert rows[1]["Spacing (mm)"] == pytest.approx(20.0)
    assert rows[-1]["Distance from end (mm)"] == pytest.approx(50.0)
    assert rows[0]["Drill diameter (mm)"] == pytest.approx(6.0)


def test_both_registers_use_measured_peaks_and_vented_upper_root(layout, monkeypatch):
    calls = []

    def impedance(model, frequencies, note):
        calls.append((model, note))
        target = (frequencies[0] + frequencies[-1]) / 2.0
        # A distinct +12 cent upper-register resonance must not be derived by doubling.
        center = target * 2.0 ** ((12.0 if len(calls) > 8 else -3.0) / 1200.0)
        return (1.0 + ((frequencies - center) / 2.0) ** 2).astype(complex)

    monkeypatch.setattr("tin_whistle_openwind.reporting.OpenWInDToneholeModel.impedance", impedance)
    mouthpiece = MouthpieceGeometry.load("MouthpieceDN16V2")
    bore = TaperedBoreGeometry.cylindrical("test", 0.0065, 0.25)
    rows = fingering_tuning_rows(
        SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID, bore, layout, mouthpiece
    )
    assert len(rows) == 16
    assert rows[0]["Deviation (cents)"] == pytest.approx(-3.0, abs=0.1)
    assert rows[8]["Deviation (cents)"] == pytest.approx(12.0, abs=0.1)
    assert rows[0]["L1"] == "x"
    assert rows[8]["L1"] == "o"
    assert calls[8][1] == "Octave_Degree_1"
    assert calls[0][0].hole_position_offset_m == mouthpiece.l_mouth
    assert calls[0][0].bore.bore[-1][0] == pytest.approx(mouthpiece.l_mouth + bore.length_m)
    targets = generate_key_frequencies(SUPPORTED_KEYS["Soprano_D"], IRISH_FOLK_HYBRID)
    assert rows[9]["Target (Hz)"] == pytest.approx(2 * targets["Degree_2"])
    assert sum(row["Degree"] == "Fork_Minor7" for row in rows) == 2
    assert all(row["Status"] == "OK" for row in rows)


def test_missing_resonance_is_not_reported_as_a_frequency(layout, monkeypatch):
    monkeypatch.setattr(
        "tin_whistle_openwind.reporting.OpenWInDToneholeModel.impedance",
        lambda model, frequencies, note: np.ones_like(frequencies, dtype=complex),
    )
    rows = fingering_tuning_rows(
        SUPPORTED_KEYS["Soprano_D"],
        IRISH_FOLK_HYBRID,
        TaperedBoreGeometry.cylindrical("test", 0.0065, 0.25),
        layout,
        MouthpieceGeometry.load("MouthpieceDN16V2"),
    )
    assert all(row["Simulated (Hz)"] is None for row in rows)
    assert all(row["Deviation (cents)"] is None for row in rows)
    assert all(row["Status"] == "No resonance in search window" for row in rows)


def test_invalid_impedance_raises_instead_of_showing_a_successful_report(layout, monkeypatch):
    monkeypatch.setattr(
        "tin_whistle_openwind.reporting.OpenWInDToneholeModel.impedance",
        lambda model, frequencies, note: np.full_like(frequencies, np.nan, dtype=complex),
    )
    with pytest.raises(RuntimeError, match="Invalid impedance"):
        fingering_tuning_rows(
            SUPPORTED_KEYS["Soprano_D"],
            IRISH_FOLK_HYBRID,
            TaperedBoreGeometry.cylindrical("test", 0.0065, 0.25),
            layout,
            MouthpieceGeometry.load("MouthpieceDN16V2"),
        )
