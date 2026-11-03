import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tin_whistle_openwind.geometry import MouthpieceGeometry, TaperedBoreGeometry, UndercutTonehole
from tin_whistle_openwind.optimizer import ACTIVE_HOLE_LABELS, ToneholeLayout
from tin_whistle_openwind.resonances import (
    compare_fingering_resonances,
    plot_resonance_comparison,
    resonance_peak_rows,
)
from tin_whistle_openwind.tuning import STANDARD_FINGERINGS


@pytest.mark.parametrize("quantity", ["Z", "Y"])
def test_half_power_q_of_known_resonance(quantity):
    frequencies = np.arange(400.0, 600.01, 0.1)
    # Lorentzian amplitude: f0=500 Hz, FWHM of power=10 Hz, hence Q=50.
    response = 2e6 / (1 + 1j * (frequencies - 500) / 5)
    impedance = response if quantity == "Z" else 1 / response
    row, = resonance_peak_rows(frequencies, impedance, quantity)
    assert row["Peak (Hz)"] == pytest.approx(500, abs=0.01)
    assert row["Peak amplitude"] == pytest.approx(2e6)
    assert row["Lower -3 dB (Hz)"] == pytest.approx(495, abs=0.01)
    assert row["Upper -3 dB (Hz)"] == pytest.approx(505, abs=0.01)
    assert row["Bandwidth (Hz)"] == pytest.approx(10, abs=0.01)
    assert row["Q"] == pytest.approx(50, abs=0.05)
    assert row["Status"] == "OK"


def test_grid_refinement_converges_for_off_grid_peak():
    rows = []
    for step in (0.25, 0.125):
        frequencies = np.arange(400, 600, step)
        impedance = 1 / (1 + 1j * (frequencies - 500.07) / 5)
        rows.append(resonance_peak_rows(frequencies, impedance, "Z")[0])
    assert rows[0]["Q"] == pytest.approx(rows[1]["Q"], rel=0.001)
    assert rows[0]["Peak (Hz)"] == pytest.approx(500.07, abs=0.001)


def test_missing_half_power_crossings_are_not_fabricated():
    frequencies = np.linspace(490, 510, 101)
    impedance = 1 / (1 + 1j * (frequencies - 500) / 20)
    row, = resonance_peak_rows(frequencies, impedance, "Z")
    assert row["Q"] is None
    assert row["Bandwidth (Hz)"] is None
    assert "unavailable" in row["Status"]


def test_narrow_peaks_require_grid_refinement():
    frequencies = np.arange(490.0, 511.0)
    row, = resonance_peak_rows(frequencies, 1 / (1 + 1j * (frequencies - 500)), "Z")
    assert row["Q"] is None
    assert row["Status"] == "Unresolved bandwidth: refine grid"


def test_overlapping_peaks_do_not_borrow_crossings_from_other_modes():
    frequencies = np.linspace(400, 600, 2001)
    amplitude = (
        1 + 0.2 * np.exp(-((frequencies - 480) / 5) ** 2)
        + 0.3 * np.exp(-((frequencies - 520) / 5) ** 2)
    )
    rows = resonance_peak_rows(frequencies, amplitude, "Z")
    assert len(rows) == 2
    assert all(row["Q"] is None for row in rows)


@pytest.mark.parametrize("frequencies,impedance", [
    (np.array([1, 2, 4]), np.ones(3)),
    (np.array([3, 2, 1]), np.ones(3)),
    (np.arange(1, 4), np.array([1, np.nan, 1])),
    (np.arange(1, 4), np.array([1, 0, 1])),
    (np.arange(1, 4), np.ones(2)),
])
def test_invalid_sweeps_raise(frequencies, impedance):
    with pytest.raises(ValueError):
        resonance_peak_rows(frequencies, impedance, "Z")


def test_comparison_covers_all_grips_and_both_quantities(monkeypatch):
    calls = []

    def impedance(model, frequencies, note):
        calls.append((model, note))
        return 2e6 / (1 + 1j * (frequencies - 500) / 5)

    monkeypatch.setattr("tin_whistle_openwind.resonances.OpenWInDToneholeModel.impedance", impedance)
    layout = ToneholeLayout([
        UndercutTonehole(label, 0.1 + i * 0.02, 0.006, 0.003)
        for i, label in enumerate(ACTIVE_HOLE_LABELS)
    ])
    mouthpiece = MouthpieceGeometry.load("MouthpieceDN16V2")
    comparison = compare_fingering_resonances(
        TaperedBoreGeometry.cylindrical("DN16", 0.0071, 0.246),
        layout, mouthpiece, np.arange(400, 600, 0.25),
    )
    assert list(comparison.impedances) == list(STANDARD_FINGERINGS)
    assert len(calls) == 9
    assert len(comparison.peak_rows) == 18
    assert {row["Quantity"] for row in comparison.peak_rows} == {"Z", "Y"}
    assert comparison.peak_rows[0]["Holes (L1-R3)"] == "x x x x x x"
    assert calls[0][0].hole_position_offset_m == mouthpiece.l_mouth
    assert calls[0][0].bore.bore[-1][0] == pytest.approx(0.246 + mouthpiece.l_mouth)
    assert sum(row["Status"] == "No interior peak" for row in comparison.peak_rows) == 9
    assert all(row.keys() == comparison.peak_rows[0].keys() for row in comparison.peak_rows)
    figure = plot_resonance_comparison(comparison)
    assert len(figure.axes) == 6
    assert all(len(axis.lines) == 9 for axis in figure.axes[:2])
    assert len(figure.legends[0].texts) == 9
    import matplotlib.pyplot as plt
    plt.close(figure)
