"""Impedance/admittance peak and half-power Q comparisons across fingerings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from scipy.signal import find_peaks

from .geometry import MouthpieceGeometry, TaperedBoreGeometry, WhistleGeometryFactory
from .optimizer import OpenWInDToneholeModel, ToneholeLayout
from .tuning import STANDARD_FINGERINGS

PeakRow = dict[str, str | float | int | None]


@dataclass
class ResonanceComparison:
    frequencies_hz: np.ndarray
    impedances: dict[str, np.ndarray]
    peak_rows: list[PeakRow]


def resonance_peak_rows(
    frequencies_hz: np.ndarray,
    impedance: np.ndarray,
    quantity: Literal["Z", "Y"],
) -> list[PeakRow]:
    """Measure all interior peaks using absolute half-power bandwidth.

    Q = f_peak / (f_right - f_left), at amplitude peak/sqrt(2), not half
    prominence. Crossings must belong to the same peak basin. Bandwidths with
    fewer than eight frequency intervals are flagged rather than reported as Q.
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    magnitude = np.abs(np.asarray(impedance, dtype=complex))
    if (
        frequencies.ndim != 1
        or len(frequencies) < 3
        or magnitude.shape != frequencies.shape
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0)
        or np.any(np.diff(frequencies) <= 0)
        or not np.allclose(np.diff(frequencies), np.diff(frequencies)[0])
    ):
        raise ValueError("A positive, increasing, uniform frequency grid is required.")
    if not np.all(np.isfinite(magnitude)) or np.any(magnitude <= 0):
        raise ValueError("Impedance must be finite and nonzero on the frequency grid.")
    if quantity not in ("Z", "Y"):
        raise ValueError("quantity must be 'Z' or 'Y'.")
    amplitude = magnitude if quantity == "Z" else 1.0 / magnitude
    peaks, _ = find_peaks(amplitude)
    step = frequencies[1] - frequencies[0]
    rows: list[PeakRow] = []
    for mode, peak in enumerate(peaks, start=1):
        left, center, right = amplitude[peak - 1:peak + 2]
        curvature = left - 2 * center + right
        offset = 0.5 * (left - right) / curvature if curvature != 0 else 0.0
        peak_frequency = float(frequencies[peak] + offset * step)
        peak_amplitude = float(center - 0.25 * (left - right) * offset)
        threshold = peak_amplitude / np.sqrt(2.0)
        start = int(peaks[mode - 2]) if mode > 1 else 0
        end = int(peaks[mode]) if mode < len(peaks) else len(amplitude) - 1
        left_valley = start + int(np.argmin(amplitude[start:peak + 1]))
        right_valley = peak + int(np.argmin(amplitude[peak:end + 1]))
        left_candidates = np.flatnonzero(amplitude[left_valley:peak] <= threshold)
        right_candidates = np.flatnonzero(amplitude[peak + 1:right_valley + 1] <= threshold)
        low = high = bandwidth = q = None
        status = "Half-power crossings unavailable in peak basin"
        if len(left_candidates) and len(right_candidates):
            low_index = left_valley + int(left_candidates[-1])
            high_index = peak + 1 + int(right_candidates[0])
            low = float(np.interp(
                threshold, amplitude[low_index:low_index + 2],
                frequencies[low_index:low_index + 2],
            ))
            high = float(np.interp(
                threshold, amplitude[high_index - 1:high_index + 1][::-1],
                frequencies[high_index - 1:high_index + 1][::-1],
            ))
            bandwidth = high - low
            status = "OK" if bandwidth >= 8 * step else "Unresolved bandwidth: refine grid"
            if status == "OK":
                q = peak_frequency / bandwidth
        rows.append({
            "Quantity": quantity,
            "Mode": mode,
            "Peak (Hz)": peak_frequency,
            "Peak amplitude": peak_amplitude,
            "Unit": "Pa s/m^3" if quantity == "Z" else "m^3/(Pa s)",
            "Lower -3 dB (Hz)": low,
            "Upper -3 dB (Hz)": high,
            "Bandwidth (Hz)": bandwidth,
            "Q": q,
            "Status": status,
        })
    return rows


def compare_fingering_resonances(
    bore: TaperedBoreGeometry,
    layout: ToneholeLayout,
    mouthpiece: MouthpieceGeometry,
    frequencies_hz: np.ndarray,
    temperature: float = 20.0,
) -> ResonanceComparison:
    """Simulate all nine standard grips, including fork and vented upper root."""
    instrument = WhistleGeometryFactory(temperature=temperature).create_tapered_instrument(
        mouthpiece, bore
    )
    model = OpenWInDToneholeModel(
        instrument, layout, temperature, hole_position_offset_m=mouthpiece.l_mouth
    )
    spectra: dict[str, np.ndarray] = {}
    rows: list[PeakRow] = []
    for note, fingering in STANDARD_FINGERINGS.items():
        impedance = model.impedance(frequencies_hz, note)
        spectra[note] = impedance
        quantities: tuple[Literal["Z", "Y"], ...] = ("Z", "Y")
        for quantity in quantities:
            peaks = resonance_peak_rows(frequencies_hz, impedance, quantity)
            if not peaks:
                peaks = [{
                    "Quantity": quantity, "Mode": None, "Peak (Hz)": None,
                    "Peak amplitude": None,
                    "Unit": "Pa s/m^3" if quantity == "Z" else "m^3/(Pa s)",
                    "Lower -3 dB (Hz)": None, "Upper -3 dB (Hz)": None,
                    "Bandwidth (Hz)": None, "Q": None, "Status": "No interior peak",
                }]
            rows.extend({
                "Fingering": note,
                "Holes (L1-R3)": " ".join("x" if closed else "o" for closed in fingering),
                **peak,
            } for peak in peaks)
    return ResonanceComparison(np.asarray(frequencies_hz), spectra, rows)


def plot_resonance_comparison(comparison: ResonanceComparison) -> Figure:
    """Plot spectra, absolute peak amplitudes and Q for both Z and Y."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for index, (note, impedance) in enumerate(comparison.impedances.items()):
        color = f"C{index}"
        for column, quantity in enumerate(("Z", "Y")):
            magnitude = np.abs(impedance)
            amplitude = magnitude if quantity == "Z" else 1.0 / magnitude
            axes[0, column].semilogy(
                comparison.frequencies_hz, amplitude, color=color, label=note
            )
            peaks = [
                row for row in comparison.peak_rows
                if row["Fingering"] == note and row["Quantity"] == quantity
                and row["Peak (Hz)"] is not None
            ]
            if peaks:
                axes[1, column].scatter(
                    [row["Peak (Hz)"] for row in peaks],
                    [row["Peak amplitude"] for row in peaks], color=color,
                )
            resolved = [row for row in peaks if row["Q"] is not None]
            if resolved:
                axes[2, column].scatter(
                    [row["Peak (Hz)"] for row in resolved],
                    [row["Q"] for row in resolved], color=color,
                )
    for column, (quantity, unit) in enumerate((("Z", "Pa s/m^3"), ("Y", "m^3/(Pa s)"))):
        axes[0, column].set_title(f"|{quantity}|: all standard fingerings")
        axes[0, column].set_ylabel(f"|{quantity}| [{unit}]")
        axes[1, column].set_ylabel(f"Peak |{quantity}| [{unit}]")
        if axes[1, column].collections:
            axes[1, column].set_yscale("log")
        axes[2, column].set_ylabel("Q (half-power bandwidth)")
        axes[2, column].set_xlabel("Frequency [Hz]")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
        if not axis.lines and not axis.collections:
            axis.text(
                0.5, 0.5, "No peaks / no resolved Q", ha="center",
                transform=axis.transAxes,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure
