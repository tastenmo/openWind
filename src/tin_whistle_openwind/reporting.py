"""Bohrplan and two-register acoustic evaluation for notebook tables."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .geometry import MouthpieceGeometry, TaperedBoreGeometry, WhistleGeometryFactory
from .keys import WhistleKeyConfig
from .optimizer import OpenWInDToneholeModel, ToneholeLayout, find_play_resonance
from .tuning import STANDARD_FINGERINGS, TuningProfile, generate_key_frequencies


def drilling_chart_rows(
    layout: ToneholeLayout, pipe_length_m: float
) -> list[dict[str, str | float | int | None]]:
    """Return unrounded manufacturing dimensions, relative to the pipe start."""
    rows: list[dict[str, str | float | int | None]] = []
    previous_position = None
    for index, hole in enumerate(layout.holes, start=1):
        rows.append(
            {
                "Hole name": hole.label,
                "Hole": index,
                "Position from pipe start (mm)": float(hole.position_m * 1000.0),
                "Spacing (mm)": (
                    None if previous_position is None
                    else float((hole.position_m - previous_position) * 1000.0)
                ),
                "Distance from end (mm)": float((pipe_length_m - hole.position_m) * 1000.0),
                "Drill diameter (mm)": float(hole.diameter_m * 1000.0),
            }
        )
        previous_position = hole.position_m
    return rows


def fingering_tuning_rows(
    key_config: WhistleKeyConfig,
    tuning: TuningProfile,
    bore: TaperedBoreGeometry,
    layout: ToneholeLayout,
    mouthpiece: MouthpieceGeometry,
    temperature: float = 20.0,
) -> list[dict[str, str | float | int | None]]:
    """Evaluate actual admittance peaks in both registers, including the fork.

    The upper root uses the existing vented fingering; the other upper-register
    notes use the corresponding lower-register fingering at twice the target.
    No third-octave root is added. Missing peaks are reported, not substituted
    with a sweep endpoint.
    """
    instrument = WhistleGeometryFactory(temperature=temperature).create_tapered_instrument(
        mouthpiece, bore
    )
    model = OpenWInDToneholeModel(
        instrument,
        layout,
        temperature,
        hole_position_offset_m=mouthpiece.l_mouth,
    )
    targets = generate_key_frequencies(key_config, tuning)
    rows: list[dict[str, str | float | int | None]] = []
    for register in (1, 2):
        for degree, base_target in targets.items():
            if degree == "Octave_Degree_1":
                continue
            fingering_name = (
                "Octave_Degree_1" if register == 2 and degree == "Degree_1" else degree
            )
            target = (
                targets["Octave_Degree_1"]
                if register == 2 and degree == "Degree_1" else base_target * register
            )
            frequencies = np.linspace(0.8 * target, 1.2 * target, 801)
            impedance = model.impedance(frequencies, fingering_name)
            magnitude = np.abs(impedance)
            if not np.all(np.isfinite(magnitude)) or np.any(magnitude <= 0):
                raise RuntimeError(f"Invalid impedance for {degree}, register {register}.")
            peaks, _ = find_peaks(1.0 / magnitude)
            frequency = (
                find_play_resonance(impedance, frequencies, target) if len(peaks) else None
            )
            rows.append(
                {
                    "Register": register,
                    "Degree": degree,
                    **{
                        hole.label: "x" if closed else "o"
                        for hole, closed in zip(
                            layout.holes, STANDARD_FINGERINGS[fingering_name]
                        )
                    },
                    "Target (Hz)": float(target),
                    "Simulated (Hz)": frequency,
                    "Deviation (cents)": (
                        float(1200.0 * np.log2(frequency / target))
                        if frequency is not None else None
                    ),
                    "Status": "OK" if frequency is not None else "No resonance in search window",
                }
            )
    return rows
